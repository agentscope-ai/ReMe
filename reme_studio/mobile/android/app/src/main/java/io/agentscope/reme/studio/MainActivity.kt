package io.agentscope.reme.studio

import android.Manifest
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.net.Uri
import android.os.Bundle
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.graphics.Insets
import androidx.core.net.toUri
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsAnimationCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import org.json.JSONObject

class MainActivity : ComponentActivity() {
    private lateinit var webView: WebView
    private var rendererExited = false
    private var studioLoaded = false
    private val recorder = AndroidRecorder()
    private var pendingAudio: ((Boolean) -> Unit)? = null
    private var safeArea = Insets.NONE
    private var backGestureInset = 0
    private var keyboardVisible = false

    private val microphone = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val complete = pendingAudio
        pendingAudio = null
        complete?.invoke(granted)
    }

    @SuppressLint("SetJavaScriptEnabled") // The Studio application is loaded from bundled assets.
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val transparent = android.graphics.Color.TRANSPARENT
        enableEdgeToEdge(
            SystemBarStyle.auto(transparent, transparent),
            SystemBarStyle.auto(transparent, transparent),
        )
        window.isNavigationBarContrastEnforced = false
        val surface = FrameLayout(this).apply { setBackgroundResource(R.color.studio_surface) }
        webView = WebView(this).apply {
            setBackgroundColor(transparent)
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                // Narration starts when synthesized audio arrives, asynchronously from the UI.
                mediaPlaybackRequiresUserGesture = false
                allowFileAccess = false
                allowContentAccess = false
            }
        }
        surface.addView(webView, FrameLayout.LayoutParams(-1, -1))
        val safeTypes = WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
        val imeType = WindowInsetsCompat.Type.ime()
        var keyboardBottom = 0
        var keyboardAnimation: WindowInsetsAnimationCompat? = null
        ViewCompat.setWindowInsetsAnimationCallback(
            surface,
            object : WindowInsetsAnimationCompat.Callback(DISPATCH_MODE_STOP) {
                override fun onPrepare(animation: WindowInsetsAnimationCompat) {
                    if (animation.typeMask and imeType != 0) keyboardAnimation = animation
                }

                override fun onProgress(
                    insets: WindowInsetsCompat,
                    runningAnimations: MutableList<WindowInsetsAnimationCompat>,
                ): WindowInsetsCompat {
                    if (keyboardAnimation in runningAnimations)
                        surface.setPadding(0, 0, 0, insets.getInsets(imeType).bottom)
                    return insets
                }

                override fun onEnd(animation: WindowInsetsAnimationCompat) {
                    if (keyboardAnimation === animation) {
                        keyboardAnimation = null
                        surface.setPadding(0, 0, 0, keyboardBottom)
                    }
                }
            },
        )
        ViewCompat.setOnApplyWindowInsetsListener(surface) { view, insets ->
            keyboardBottom = insets.getInsets(imeType).bottom
            val system = insets.getInsets(safeTypes)
            safeArea = Insets.of(system.left, system.top, system.right, maxOf(0, system.bottom - keyboardBottom))
            backGestureInset = insets.getInsets(WindowInsetsCompat.Type.systemGestures()).left
            keyboardVisible = insets.isVisible(imeType)
            if (keyboardAnimation == null) view.setPadding(0, 0, 0, keyboardBottom)
            publishWindowInsets()
            // The page applies the safe area in CSS. Avoid applying it again in WebView.
            WindowInsetsCompat.Builder(insets).setInsets(safeTypes or imeType, Insets.NONE).build()
        }
        setContentView(surface)

        val assets = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()
        webView.webViewClient = object : WebViewClientCompat() {
            override fun onPageFinished(view: WebView, url: String) {
                studioLoaded = isStudioOrigin(url.toUri())
                publishWindowInsets()
            }

            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                assets.shouldInterceptRequest(request.url)

            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                if (!request.isForMainFrame) return false
                if (isStudioOrigin(request.url)) return false
                if (request.hasGesture()) openExternal(request.url)
                return true
            }

            override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
                rendererExited = true
                pendingAudio = null
                recorder.cancel()
                disposeWebView(view)
                recreate()
                return true
            }
        }
        if (WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) {
            WebViewCompat.addWebMessageListener(webView, "RemeAndroid", setOf(STUDIO_ORIGIN)) { _, message, origin, mainFrame, reply ->
                if (mainFrame && isStudioOrigin(origin)) handleMessage(message.data, surface) { response ->
                    reply.postMessage(response)
                }
            }
        }
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                webView.evaluateJavascript("window.remeHandleBack?.() === true") { handled ->
                    if (handled != "true") {
                        isEnabled = false
                        onBackPressedDispatcher.onBackPressed()
                        isEnabled = true
                    }
                }
            }
        })

        webView.loadUrl("$STUDIO_ORIGIN/assets/index.html")
    }

    private fun publishWindowInsets() {
        if (!studioLoaded || rendererExited) return
        val density = resources.displayMetrics.density
        val pixels = listOf(safeArea.top, safeArea.right, safeArea.bottom, safeArea.left, backGestureInset)
            .joinToString(",") { (it / density).toString() }
        webView.evaluateJavascript("window.remeSetWindowInsets?.($pixels,$keyboardVisible)", null)
    }

    private fun handleMessage(raw: String?, surface: FrameLayout, reply: (String) -> Unit) {
        val message = try {
            JSONObject(raw ?: return)
        } catch (_: org.json.JSONException) {
            return
        }
        val id = message.optInt("id", -1)
        if (id < 0) return
        fun complete(success: Boolean, payload: JSONObject = JSONObject()) {
            if (!rendererExited) {
                reply(payload.put("id", id).put("success", success).toString())
            }
        }
        when (message.optString("action")) {
            "openExternal" -> complete(openExternal(Uri.parse(message.optString("value"))))
            "recording.start" -> {
                val start: (Boolean) -> Unit = { granted ->
                    if (granted) recorder.start { complete(it) }
                    else complete(false, JSONObject().put("denied", true))
                }
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    start(true)
                } else if (pendingAudio == null) {
                    pendingAudio = start
                    microphone.launch(Manifest.permission.RECORD_AUDIO)
                } else {
                    complete(false)
                }
            }
            "recording.progress" -> complete(
                true,
                JSONObject().put("seconds", recorder.seconds).put("level", recorder.level),
            )
            "recording.stop" -> recorder.stop { audio ->
                complete(audio != null, JSONObject().put("audio", audio))
            }
            "recording.cancel" -> {
                pendingAudio?.invoke(false)
                pendingAudio = null
                recorder.cancel()
                complete(true)
            }
            "theme" -> {
                val dark = when (message.optString("value")) {
                    "dark" -> true
                    "light" -> false
                    else -> resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
                }
                val controller = WindowInsetsControllerCompat(window, surface)
                controller.isAppearanceLightStatusBars = !dark
                controller.isAppearanceLightNavigationBars = !dark
                surface.setBackgroundResource(if (dark) R.color.studio_surface_dark else R.color.studio_surface_light)
                complete(true)
            }
            else -> complete(false)
        }
    }

    private fun openExternal(uri: Uri): Boolean {
        if (uri.scheme !in setOf("http", "https") || uri.host.isNullOrEmpty() || uri.userInfo != null) return false
        return try {
            startActivity(Intent(Intent.ACTION_VIEW, uri))
            true
        } catch (_: ActivityNotFoundException) {
            false
        }
    }

    private fun disposeWebView(view: WebView) {
        (view.parent as? FrameLayout)?.removeView(view)
        view.destroy()
    }

    override fun onStop() {
        recorder.cancel()
        super.onStop()
    }

    override fun onDestroy() {
        pendingAudio = null
        recorder.close()
        if (!rendererExited) disposeWebView(webView)
        super.onDestroy()
    }

    private fun isStudioOrigin(uri: Uri): Boolean =
        uri.scheme == "https" && uri.host == WebViewAssetLoader.DEFAULT_DOMAIN && uri.port in setOf(-1, 443)

    private companion object {
        const val STUDIO_ORIGIN = "https://appassets.androidplatform.net"
    }
}
