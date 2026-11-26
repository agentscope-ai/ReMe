package io.agentscope.reme.studio

import android.Manifest
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.net.DnsResolver
import android.net.InetAddresses
import android.net.IpPrefix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
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
import java.net.InetAddress
import java.net.NetworkInterface

class MainActivity : ComponentActivity() {
    private lateinit var webView: WebView
    private var rendererExited = false
    private var studioLoaded = false
    private val recorder = AndroidRecorder()
    private var pendingAudio: ((Boolean) -> Unit)? = null
    private var pendingNetwork: ((Boolean) -> Unit)? = null
    private var safeArea = Insets.NONE
    private var backGestureInset = 0
    private var keyboardVisible = false

    private val microphone = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val complete = pendingAudio
        pendingAudio = null
        complete?.invoke(granted)
    }

    private val localNetwork = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val complete = pendingNetwork
        pendingNetwork = null
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
                // The bundled HTTPS page can connect to a user-entered HTTP ReMe server.
                mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
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
                pendingNetwork = null
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
            "network.access" -> requestNetworkAccess(Uri.parse(message.optString("value")).host) { granted ->
                complete(granted)
            }
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

    private fun requestNetworkAccess(host: String?, complete: (Boolean) -> Unit) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.CINNAMON_BUN) return complete(true)
        val permission = Manifest.permission.ACCESS_LOCAL_NETWORK
        if (checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED) return complete(true)
        LocalNetwork.check(host, this) { needsPermission ->
            if (isDestroyed || rendererExited) return@check
            if (!needsPermission || checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED) {
                complete(true)
            } else if (pendingNetwork == null) {
                pendingNetwork = complete
                localNetwork.launch(permission)
            } else {
                complete(false)
            }
        }
    }

    override fun onDestroy() {
        pendingAudio = null
        pendingNetwork = null
        recorder.close()
        if (!rendererExited) disposeWebView(webView)
        super.onDestroy()
    }

    private fun isStudioOrigin(uri: Uri): Boolean =
        uri.scheme == "https" && uri.host == WebViewAssetLoader.DEFAULT_DOMAIN && uri.port in setOf(-1, 443)

    private companion object {
        const val STUDIO_ORIGIN = "https://appassets.androidplatform.net"
    }

    @RequiresApi(Build.VERSION_CODES.CINNAMON_BUN)
    private object LocalNetwork {
        private val prefixes = listOf(
            "10.0.0.0" to 8, "100.64.0.0" to 10, "169.254.0.0" to 16,
            "172.16.0.0" to 12, "192.168.0.0" to 16, "fc00::" to 7, "fe80::" to 10,
        ).map { (address, bits) -> IpPrefix(InetAddresses.parseNumericAddress(address), bits) }

        fun check(host: String?, context: Context, complete: (Boolean) -> Unit) {
            val name = host?.trimEnd('.')?.removeSurrounding("[", "]") ?: return complete(false)
            if (name.equals("localhost", ignoreCase = true)) return complete(false)
            if (name.endsWith(".local", ignoreCase = true)) return complete(true)
            if (InetAddresses.isNumericAddress(name)) {
                return complete(contains(InetAddresses.parseNumericAddress(name)))
            }
            DnsResolver(context, context.mainLooper).query(
                null, name, DnsResolver.FLAG_EMPTY, context.mainExecutor, null,
                object : DnsResolver.Callback<List<InetAddress>> {
                    override fun onAnswer(answer: List<InetAddress>, rcode: Int) = complete(answer.any(::contains))
                    override fun onError(error: DnsResolver.DnsException) = complete(false)
                },
            )
        }

        private fun contains(address: InetAddress): Boolean {
            if (address.isLoopbackAddress) return false
            if (prefixes.any { it.contains(address) }) return true
            // Some Wi-Fi networks use globally scoped addresses on their local subnet.
            return runCatching {
                NetworkInterface.getNetworkInterfaces().toList().any { network ->
                    network.interfaceAddresses.any { link ->
                        link.networkPrefixLength > 0 &&
                            IpPrefix(link.address, link.networkPrefixLength.toInt()).contains(address)
                    }
                }
            }.getOrDefault(false)
        }
    }
}
