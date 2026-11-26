import os
import UIKit
import WebKit

final class StudioViewController: UIViewController, WKNavigationDelegate, WKUIDelegate {
    private static let bridgeName = "remeStudio"
    private static let log = Logger(subsystem: "io.agentscope.reme.studio", category: "web")
    private static let pageURL = URL(string: "\(StudioAssets.origin)/index.html")!

    private var assets: StudioAssets?
    private var assetsReady = false
    private var webView: WKWebView!
    private var failure: UIStackView?
    private var darkStatusBar = false
    private let requests = StudioRequests()

    override var preferredStatusBarStyle: UIStatusBarStyle {
        darkStatusBar ? .lightContent : .darkContent
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        let configuration = WKWebViewConfiguration()
        configuration.userContentController.addScriptMessageHandler(
            StudioBridge(controller: self), contentWorld: .page, name: Self.bridgeName
        )
        webView = WKWebView(frame: .zero, configuration: configuration)
        applyTheme(dark: traitCollection.userInterfaceStyle == .dark)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        #if DEBUG
        webView.isInspectable = true
        #endif
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        webView.scrollView.bounces = false
        webView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(webView)
        let area = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            webView.topAnchor.constraint(equalTo: area.topAnchor),
            webView.bottomAnchor.constraint(lessThanOrEqualTo: area.bottomAnchor),
            webView.bottomAnchor.constraint(lessThanOrEqualTo: view.keyboardLayoutGuide.topAnchor),
            webView.leadingAnchor.constraint(equalTo: area.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: area.trailingAnchor),
        ])
        let bottom = webView.bottomAnchor.constraint(equalTo: area.bottomAnchor)
        bottom.priority = .defaultHigh
        bottom.isActive = true
        NotificationCenter.default.addObserver(
            self, selector: #selector(suspendRecording), name: UIApplication.didEnterBackgroundNotification, object: nil
        )
        loadStudio()
    }

    private func applyTheme(dark: Bool) {
        let traits = UITraitCollection(userInterfaceStyle: dark ? .dark : .light)
        let surface = UIColor(named: "StudioSurface", in: .main, compatibleWith: traits)?.resolvedColor(with: traits)
            ?? .systemBackground
        darkStatusBar = dark
        view.backgroundColor = surface
        webView.underPageBackgroundColor = surface
        setNeedsStatusBarAppearanceUpdate()
    }

    @objc private func suspendRecording() {
        webView.setMicrophoneCaptureState(.none, completionHandler: nil)
    }

    @objc private func loadStudio() {
        failure?.removeFromSuperview()
        failure = nil
        if assetsReady {
            loadPage()
            return
        }
        let root = Bundle.main.bundleURL.appendingPathComponent("Studio", isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent("index.html").path) else {
            showFailure("The bundled Studio page is missing.")
            return
        }
        do {
            let server = try StudioAssets(root: root)
            assets = server
            server.start { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    self.assetsReady = true
                    self.loadPage()
                case .waiting(let error), .failed(let error):
                    self.assetsReady = false
                    self.assets = nil
                    Self.log.error("Studio listener failed: \(error.localizedDescription, privacy: .public)")
                    self.showFailure("Could not start the bundled Studio page.")
                default: break
                }
            }
        } catch {
            Self.log.error("Studio listener failed: \(error.localizedDescription, privacy: .public)")
            showFailure("Could not start the bundled Studio page.")
        }
    }

    private func loadPage() {
        requests.cancelAll()
        webView.load(URLRequest(url: Self.pageURL))
    }

    private func showFailure(_ message: String) {
        failure?.removeFromSuperview()
        let label = UILabel()
        label.text = message
        label.numberOfLines = 0
        label.textAlignment = .center
        let retry = UIButton(type: .system)
        retry.setTitle("Retry", for: .normal)
        retry.addTarget(self, action: #selector(loadStudio), for: .touchUpInside)
        let content = UIStackView(arrangedSubviews: [label, retry])
        content.axis = .vertical
        content.spacing = 12
        content.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(content)
        NSLayoutConstraint.activate([
            content.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            content.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            content.widthAnchor.constraint(lessThanOrEqualTo: view.safeAreaLayoutGuide.widthAnchor, constant: -32),
        ])
        failure = content
        webView.isHidden = true
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        webView.isHidden = false
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        if (error as NSError).code == NSURLErrorCancelled { return }
        Self.log.error("Studio page failed: \(error.localizedDescription, privacy: .public)")
        showFailure("Could not load the bundled Studio page.")
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        requests.cancelAll()
        showFailure("The Studio page closed unexpectedly.")
    }

    func webView(
        _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        let url = navigationAction.request.url
        if navigationAction.targetFrame?.isMainFrame == true, url == Self.pageURL {
            decisionHandler(.allow)
            return
        }
        if navigationAction.navigationType == .linkActivated,
           isStudioFrame(navigationAction.sourceFrame), let url,
           Self.validWebURL(url) {
            UIApplication.shared.open(url)
        }
        decisionHandler(.cancel)
    }

    func webView(
        _ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
        initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
        decisionHandler: @escaping (WKPermissionDecision) -> Void
    ) {
        decisionHandler(type == .microphone && isStudioFrame(frame) && Self.isStudioOrigin(origin) ? .prompt : .deny)
    }

    private static func isStudioOrigin(_ origin: WKSecurityOrigin) -> Bool {
        origin.protocol == "http" && origin.host == StudioAssets.host && origin.port == Int(StudioAssets.port)
    }

    private func isStudioFrame(_ frame: WKFrameInfo) -> Bool {
        assetsReady && frame.isMainFrame && Self.isStudioOrigin(frame.securityOrigin)
    }

    private static func validWebURL(_ url: URL) -> Bool {
        ["http", "https"].contains(url.scheme?.lowercased() ?? "") &&
            url.host?.isEmpty == false && url.user == nil && url.password == nil
    }

    fileprivate func handle(_ message: WKScriptMessage, reply: @escaping (Any?, String?) -> Void) {
        func complete(_ success: Bool) {
            reply(["success": success], nil)
        }
        guard message.webView === webView, isStudioFrame(message.frameInfo),
              let payload = message.body as? [String: Any], let action = payload["action"] as? String else {
            complete(false)
            return
        }
        let value = payload["value"] as? String
        let id = payload["id"] as? Int
        switch action {
        case "request.open":
            guard let id, let urlString = payload["url"] as? String,
                  let url = URL(string: urlString), Self.validWebURL(url),
                  let method = payload["method"] as? String, ["GET", "POST"].contains(method),
                  let body = payload["body"] as? String, let accept = payload["accept"] as? String,
                  let adminToken = payload["adminToken"] as? String else {
                reply(nil, "Invalid ReMe request")
                return
            }
            requests.open(id: id, url: url, method: method, body: body, accept: accept, adminToken: adminToken,
                          reply: reply)
        case "request.read":
            guard let id else {
                reply(nil, "Invalid ReMe request")
                return
            }
            requests.read(id: id, reply: reply)
        case "request.cancel":
            if let id { requests.cancel(id: id) }
            reply(true, nil)
        case "openExternal":
            guard let value, let url = URL(string: value), Self.validWebURL(url) else {
                complete(false)
                return
            }
            UIApplication.shared.open(url) { complete($0) }
        case "theme":
            guard let value, value == "dark" || value == "light" else {
                complete(false)
                return
            }
            applyTheme(dark: value == "dark")
            complete(true)
        default:
            complete(false)
        }
    }
}

private final class StudioBridge: NSObject, WKScriptMessageHandlerWithReply {
    weak var controller: StudioViewController?

    init(controller: StudioViewController) { self.controller = controller }

    func userContentController(
        _ userContentController: WKUserContentController, didReceive message: WKScriptMessage,
        replyHandler: @escaping (Any?, String?) -> Void
    ) {
        guard let controller else {
            replyHandler(nil, "Studio is unavailable")
            return
        }
        controller.handle(message, reply: replyHandler)
    }
}
