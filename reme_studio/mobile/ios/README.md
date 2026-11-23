# ReMe Studio for iOS

ReMe Studio for iOS 启动后直接连接 `http://127.0.0.1:2333`。部署目标为 iOS 18.0 及以上。 [Apple 兼容机型](https://support.apple.com/en-us/104985)

## Android 与 iOS 共用的基建

| 位置 | 复用范围 |
| --- | --- |
| `client/` | Web 和移动端共用 `BrowserHost`、服务连接、HTTP/SSE、会话恢复、已读状态、文件操作和语音请求。 [共享客户端](../../client/src/reme-client.ts) · [BrowserHost](../../client/src/browser-host.ts) |
| `workbench/` | 共用页面、右推抽屉、二级菜单、Remy 首页、`/clear` 确认和录音交互。 [Workbench](../../workbench/src/workspace.tsx) · [移动样式](../../workbench/src/mobile.css) |
| `mobile/` | 两端共用 HTML/React 入口、Vite 配置和原生桥接口。 [移动入口](../src/main.tsx) · [原生桥](../src/native-bridge.ts) · [Vite 配置](../vite.config.mts) |
| `android/` 与 `ios/` | 各自打包 WebView 资源，处理系统权限、安全区、键盘和生命周期。 [Android 开发文档](../android/README.md) |

## iOS 宿主

- SwiftUI 承载 `WKWebView`。Xcode 构建 `safari18` 目标的共享页面，将页面打包到 App Bundle，并同步 ReMe 版本。 [静态资源加载](ReMeStudio/StudioAssets.swift) · [ReMe 版本](../../../reme/_version.py)
- 原生桥处理外链和状态栏主题。宿主只为内置页面的主 frame 处理麦克风授权，应用进入后台时释放录音。 [iOS 宿主](ReMeStudio/StudioViewController.swift)
- WebView 使用共享客户端直接发送 HTTP 请求和订阅 SSE。会话、工作区文件、模型凭据和后台任务由 ReMe 服务管理。 [服务配置](../../../reme/config/proactive.yaml)

## 构建与联调

准备 macOS、完整 Xcode、iOS Simulator runtime、Node.js 22.13+ 和仓库的 Python 开发环境。从仓库根目录安装 Studio 依赖。使用 Node 版本管理器时，可以从带有对应 PATH 的终端运行 `xcodebuild`，或者为 Xcode Build Settings 指定 `NODE_BINARY`。 [Xcode 模拟器组件](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components) · [仓库环境说明](../../../AGENTS.md)

```bash
npm ci --prefix reme_studio
xcodebuild -version
xcrun simctl list devices available
```

在 Mac 上启动 ReMe 服务，或使用已经运行在 `2333` 上的服务。需要模型回复或语音时，在后端配置对应供应商。

```bash
reme start config=proactive
```

在 Xcode 打开 `reme_studio/mobile/ios/ReMeStudio.xcodeproj`，选择 `ReMeStudio` scheme 和 iPhone Simulator 后运行。在「设置 → 通用」确认服务显示「已连接」。也可从仓库根目录运行 UI 测试，验证启动和重启后自动连接。

```bash
xcodebuild -project reme_studio/mobile/ios/ReMeStudio.xcodeproj \
  -scheme ReMeStudio -destination 'platform=iOS Simulator,id=<Simulator UDID>' \
  -parallel-testing-enabled NO test
```

## 验收

| 环境 | 验收内容 |
| --- | --- |
| iPhone Simulator | 无服务时仍能打开内置页面。连接后验证分段显示流式回复、停止生成、`/clear` 确认、任务侧栏、随记写入和知识库文件。 |
| iPhone Simulator | 验证抽屉和二级菜单、深浅色、安全区和软键盘。重启或暂时断开网络后确认自动重连和会话同步。 |
| Android 回归 | 修改共享移动入口或样式后，检查 APK 打包和关键交互。 [Android 构建与验收](../android/README.md#构建与验收) |
