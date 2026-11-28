# ReMe Studio for Android

Android 默认连接 `http://127.0.0.1:2333`。在「设置 → 连接」可填写手机能访问的 ReMe 地址，例如
`http://203.0.113.10:2333`（将 IP 替换为你的服务地址）。点击「连接」后应用会保存地址，重启时自动连接。
在 Android 17 上连接局域网服务时，允许系统弹窗请求的局域网访问权限。

## 开发方案

- Android 宿主使用 Kotlin、Android Gradle Plugin 9.5.2、AndroidX WebKit 1.18 和 Android SDK Platform 37.2。应用最低支持 Android 10，目标 API 为 37。APK 内置 Vite 构建的 Studio 页面，通过 WebViewAssetLoader 的 HTTPS origin 加载。
- 底部固定「主动智能、随记、Remy、记忆、设置」五个入口，Remy 居中。页面标题可打开次级导航面板；系统返回先关闭弹窗、录音或文档，再返回上一页。版本与更新位于设置内。
- 轻点输入框输入文字，长按空白输入框录音，输入框显示「松手发送、上滑取消」。松手后识别并发送，上滑后松手取消；已有草稿保留原生文本选择。点按话筒也可开始和结束录音，是否自动发送由语音设置决定。
- 页面背景延伸到状态栏和导航栏。Android 将安全区和键盘状态交给共享界面，并按需申请麦克风权限。WebView 通过共享客户端直接消费流式回复和会话订阅。

## 构建与验收

从仓库根目录安装依赖。使用 Node.js 22.13+、JDK 17+、Android SDK Platform 37.2 和 Build Tools 36.0.0。设置 `JAVA_HOME` 和 `ANDROID_HOME`，并将 Android SDK 的 `platform-tools` 加入 `PATH`。

```bash
npm ci --prefix reme_studio
cd reme_studio/mobile/android
./gradlew :app:assembleDebug :app:lintDebug
```

Gradle 会构建 Android 和 iOS 共用的 `reme_studio/mobile` 入口，并将 Workbench 打包进 APK。调试 APK 位于 `app/build/outputs/apk/debug/app-debug.apk`，使用开发设备的调试密钥签名，可直接安装。 [iOS 开发文档](../ios/README.md)

在 Mac 上启动 ReMe 服务，或使用已经运行在 `2333` 上的服务。需要模型回复时，按 `proactive` 配置设置后端供应商凭据。

```bash
reme start config=proactive
```

启动 Android 模拟器，在另一个终端从 `reme_studio/mobile/android` 安装并运行。`adb reverse` 把 Android 的环回端口 `2333` 转发到 Mac 的同一个端口。连接多台设备时，用 `adb -s <设备序列号>` 选择目标。

```bash
adb reverse tcp:2333 tcp:2333
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n io.agentscope.reme.studio/.MainActivity
```

启动后在「设置 → 连接」确认服务显示「已连接」。验收也覆盖重启后自动连接、流式对话、`/clear`、任务总览与全部任务、随记、记忆文件、长按录音与取消、Android 返回和软键盘。键盘弹出时隐藏底部导航，输入框跟随键盘上沿。

平台参考见 [Android Gradle Plugin](https://developer.android.com/build/releases/about-agp) 和 [WebViewAssetLoader](https://developer.android.com/develop/ui/views/layout/webapps/load-local-content)。
