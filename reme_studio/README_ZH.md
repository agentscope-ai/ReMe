# ReMe Studio

[English](README.md)

ReMe Studio 的网页端、桌面端、Android 和 iOS 共用一套工作台，包含 Remy、主动智能、随记和知识库。
在主动智能里管理后台任务，在随记里记录待办和想法。
Desktop 还有桌宠、托盘和系统通知。会话、任务状态和工作区文件由 ReMe 服务管理。

## 启动

安装 `reme-ai[core]`，在后端环境配置模型，然后在一个终端启动服务：

```bash
reme start config=proactive
```

浏览器打开 <http://127.0.0.1:2333/> 即可使用 Studio。桌面端、Android 和 iOS 默认连接同一个地址。
使用其他服务时，在「设置 → 连接」填入完整的 HTTP 或 HTTPS 地址并点击「连接」。重启后应用会记住该地址。

## 开发

使用 Node.js 22.13+，在仓库根目录执行：

```bash
npm ci --prefix reme_studio
npm run dev --prefix reme_studio
npm run dev:desktop --prefix reme_studio
```

Web 开发服务器将请求转发到 ReMe 的 2333 端口。运行 `npm run build:static --prefix reme_studio` 后，
可以由 ReMe 直接提供网页。运行 `npm test --prefix reme_studio` 运行共享代码和各端的测试。

- `client/` 管理 HTTP/SSE 契约、会话状态和请求。
- `workbench/` 提供共用的 React 页面和交互。
- `desktop/` 提供 Windows、macOS 和 Linux 共用的 Electron 应用、IPC、桌面集成和桌宠动画。
- `web/` 提供浏览器入口和静态打包。
- `mobile/` 提供共用的移动入口和原生桥接。原生工程的构建与联调见
  [Android](mobile/android/README.md) 和 [iOS](mobile/ios/README.md) 开发文档。

用 `npm run build:mobile --prefix reme_studio` 构建 Android 网页资源，加上 `-- --mode ios` 构建 iOS 网页资源。

## 发布

用 `python scripts/bump_version.py X.Y.Z` 更新共用版本。发布 GitHub Release 后，PyPI 包、npm 包、桌面安装包
和 Docker 镜像都从同一个标签构建并发布。

手动发布时，从版本标签运行 **Release / ReMe**，例如 `gh workflow run release-python.yml --ref vX.Y.Z`。
可选择 PyPI、npm、桌面或 Docker，四项默认启用。桌面选项将 Windows、macOS 和 Linux 安装包上传到对应的 Release，
必要时会创建 Release。
准备好向桌面端推送更新后，在 GitHub 中将该 Release 标记为 **Latest**。手动发布 Docker 会推送指定版本的镜像。
Android 和 iOS 发布暂未启用。

PyPI 和 npm 的 Trusted Publishing 绑定到 `release-python.yml`。GitHub 的 `pypi` 和 `npm` 环境需要允许
`v*` 等版本标签。

桌面体验版通过 GitHub Releases 分发，不需要 Apple 或 Windows 发布证书。macOS 使用临时签名，不进行
Apple 公证；Windows 安装包不签名。首次打开下载的应用时，系统可能要求确认允许运行。

桌面端左侧「设置」上方提供「检查更新」。Windows 和 Linux AppImage 发布包从 GitHub Releases 检查新版，
用户选择下载后，再通过「重启安装」覆盖当前应用。下载文件会按发布元数据中的 SHA-512 校验。
macOS 原生自动更新要求正式签名身份，因此 macOS 和本地预览包通过 GitHub 下载页更新。

macOS 升级时先退出桌面端，再用 DMG 中的新版替换 `/Applications/ReMe Studio.app`。Windows 也可直接运行
新版安装程序，保持原安装目录和用户范围。这些升级会保留桌面端设置和 ReMe 工作区。ReMe 服务按需单独升级。

本地 `pack` 和 `dist` 构建复用 `desktop/dist.noindex/`，避免生成的应用进入 Spotlight。日常预览使用
`npm run dev:desktop`；自定义预览输出目录也应以 `.noindex` 结尾。
