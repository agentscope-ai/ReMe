# Codex screenshot checklist / 截图清单

Use real Codex screenshots only. The English and Chinese guides share these PNG files.
仅使用真实 Codex 界面截图；中英文说明共用以下 PNG 文件，不使用示意图或重绘的界面。

| Filename / 文件名 | Capture / 拍摄内容 |
| --- | --- |
| `plugin-installed.png` | Installed ReMe plugin detail: name, enabled state, and `reme` MCP card. / 已安装的 ReMe 插件详情：名称、启用状态及 MCP 卡片。 |
| `hooks-trusted.png` | All seven ReMe handlers across five events enabled and trusted, in desktop settings or CLI `/hooks`. / 五类事件下的七个 ReMe Hook 条目均已启用、已信任，可使用桌面设置或 CLI。 |
| `mcp-settings.png` | Native MCP form: URL, auto switches, batch size, and the single ReMe status entry. / 原生 MCP 表单：地址、自动开关、批次大小及唯一的 ReMe status 入口。 |
| `memory-recorded.png` | Conversation A: the test fact and final acknowledgement, without explicit tool calls. / 会话 A：测试事实及最终确认，不显式调用工具。 |
| `memory-recalled.png` | Separate conversation B: recall question, correct fact, and ReMe source path. / 独立会话 B：召回问题、正确事实和 ReMe 来源路径。 |
| `plugin-status.png` | ReMe status panel opened from native settings, or a real `reme_status` call in Codex: queued turns, recent activity, and the next Dream run. / 从原生设置打开的 ReMe status 面板或真实 Codex 对话中的 `reme_status` 调用，保留待提交轮数、最近活动和下次 Dream 时间。 |

## Add the screenshots / 补图步骤

1. Save the original PNG files in this directory using the names above. Keep text readable;
   include enough of the Codex interface to identify the page. Crop unrelated conversations and account details.
   将 PNG 原图按以上文件名放入本目录，保持文字清晰并保留可辨认的 Codex 界面，裁去无关对话和账号信息。
2. In both `../README.md` and `../README_ZH.md`, find the matching screenshot placeholder,
   remove its visible blockquote, and remove the `<!--` / `-->` around the image line.
   在两份说明中找到对应槽位，删除可见的占位引用块，并移除图片行外的 HTML 注释标记。
3. Keep the Markdown image path relative, such as `./figures/mcp-settings.png`.
   The documentation build copies these files and rewrites the path for the website.
   保留相对图片路径；文档构建会复制图片并转换为站点路径。
4. From `github-pages/`, run `npm test` and `npm run build`, then preview both language pages.
   在 `github-pages/` 执行上述检查，并预览中英文页面。

Use the same unique test project and facts for both conversation screenshots. Keep the
connection, trust, and memory checks separate: an acknowledgement alone does not prove a successful write.
两张对话截图使用同一组独立测试事实。连接健康、Hook 信任和记忆交付分别验证；仅有确认回复不能证明写入成功。

MCP settings require a desktop host that supports the native form; a CLI capture cannot substitute for this image.
MCP 设置图需在支持原生表单的桌面宿主中截取，不能用 CLI 输出代替。
