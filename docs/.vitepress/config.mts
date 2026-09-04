import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type DefaultTheme } from "vitepress";

const sourceRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repository = "https://github.com/agentscope-ai/ReMe";
const base = process.env.DOCS_BASE || "/";

function readSourceMap(): Record<string, string> {
  try {
    return JSON.parse(fs.readFileSync(path.join(sourceRoot, ".source-map.json"), "utf8"));
  } catch {
    return {};
  }
}

const sourceMap = readSourceMap();

function collectMarkdown(directory: string, root = directory): string[] {
  const files: string[] = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.name.startsWith(".") || entry.name === "public" || entry.name === "figure") continue;
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...collectMarkdown(absolute, root));
    else if (entry.name.endsWith(".md")) files.push(path.relative(root, absolute).replaceAll(path.sep, "/"));
  }
  return files.sort();
}

function buildLlmsFiles(outDir: string) {
  const pages = collectMarkdown(sourceRoot);
  const index = [
    "# ReMe Documentation",
    "",
    "> Local-first, file-native memory for agents.",
    "",
    ...pages.map((relativePath) => {
      const source = fs.readFileSync(path.join(sourceRoot, relativePath), "utf8");
      const title = source.match(/^#\s+(.+)$/m)?.[1]
        || source.match(/^title:\s*(.+)$/m)?.[1]
        || path.basename(relativePath, ".md");
      const route = relativePath.replace(/(?:^|\/)index\.md$/, "").replace(/\.md$/, "");
      return `- [${title}](https://reme.agentscope.io/${route})`;
    }),
    "",
  ];
  fs.writeFileSync(path.join(outDir, "llms.txt"), index.join("\n"), "utf8");

  const full = ["# ReMe Documentation", ""];
  for (const relativePath of pages) {
    const source = fs.readFileSync(path.join(sourceRoot, relativePath), "utf8");
    full.push(`<!-- source: ${sourcePathFor(relativePath)} -->`, "", source, "", "---", "");
    const pageDir = path.join(outDir, relativePath.replace(/\.md$/, ""));
    fs.mkdirSync(pageDir, { recursive: true });
    fs.writeFileSync(path.join(pageDir, "llms.txt"), source, "utf8");
  }
  fs.writeFileSync(path.join(outDir, "llms-full.txt"), full.join("\n"), "utf8");
}

function sourcePathFor(relativePath: string) {
  return sourceMap[relativePath] || `docs/${relativePath}`;
}

function nav(language: "zh" | "en"): DefaultTheme.NavItem[] {
  const zh = language === "zh";
  return [
    { text: zh ? "开始使用" : "Get Started", link: `/${language}/quick_start` },
    { text: zh ? "核心概念" : "Concepts", link: `/${language}/memory_as_file` },
    { text: zh ? "指南" : "Guides", link: `/${language}/auto_memory` },
    { text: zh ? "集成" : "Integrations", link: `/${language}/integrations` },
    { text: zh ? "API 参考" : "API Reference", link: `/${language}/reference/cli` },
    { text: zh ? "常见问题" : "FAQ", link: `/${language}/faq` },
  ];
}

function sidebar(language: "zh" | "en"): DefaultTheme.SidebarItem[] {
  const zh = language === "zh";
  return [
    {
      text: zh ? "开始使用" : "Get Started",
      collapsed: false,
      items: [
        { text: zh ? "项目介绍" : "Introduction", link: `/${language}/` },
        { text: zh ? "快速开始" : "Quick Start", link: `/${language}/quick_start` },
        { text: zh ? "基础配置" : "Configuration", link: `/${language}/configuration` },
        { text: zh ? "服务与部署" : "Services and Deployment", link: `/${language}/services` },
      ],
    },
    {
      text: zh ? "核心概念" : "Core Concepts",
      collapsed: false,
      items: [
        { text: zh ? "文件即记忆" : "Memory as File", link: `/${language}/memory_as_file` },
        { text: zh ? "记忆检索" : "Memory Search", link: `/${language}/memory_search` },
        { text: zh ? "自动关联" : "Auto Link", link: `/${language}/auto_link` },
        { text: zh ? "应用场景" : "Application Scenarios", link: `/${language}/reme_scene` },
      ],
    },
    {
      text: zh ? "记忆工作流" : "Memory Workflows",
      collapsed: false,
      items: [
        { text: "Auto Memory", link: `/${language}/auto_memory` },
        { text: "Auto Resource", link: `/${language}/auto_resource` },
        { text: "Auto Dream", link: `/${language}/auto_dream` },
        { text: "Proactive", link: `/${language}/proactive` },
      ],
    },
    {
      text: zh ? "Agent 集成" : "Agent Integrations",
      collapsed: true,
      items: [
        { text: zh ? "集成总览" : "Overview", link: `/${language}/integrations` },
        { text: "Claude Code", link: `/${language}/integrations/claude-code` },
        { text: "Hermes Agent", link: `/${language}/integrations/hermes` },
        { text: "TypeScript / OpenClaw / DSH", link: `/${language}/integrations/typescript` },
      ],
    },
    {
      text: zh ? "工作区与插件" : "Workspace and Plugins",
      collapsed: true,
      items: [
        { text: "ReMe Studio", link: `/${language}/workspace/studio` },
        { text: zh ? "插件管理" : "Plugin Management", link: `/${language}/plugin_management` },
        { text: zh ? "插件开发" : "Plugin Development", link: `/${language}/plugin_development` },
        { text: zh ? "每日论文" : "Daily Paper", link: `/${language}/plugins/daily-paper` },
        { text: "Auto Fin", link: `/${language}/plugins/auto-fin` },
        { text: "LME", link: `/${language}/plugins/lme` },
        { text: "BEAM", link: `/${language}/plugins/beam` },
      ],
    },
    {
      text: zh ? "API 参考" : "API Reference",
      collapsed: true,
      items: [
        { text: "CLI", link: `/${language}/reference/cli` },
        { text: zh ? "Job API" : "Job API", link: `/${language}/reference/jobs` },
        { text: "HTTP / MCP", link: `/${language}/services#http-api` },
      ],
    },
    {
      text: zh ? "运维" : "Operations",
      collapsed: true,
      items: [
        { text: zh ? "诊断、备份与恢复" : "Diagnostics, Backup, and Recovery", link: `/${language}/operations` },
        { text: zh ? "常见问题" : "FAQ", link: `/${language}/faq` },
      ],
    },
    {
      text: zh ? "开发者" : "Development",
      collapsed: true,
      items: [
        { text: zh ? "代码框架" : "Framework", link: `/${language}/framework` },
        { text: zh ? "开源与贡献" : "Contributing", link: `/${language}/contributing` },
      ],
    },
    {
      text: zh ? "评测" : "Benchmarks",
      collapsed: true,
      items: [
        { text: "BEAM", link: `/${language}/benchmarks/beam` },
        { text: "LongMemEval", link: `/${language}/benchmarks/longmemeval` },
        { text: "π-Bench", link: `/${language}/benchmarks/pibench` },
        { text: "Tool Memory / ExpG", link: `/${language}/benchmarks/toolmemory` },
      ],
    },
  ];
}

function configureRepositoryLinks(md: any) {
  for (const ruleName of ["link_open", "image"] as const) {
    const original = md.renderer.rules[ruleName];
    md.renderer.rules[ruleName] = (tokens: any[], index: number, options: any, env: any, self: any) => {
      const attribute = ruleName === "image" ? "src" : "href";
      const token = tokens[index];
      const attributeIndex = token.attrIndex(attribute);
      const target = attributeIndex >= 0 ? token.attrs[attributeIndex][1] : "";
      if (target && !/^(?:[a-z]+:|#|\/)/i.test(target)) {
        const cleanTarget = target.split("#")[0].split("?")[0];
        const generatedTarget = path.resolve(sourceRoot, path.dirname(env.relativePath), cleanTarget);
        if (!fs.existsSync(generatedTarget)) {
          const originalPage = sourcePathFor(env.relativePath);
          const originalTarget = path.posix.normalize(path.posix.join(path.posix.dirname(originalPage), cleanTarget));
          const suffix = target.slice(cleanTarget.length);
          const url = ruleName === "image"
            ? `https://raw.githubusercontent.com/agentscope-ai/ReMe/main/${originalTarget}${suffix}`
            : `${repository}/blob/main/${originalTarget}${suffix}`;
          token.attrs[attributeIndex][1] = url;
        }
      }
      return original ? original(tokens, index, options, env, self) : self.renderToken(tokens, index, options);
    };
  }
}

export default defineConfig({
  title: "ReMe",
  description: "Local-first, file-native memory for agents",
  base,
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: [/^http:\/\/localhost(?::\d+)?(?:\/|$)/],
  sitemap: { hostname: "https://reme.agentscope.io" },
  head: [
    ["link", { rel: "icon", type: "image/svg+xml", href: `${base}favicon.svg` }],
    ["meta", { name: "theme-color", content: "#087f6a" }],
  ],
  markdown: {
    config: configureRepositoryLinks,
  },
  transformPageData(pageData, { siteConfig }) {
    const sourcePath = path.join(siteConfig.srcDir, pageData.relativePath);
    pageData.frontmatter._sourcePath = sourcePathFor(pageData.relativePath);
    try {
      pageData.frontmatter._rawMarkdown = fs.readFileSync(sourcePath, "utf8");
    } catch {
      pageData.frontmatter._rawMarkdown = "";
    }
  },
  buildEnd(siteConfig) {
    buildLlmsFiles(siteConfig.outDir);
  },
  themeConfig: {
    logo: "/favicon.svg",
    siteTitle: "ReMe",
    search: {
      provider: "local",
      options: {
        locales: {
          zh: {
            translations: {
              button: { buttonText: "搜索文档", buttonAriaLabel: "搜索文档" },
              modal: {
                noResultsText: "没有找到相关内容",
                resetButtonTitle: "清除查询",
                footer: { selectText: "选择", navigateText: "切换", closeText: "关闭" },
              },
            },
          },
        },
      },
    },
    socialLinks: [{ icon: "github", link: repository }],
    footer: {
      message: "Released under the Apache-2.0 License.",
      copyright: "Copyright ReMe contributors",
    },
  },
  locales: {
    root: {
      label: "简体中文",
      lang: "zh-CN",
      link: "/zh/",
      themeConfig: {
        nav: nav("zh"),
        sidebar: { "/": sidebar("zh") },
        outline: { label: "页面导航", level: [2, 3] },
      },
    },
    zh: {
      label: "简体中文",
      lang: "zh-CN",
      link: "/zh/",
      themeConfig: {
        nav: nav("zh"),
        sidebar: { "/zh/": sidebar("zh") },
        outline: { label: "页面导航", level: [2, 3] },
        docFooter: { prev: "上一页", next: "下一页" },
        darkModeSwitchLabel: "外观",
        sidebarMenuLabel: "菜单",
        returnToTopLabel: "返回顶部",
        langMenuLabel: "切换语言",
      },
    },
    en: {
      label: "English",
      lang: "en-US",
      link: "/en/",
      themeConfig: {
        nav: nav("en"),
        sidebar: { "/en/": sidebar("en") },
        outline: { label: "On this page", level: [2, 3] },
        docFooter: { prev: "Previous page", next: "Next page" },
      },
    },
  },
});
