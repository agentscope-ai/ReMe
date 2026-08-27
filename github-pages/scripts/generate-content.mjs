import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoDir = path.resolve(siteDir, "..");
const outputDir = path.join(siteDir, ".generated", "content");

const navigationGroupOrder = [
  "overview",
  "start",
  "integration",
  "fundamentals",
  "automation",
  "concepts",
  "workspace",
  "plugins",
  "benchmarks",
  "development",
];

const topicOrder = [
  "quick_start",
  "plugin_management",
  "memory_as_file",
  "memory_search",
  "auto_memory",
  "auto_resource",
  "auto_link",
  "auto_dream",
  "proactive",
  "reme_scene",
  "framework",
  "reme-blog",
  "contributing",
];

const groups = {
  quick_start: "start",
  plugin_management: "start",
  memory_as_file: "fundamentals",
  memory_search: "fundamentals",
  auto_memory: "automation",
  auto_resource: "automation",
  auto_link: "automation",
  auto_dream: "automation",
  proactive: "automation",
  reme_scene: "concepts",
  framework: "concepts",
  "reme-blog": "concepts",
  contributing: "development",
};

const localizedTitles = {
  quick_start: { zh: "快速开始", en: "Quick Start" },
  plugin_management: { zh: "插件管理", en: "Plugin Management" },
  memory_as_file: { zh: "文件即记忆", en: "Memory as File" },
  memory_search: { zh: "记忆检索", en: "Memory Search" },
  auto_memory: { zh: "自动记忆", en: "Auto Memory" },
  auto_resource: { zh: "自动资料整理", en: "Auto Resource" },
  auto_link: { zh: "自动关联", en: "Auto Link" },
  auto_dream: { zh: "自动沉淀", en: "Auto Dream" },
  proactive: { zh: "主动发现", en: "Proactive" },
  reme_scene: { zh: "ReMe 应用场景", en: "ReMe Application Scenarios" },
  framework: { zh: "ReMe 代码框架", en: "ReMe Framework" },
  "reme-blog": { zh: "ReMe 博客", en: "ReMe Blog" },
  contributing: { zh: "开源与贡献", en: "Open Source and Contributing" },
};

const productDocuments = [
  {
    slug: "typescript",
    source: "typescript",
    titles: { zh: "TypeScript Agent 集成", en: "TypeScript Agent Integrations" },
    descriptions: {
      zh: "配置统一 HTTP client，以及 DeepSeek Harness 和 OpenClaw 原生适配器。",
      en: "Configure the shared HTTP client and native DeepSeek Harness and OpenClaw adapters.",
    },
    group: "integration",
  },
  {
    slug: "studio",
    source: "reme_studio",
    titles: { zh: "ReMe 工作台", en: "ReMe Studio" },
    descriptions: {
      zh: "浏览、编辑和搜索本地记忆，并探索记忆图谱。",
      en: "Browse, edit, search, and explore local memory from the web workspace.",
    },
    group: "workspace",
  },
  {
    slug: "daily-paper",
    source: "plugins/daily_paper",
    titles: { zh: "每日论文插件", en: "Daily Paper Plugin" },
    descriptions: {
      zh: "发现论文、解析 PDF，并生成阅读笔记与每日简报。",
      en: "Discover papers, analyze PDFs, and produce reading notes and a daily brief.",
    },
    group: "plugins",
  },
  {
    slug: "auto-fin",
    source: "plugins/auto-fin",
    titles: { zh: "Auto Fin 插件", en: "Auto Fin Plugin" },
    descriptions: {
      zh: "结合最新财联社新闻与本地历史记忆生成研究报告。",
      en: "Research recent CLS news with historical context from local memory.",
    },
    group: "plugins",
  },
  {
    slug: "beam",
    source: "benchmark/beam",
    titles: { zh: "BEAM", en: "BEAM" },
    descriptions: {
      zh: "评测大规模记忆检索能力。",
      en: "Evaluate memory retrieval at scale.",
    },
    group: "benchmarks",
  },
  {
    slug: "longmemeval",
    source: "benchmark/longmemeval",
    titles: { zh: "LongMemEval", en: "LongMemEval" },
    descriptions: {
      zh: "评测跨会话长期记忆问答能力。",
      en: "Evaluate long-term, cross-session memory question answering.",
    },
    group: "benchmarks",
  },
  {
    slug: "pibench",
    source: "benchmark/pibench",
    titles: { zh: "π-Bench", en: "π-Bench" },
    descriptions: {
      zh: "评测带持久记忆的个人智能体。",
      en: "Evaluate personal agents with persistent memory.",
    },
    group: "benchmarks",
  },
  {
    slug: "toolmemory",
    source: "benchmark/toolmemory",
    titles: { zh: "Tool Memory / ExpG", en: "Tool Memory / ExpG" },
    descriptions: {
      zh: "通过经验驱动的自适应指导增强 Agent 工具使用。",
      en: "Improve agent tool use through experience-driven adaptive guidance.",
    },
    group: "benchmarks",
  },
];

const sharedDocuments = [
  {
    id: "agents-guide",
    path: "AGENTS.md",
    sourcePath: "AGENTS.md",
    titles: {
      zh: "Agent 开发指南",
      en: "Agent Development Guide",
    },
    description: "Repository contracts, lifecycle rules, safety boundaries, and validation.",
    group: "development",
    language: "shared",
  },
];

async function markdownTitle(filePath) {
  const source = await readFile(filePath, "utf8");
  return source.match(/^#\s+(.+)$/m)?.[1]?.replace(/[`*_]/g, "") || path.basename(filePath, ".md");
}

async function buildManifest() {
  const documents = [
    {
      id: "readme-zh",
      path: "README_ZH.md",
      sourcePath: "README_ZH.md",
      title: "ReMe 项目介绍",
      description: "核心理念、快速开始、使用场景与社区入口。",
      group: "overview",
      language: "zh",
    },
    {
      id: "readme-en",
      path: "README.md",
      sourcePath: "README.md",
      title: "Introducing ReMe",
      description: "Core ideas, quick start, use cases, and community resources.",
      group: "overview",
      language: "en",
    },
  ];

  for (const language of ["zh", "en"]) {
    for (const topic of topicOrder) {
      const sourcePath = `docs/${language}/${topic}.md`;
      documents.push({
        id: `${language}-${topic}`,
        path: sourcePath,
        sourcePath,
        title: localizedTitles[topic]?.[language] || (await markdownTitle(path.join(repoDir, sourcePath))),
        description: "",
        group: groups[topic],
        language,
      });
    }

    for (const product of productDocuments) {
      const filename = language === "zh" ? "README_ZH.md" : "README.md";
      documents.push({
        id: `${product.slug}-${language}`,
        path: `${product.source}/${filename}`,
        sourcePath: `${product.source}/${filename}`,
        title: product.titles[language],
        description: product.descriptions[language],
        group: product.group,
        language,
      });
    }
  }

  return [...documents, ...sharedDocuments].sort(
    (left, right) => navigationGroupOrder.indexOf(left.group) - navigationGroupOrder.indexOf(right.group),
  );
}

await rm(path.join(siteDir, ".generated"), { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
await cp(path.join(siteDir, "public", "favicon.svg"), path.join(siteDir, ".generated", "favicon.svg"));
await cp(path.join(siteDir, "public", "CNAME"), path.join(siteDir, ".generated", "CNAME"));

for (const file of ["README.md", "README_ZH.md", "AGENTS.md"]) {
  await cp(path.join(repoDir, file), path.join(outputDir, file));
}
await cp(path.join(repoDir, "docs"), path.join(outputDir, "docs"), {
  recursive: true,
  filter: (source) => path.basename(source) !== ".DS_Store",
});
for (const product of productDocuments) {
  await mkdir(path.join(outputDir, product.source), { recursive: true });
  for (const filename of ["README.md", "README_ZH.md"]) {
    await cp(path.join(repoDir, product.source, filename), path.join(outputDir, product.source, filename));
  }
}
await mkdir(path.join(outputDir, "reme_studio", "public"), { recursive: true });
await cp(
  path.join(repoDir, "reme_studio", "public", "og.jpg"),
  path.join(outputDir, "reme_studio", "public", "og.jpg"),
);
await writeFile(
  path.join(outputDir, "manifest.json"),
  `${JSON.stringify({ documents: await buildManifest() }, null, 2)}\n`,
);
