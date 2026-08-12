import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoDir = path.resolve(siteDir, "..");
const outputDir = path.join(siteDir, ".generated", "content");

const topicOrder = [
  "quick_start",
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

const sharedDocuments = [
  {
    id: "reme-memory-skill",
    path: "skills/reme_memory/SKILL.md",
    sourcePath: "skills/reme_memory/SKILL.md",
    titles: {
      zh: "ReMe 记忆技能",
      en: "ReMe Memory Skill",
    },
    description: "Bootstrap, retrieve, write, and consolidate memory from an agent.",
    group: "integration",
    language: "shared",
  },
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
  }

  return [...documents, ...sharedDocuments];
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
await mkdir(path.join(outputDir, "skills", "reme_memory"), { recursive: true });
await cp(
  path.join(repoDir, "skills", "reme_memory", "SKILL.md"),
  path.join(outputDir, "skills", "reme_memory", "SKILL.md"),
);

await writeFile(
  path.join(outputDir, "manifest.json"),
  `${JSON.stringify({ documents: await buildManifest() }, null, 2)}\n`,
);
