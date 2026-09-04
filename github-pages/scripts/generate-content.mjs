import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoDir = path.resolve(siteDir, "..");
const outputDir = path.join(siteDir, ".generated", "site");

const externalDocuments = [
  ["zh/overview.md", "README_ZH.md"],
  ["en/overview.md", "README.md"],
  ["en/integrations/claude-code.md", "integrations/claude_code/README.md"],
  ["en/integrations/hermes.md", "integrations/hermes_agent/README.md"],
  ["zh/integrations/typescript.md", "typescript/README_ZH.md"],
  ["en/integrations/typescript.md", "typescript/README.md"],
  ["zh/integrations/dsh.md", "typescript/docs/dsh.zh-CN.md"],
  ["en/integrations/dsh.md", "typescript/docs/dsh.md"],
  ["zh/integrations/openclaw.md", "typescript/docs/openclaw.zh-CN.md"],
  ["en/integrations/openclaw.md", "typescript/docs/openclaw.md"],
  ["zh/workspace/studio.md", "reme_studio/README_ZH.md"],
  ["en/workspace/studio.md", "reme_studio/README.md"],
  ["zh/plugins/daily-paper.md", "plugins/daily_paper/README_ZH.md"],
  ["en/plugins/daily-paper.md", "plugins/daily_paper/README.md"],
  ["zh/plugins/auto-fin.md", "plugins/auto-fin/README_ZH.md"],
  ["en/plugins/auto-fin.md", "plugins/auto-fin/README.md"],
  ["zh/plugins/lme.md", "plugins/lme/README_ZH.md"],
  ["en/plugins/lme.md", "plugins/lme/README.md"],
  ["zh/plugins/beam.md", "plugins/beam/README_ZH.md"],
  ["en/plugins/beam.md", "plugins/beam/README.md"],
  ["zh/benchmarks/beam.md", "benchmark/beam/README_ZH.md"],
  ["en/benchmarks/beam.md", "benchmark/beam/README.md"],
  ["zh/benchmarks/longmemeval.md", "benchmark/longmemeval/README_ZH.md"],
  ["en/benchmarks/longmemeval.md", "benchmark/longmemeval/README.md"],
  ["zh/benchmarks/pibench.md", "benchmark/pibench/README_ZH.md"],
  ["en/benchmarks/pibench.md", "benchmark/pibench/README.md"],
  ["zh/benchmarks/toolmemory.md", "benchmark/toolmemory/README_ZH.md"],
  ["en/benchmarks/toolmemory.md", "benchmark/toolmemory/README.md"],
];

const externalDocumentRewrites = {
  "README.md": [
    ['href="./LICENSE"', 'href="https://github.com/agentscope-ai/ReMe/blob/main/LICENSE"'],
    ['href="./README.md"', 'href="/en/overview"'],
    ['href="./README_ZH.md"', 'href="/zh/overview"'],
    ['src="docs/figure/', 'src="../figure/'],
    ["(docs/en/", "(./"],
  ],
  "README_ZH.md": [
    ['href="./LICENSE"', 'href="https://github.com/agentscope-ai/ReMe/blob/main/LICENSE"'],
    ['href="./README.md"', 'href="/en/overview"'],
    ['href="./README_ZH.md"', 'href="/zh/overview"'],
    ['src="docs/figure/', 'src="../figure/'],
    ["(docs/zh/", "(./"],
  ],
  "typescript/README.md": [
    ["(./README_ZH.md)", "(/zh/integrations/typescript)"],
    ["(./docs/dsh.md)", "(/en/integrations/dsh)"],
    ["(./docs/dsh.zh-CN.md)", "(/zh/integrations/dsh)"],
    ["(./docs/openclaw.md)", "(/en/integrations/openclaw)"],
    ["(./docs/openclaw.zh-CN.md)", "(/zh/integrations/openclaw)"],
    ["(./figures/dsh/", "(/figures/dsh/"],
  ],
  "typescript/README_ZH.md": [
    ["(./README.md)", "(/en/integrations/typescript)"],
    ["(./docs/dsh.md)", "(/en/integrations/dsh)"],
    ["(./docs/dsh.zh-CN.md)", "(/zh/integrations/dsh)"],
    ["(./docs/openclaw.md)", "(/en/integrations/openclaw)"],
    ["(./docs/openclaw.zh-CN.md)", "(/zh/integrations/openclaw)"],
    ["(./figures/dsh/", "(/figures/dsh/"],
  ],
  "typescript/docs/dsh.md": [
    ["(./dsh.zh-CN.md)", "(/zh/integrations/dsh)"],
    ["(../figures/dsh/", "(/figures/dsh/"],
  ],
  "typescript/docs/dsh.zh-CN.md": [
    ["(./dsh.md)", "(/en/integrations/dsh)"],
    ["(../figures/dsh/", "(/figures/dsh/"],
  ],
  "typescript/docs/openclaw.md": [
    ["(./openclaw.zh-CN.md)", "(/zh/integrations/openclaw)"],
  ],
  "typescript/docs/openclaw.zh-CN.md": [
    ["(./openclaw.md)", "(/en/integrations/openclaw)"],
  ],
};

const externalDocumentPreambles = {
  "README.md": "---\ntitle: ReMe Overview\ndescription: A local-first, self-evolving personal knowledge base for AI agents.\n---\n\n# ReMe Overview\n\n",
  "README_ZH.md": "---\ntitle: ReMe 项目介绍\ndescription: 面向 AI Agent 的 local-first 自进化个人知识库。\n---\n\n# ReMe 项目介绍\n\n",
};

const groupNames = {
  zh: {
    system: "系统与诊断",
    memory: "记忆演化",
    retrieval: "检索与图谱",
    daily: "Daily Note",
    files: "文件操作",
  },
  en: {
    system: "System and diagnostics",
    memory: "Memory evolution",
    retrieval: "Retrieval and graph",
    daily: "Daily notes",
    files: "File operations",
  },
};

const jobGroups = {
  version: "system",
  app_config: "system",
  chat: "system",
  health_check: "system",
  status: "system",
  help: "system",
  auto_dream: "memory",
  auto_memory: "memory",
  auto_memory_cc: "memory",
  auto_resource: "memory",
  proactive: "memory",
  traverse: "retrieval",
  graph_snapshot: "retrieval",
  reindex: "retrieval",
  search: "retrieval",
  node_search: "retrieval",
  daily_list: "daily",
  daily_reindex: "daily",
  daily_write: "daily",
  frontmatter_delete: "files",
  frontmatter_read: "files",
  frontmatter_update: "files",
  stat: "files",
  list: "files",
  move: "files",
  delete: "files",
  read: "files",
  load: "files",
  read_image: "files",
  write: "files",
  save: "files",
  edit: "files",
};

function typeLabel(schema = {}) {
  if (schema.oneOf) return schema.oneOf.map(typeLabel).join(" or ");
  if (schema.type === "array") return `${typeLabel(schema.items || {})}[]`;
  return schema.type || "any";
}

function markdownCell(value) {
  if (value === undefined) return "—";
  const rendered = typeof value === "string" ? value : JSON.stringify(value);
  return rendered
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("|", "\\|")
    .replaceAll("\n", " ");
}

function buildJobReference(config, language) {
  const isZh = language === "zh";
  const jobs = Object.entries(config.jobs || {}).filter(([, job]) => !["background", "cron"].includes(job.backend));
  const sections = new Map();

  for (const [name, job] of jobs) {
    const group = jobGroups[name] || "system";
    if (!sections.has(group)) sections.set(group, []);
    sections.get(group).push([name, job]);
  }

  const lines = [
    "---",
    `title: ${isZh ? "Job API 参考" : "Job API Reference"}`,
    `description: ${isZh ? "从默认配置自动生成的可调用 Job、参数和服务边界。" : "Callable jobs, parameters, and service boundaries generated from the default configuration."}`,
    "---",
    "",
    `# ${isZh ? "Job API 参考" : "Job API Reference"}`,
    "",
    isZh
      ? "本页从 `reme/config/default.yaml` 自动生成。它描述默认应用中的可调用 Job；插件和自定义配置可以增加、删除或覆盖 Job。运行 `reme help` 可查看当前服务的实际能力。"
      : "This page is generated from `reme/config/default.yaml`. It describes callable jobs in the default application; plugins and custom configurations may add, remove, or override jobs. Run `reme help` to inspect the active service.",
    "",
    isZh
      ? "> 后台 Job 和 Cron Job 不通过服务暴露，因此不列入调用参考。"
      : "> Background and cron jobs are not service-exposed and are omitted from the callable reference.",
    "",
  ];

  for (const [group, entries] of sections) {
    lines.push(`## ${groupNames[language][group]}`, "");
    for (const [name, job] of entries) {
      const properties = job.parameters?.properties || {};
      const required = new Set(job.parameters?.required || []);
      lines.push(`### \`${name}\``, "", markdownCell(job.description || ""), "");
      lines.push("```bash", `reme ${name}${Object.keys(properties).length ? " ..." : ""}`, "```", "");
      if (!Object.keys(properties).length) {
        lines.push(isZh ? "无参数。" : "No parameters.", "");
        continue;
      }
      lines.push(
        isZh
          ? "| 参数 | 类型 | 必填 | 默认值 | 说明 |"
          : "| Parameter | Type | Required | Default | Description |",
        "|---|---|---:|---|---|",
      );
      for (const [parameter, schema] of Object.entries(properties)) {
        lines.push(
          `| \`${parameter}\` | \`${markdownCell(typeLabel(schema))}\` | ${required.has(parameter) ? (isZh ? "是" : "yes") : (isZh ? "否" : "no")} | ${markdownCell(schema.default)} | ${markdownCell(schema.description || "—")} |`,
        );
      }
      lines.push("");
    }
  }
  return `${lines.join("\n")}\n`;
}

await rm(path.join(siteDir, ".generated"), { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
await cp(path.join(repoDir, "docs"), outputDir, {
  recursive: true,
  filter: (source) => ![".DS_Store", "plans"].includes(path.basename(source)),
});
await cp(path.join(siteDir, "public", "CNAME"), path.join(outputDir, "public", "CNAME"));
await cp(path.join(repoDir, "docs/figure/reme-icon.svg"), path.join(outputDir, "public", "reme-icon.svg"));
await cp(path.join(repoDir, "docs/figure/reme-logo-fashion.svg"), path.join(outputDir, "public", "reme-logo.svg"));

const sourceMap = {};
for (const [destination, source] of externalDocuments) {
  const destinationPath = path.join(outputDir, destination);
  await mkdir(path.dirname(destinationPath), { recursive: true });
  let content = await readFile(path.join(repoDir, source), "utf8");
  for (const [from, to] of externalDocumentRewrites[source] || []) content = content.replaceAll(from, to);
  content = `${externalDocumentPreambles[source] || ""}${content}`;
  await writeFile(destinationPath, content);
  sourceMap[destination] = source;
}

await cp(path.join(repoDir, "typescript/figures/dsh"), path.join(outputDir, "public/figures/dsh"), {
  recursive: true,
});

for (const language of ["zh", "en"]) {
  await cp(
    path.join(repoDir, "benchmark/toolmemory/gitcha.png"),
    path.join(outputDir, language, "benchmarks", "gitcha.png"),
  );
}

const defaultConfig = parseYaml(await readFile(path.join(repoDir, "reme/config/default.yaml"), "utf8"));
for (const language of ["zh", "en"]) {
  const destination = path.join(outputDir, language, "reference", "jobs.md");
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, buildJobReference(defaultConfig, language));
  sourceMap[`${language}/reference/jobs.md`] = "reme/config/default.yaml";
}

await writeFile(path.join(outputDir, ".source-map.json"), `${JSON.stringify(sourceMap, null, 2)}\n`);
