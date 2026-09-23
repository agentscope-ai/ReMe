import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";
import { legacyRoutes } from "../../docs/.vitepress/legacy-routes.mjs";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoDir = path.resolve(siteDir, "..");
const generatedDir = path.join(siteDir, ".generated", "site");

test("generates every required bilingual guide", async () => {
  const names = [
    "overview.md",
    "configuration.md",
    "services.md",
    "operations.md",
    "integrations.md",
    "plugin_development.md",
    "faq.md",
    "reference/cli.md",
    "reference/jobs.md",
  ];
  for (const language of ["zh", "en"]) {
    for (const name of names) await access(path.join(generatedDir, language, name));
  }
  await access(path.join(generatedDir, "zh/integrations/claude-code.md"));
  await access(path.join(generatedDir, "en/integrations/claude-code.md"));
  await access(path.join(generatedDir, "zh/integrations/hermes.md"));
  await access(path.join(generatedDir, "en/integrations/hermes.md"));
  await access(path.join(generatedDir, "zh/integrations/dsh.md"));
  await access(path.join(generatedDir, "en/integrations/dsh.md"));
  await access(path.join(generatedDir, "zh/integrations/openclaw.md"));
  await access(path.join(generatedDir, "en/integrations/openclaw.md"));
  await access(path.join(generatedDir, "public/figures/dsh/reme-status-overview.png"));
  await access(path.join(generatedDir, "public/figures/studio/studio-overview.png"));
  for (const name of ["qwenpaw.png", "deepseek-harness.svg", "openclaw.svg", "claude-code.png", "hermes.svg", "zvec.ico", "faiss.png"]) {
    await access(path.join(generatedDir, "public/ecosystem", name));
  }
});

test("maps mirrored pages back to their canonical repository sources", async () => {
  const sourceMap = JSON.parse(await readFile(path.join(generatedDir, ".source-map.json"), "utf8"));
  assert.equal(sourceMap["zh/overview.md"], "README_ZH.md");
  assert.equal(sourceMap["en/overview.md"], "README.md");
  assert.equal(sourceMap["en/integrations/dsh.md"], "integrations/dsh/README.md");
  assert.equal(sourceMap["zh/integrations/openclaw.md"], "integrations/openclaw/README_ZH.md");
  assert.equal(sourceMap["en/integrations/claude-code.md"], "integrations/claude_code/README.md");
  assert.equal(sourceMap["en/integrations/hermes.md"], "integrations/hermes_agent/README.md");
  assert.equal(sourceMap["en/workspace/studio.md"], "reme_studio/README.md");
  assert.equal(sourceMap["zh/plugins/lme.md"], "plugins/lme/README_ZH.md");
  assert.equal(sourceMap["en/reference/jobs.md"], "reme/config/default.yaml");
});

test("publishes the root READMEs as localized project overviews", async () => {
  const english = await readFile(path.join(generatedDir, "en/overview.md"), "utf8");
  const chinese = await readFile(path.join(generatedDir, "zh/overview.md"), "utf8");
  assert.match(english, /^---\ntitle: ReMe Overview/m);
  assert.match(chinese, /^---\ntitle: ReMe 项目介绍/m);
  assert.match(english, /href="\/zh\/overview"/);
  assert.match(chinese, /href="\/en\/overview"/);
  assert.match(english, /https:\/\/reme\.agentscope\.io\/en\/blog_20260920/);
  assert.doesNotMatch(english, /https:\/\/reme\.agentscope\.io\/zh\/blog_20260920/);
  assert.match(chinese, /https:\/\/reme\.agentscope\.io\/zh\/blog_20260920/);
  assert.match(english, /\[ReMe Memory Tags\]/);
  assert.match(chinese, /\[给记忆加上“标签”\]/);
  assert.match(english, /src="\.\.\/figure\/design-philosophy\.svg"/);
  assert.match(english, /\(https:\/\/reme\.agentscope\.io\/en\/memory_search\)/);
  assert.match(chinese, /\(https:\/\/reme\.agentscope\.io\/zh\/memory_search\)/);
});

test("publishes portable and accurate DSH instructions", async () => {
  const english = await readFile(path.join(generatedDir, "en/integrations/dsh.md"), "utf8");
  const chinese = await readFile(path.join(generatedDir, "zh/integrations/dsh.md"), "utf8");
  assert.doesNotMatch(english, /\/Users\//);
  assert.doesNotMatch(chinese, /\/Users\//);
  assert.match(english, /runtime counters refresh every 5 seconds/);
  assert.match(chinese, /每 5 秒仅刷新 DSH 插件的运行时计数/);
});

test("publishes Studio screenshots with site-safe links", async () => {
  const english = await readFile(path.join(generatedDir, "en/workspace/studio.md"), "utf8");
  const chinese = await readFile(path.join(generatedDir, "zh/workspace/studio.md"), "utf8");
  assert.match(english, /\(\/figures\/studio\/studio-overview\.png\)/);
  assert.match(chinese, /\(\/figures\/studio\/settings-status\.png\)/);
  assert.match(english, /\(\/zh\/workspace\/studio\)/);
  assert.match(chinese, /\(\/en\/workspace\/studio\)/);
});

test("keeps Studio source READMEs portable for package registries", async () => {
  for (const name of ["README.md", "README_ZH.md"]) {
    const source = await readFile(path.join(repoDir, "reme_studio", name), "utf8");
    assert.doesNotMatch(source, /\]\(\.\/figures\//);
    assert.match(
      source,
      /\]\(https:\/\/raw\.githubusercontent\.com\/agentscope-ai\/ReMe\/main\/reme_studio\/figures\//,
    );
  }
});

test("generates the callable Job reference from default.yaml", async () => {
  const config = parseYaml(await readFile(path.join(repoDir, "reme/config/default.yaml"), "utf8"));
  const callableJobs = Object.entries(config.jobs)
    .filter(([, job]) => !["background", "cron"].includes(job.backend))
    .map(([name]) => name);

  for (const language of ["zh", "en"]) {
    const reference = await readFile(path.join(generatedDir, language, "reference", "jobs.md"), "utf8");
    for (const job of callableJobs) assert.ok(reference.includes(`### ${"`"}${job}${"`"}`), job);
  }
});

test("keeps generated content disposable and excludes internal plans", async () => {
  await assert.rejects(access(path.join(generatedDir, "plans")));
  await access(path.join(generatedDir, ".vitepress", "config.mts"));
  await access(path.join(generatedDir, "public", "reme-icon.svg"));
  await access(path.join(generatedDir, "public", "reme-logo.svg"));
  assert.equal(
    (await readFile(path.join(generatedDir, "public", "CNAME"), "utf8")).trim(),
    "reme.agentscope.io",
  );
});

test("maps every legacy query-string document ID to a generated page", async () => {
  assert.equal(Object.keys(legacyRoutes).length, 45);
  assert.equal(legacyRoutes["studio-en"], "/en/workspace/studio");
  assert.equal(legacyRoutes["en-quick_start"], "/en/quick_start");
  assert.equal(legacyRoutes["agents-guide"], "https://github.com/agentscope-ai/ReMe/blob/main/AGENTS.md");

  for (const [id, route] of Object.entries(legacyRoutes)) {
    if (route.startsWith("https://")) continue;
    assert.match(route, /^\/(?:zh|en)\//, id);
    const relative = route.endsWith("/") ? `${route.slice(1)}index.md` : `${route.slice(1)}.md`;
    await access(path.join(generatedDir, relative));
  }
});

test("tracks every generated input in documentation CI and deployment", async () => {
  const requiredPaths = [
    "reme/config/default.yaml",
    "integrations/claude_code/README.md",
    "integrations/hermes_agent/README.md",
    "integrations/dsh/README*.md",
    "integrations/dsh/figures/**",
    "integrations/openclaw/README*.md",
    "reme_studio/README*.md",
    "reme_studio/figures/**",
    "benchmark/toolmemory/gitcha.png",
  ];
  for (const workflow of ["ci-docs.yml", "deploy-docs.yml"]) {
    const source = await readFile(path.join(repoDir, ".github/workflows", workflow), "utf8");
    for (const requiredPath of requiredPaths) assert.ok(source.includes(requiredPath), `${workflow}: ${requiredPath}`);
  }
});
