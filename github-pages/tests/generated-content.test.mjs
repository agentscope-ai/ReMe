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
});

test("maps mirrored pages back to their canonical repository sources", async () => {
  const sourceMap = JSON.parse(await readFile(path.join(generatedDir, ".source-map.json"), "utf8"));
  assert.equal(sourceMap["zh/integrations/typescript.md"], "typescript/README_ZH.md");
  assert.equal(sourceMap["en/integrations/claude-code.md"], "integrations/claude_code/README.md");
  assert.equal(sourceMap["en/integrations/hermes.md"], "integrations/hermes_agent/README.md");
  assert.equal(sourceMap["en/workspace/studio.md"], "reme_studio/README.md");
  assert.equal(sourceMap["zh/plugins/lme.md"], "plugins/lme/README_ZH.md");
  assert.equal(sourceMap["en/reference/jobs.md"], "reme/config/default.yaml");
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
  await access(path.join(generatedDir, "public", "favicon.svg"));
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
    "benchmark/toolmemory/gitcha.png",
  ];
  for (const workflow of ["ci-docs.yml", "deploy-docs.yml"]) {
    const source = await readFile(path.join(repoDir, ".github/workflows", workflow), "utf8");
    for (const requiredPath of requiredPaths) assert.ok(source.includes(requiredPath), `${workflow}: ${requiredPath}`);
  }
});
