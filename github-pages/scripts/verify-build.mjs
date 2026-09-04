import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputDir = path.join(siteDir, "dist");

const requiredFiles = [
  "index.html",
  "404.html",
  "CNAME",
  "favicon.svg",
  "hashmap.json",
  "sitemap.xml",
  "llms.txt",
  "llms-full.txt",
  "zh/index.html",
  "en/index.html",
  "zh/configuration.html",
  "en/configuration.html",
  "zh/services.html",
  "en/services.html",
  "zh/reference/jobs.html",
  "en/reference/jobs.html",
  "zh/configuration/llms.txt",
  "en/configuration/llms.txt",
];

for (const relativePath of requiredFiles) await access(path.join(outputDir, relativePath));

assert.equal((await readFile(path.join(outputDir, "CNAME"), "utf8")).trim(), "reme.agentscope.io");

const ChineseConfiguration = await readFile(path.join(outputDir, "zh/configuration.html"), "utf8");
assert.match(ChineseConfiguration, /搜索文档/);
assert.match(ChineseConfiguration, /复制 Markdown/);
assert.match(ChineseConfiguration, /在 GitHub 查看源文件/);

const jobReference = await readFile(path.join(outputDir, "en/reference/jobs.html"), "utf8");
assert.match(jobReference, /Job API Reference/);
assert.match(jobReference, /auto_memory/);

console.log(`Verified ${requiredFiles.length} documentation build artifacts.`);
