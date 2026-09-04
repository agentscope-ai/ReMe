import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputDir = path.join(siteDir, "dist");

async function collectFiles(directory, prefix = "") {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relativePath = path.posix.join(prefix, entry.name);
    if (entry.isDirectory()) files.push(...await collectFiles(path.join(directory, entry.name), relativePath));
    else files.push(relativePath);
  }
  return files;
}

function pageUrl(relativePath) {
  if (relativePath === "index.html") return "/";
  if (relativePath.endsWith("/index.html")) return `/${relativePath.slice(0, -"index.html".length)}`;
  return `/${relativePath.slice(0, -".html".length)}`;
}

function routeExists(pathname, files) {
  const relativePath = decodeURIComponent(pathname).replace(/^\/+/, "");
  if (!relativePath) return files.has("index.html");
  if (relativePath.endsWith("/")) return files.has(`${relativePath}index.html`);
  return files.has(relativePath) || files.has(`${relativePath}.html`) || files.has(`${relativePath}/index.html`);
}

const requiredFiles = [
  "index.html",
  "404.html",
  "CNAME",
  "favicon.svg",
  "reme-logo.svg",
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

const homepage = await readFile(path.join(outputDir, "index.html"), "utf8");
assert.ok(homepage.includes('href="/en/"'), "root language switch must link to /en/");
assert.ok(!homepage.includes('href="/en/ex"'), "root language switch must not produce /en/ex");
assert.ok(homepage.includes('"studio-en":"/en/workspace/studio"'), "legacy redirects must be embedded");

const sitemap = await readFile(path.join(outputDir, "sitemap.xml"), "utf8");
assert.ok(sitemap.includes("<lastmod>"), "sitemap must include canonical-source update times");
assert.ok(!sitemap.includes("<loc>https://reme.agentscope.io/</loc>"), "root redirect must not be indexed");
assert.ok(!sitemap.includes('hreflang="zh-CN" href="https://reme.agentscope.io/"'), "root must not duplicate zh-CN");

const ChineseConfiguration = await readFile(path.join(outputDir, "zh/configuration.html"), "utf8");
assert.match(ChineseConfiguration, /搜索文档/);
assert.match(ChineseConfiguration, /复制 Markdown/);
assert.match(ChineseConfiguration, /在 GitHub 查看源文件/);

const jobReference = await readFile(path.join(outputDir, "en/reference/jobs.html"), "utf8");
assert.match(jobReference, /Job API Reference/);
assert.match(jobReference, /auto_memory/);

const outputFiles = new Set(await collectFiles(outputDir));
const missingLinks = [];
for (const relativePath of [...outputFiles].filter((file) => file.endsWith(".html"))) {
  const html = await readFile(path.join(outputDir, relativePath), "utf8");
  const currentUrl = new URL(pageUrl(relativePath), "https://reme-docs.local");
  for (const match of html.matchAll(/<a\b[^>]*\bhref="([^"]+)"/g)) {
    const href = match[1].replaceAll("&amp;", "&");
    if (href.startsWith("#")) continue;
    const target = new URL(href, currentUrl);
    if (target.origin !== currentUrl.origin) continue;
    if (!routeExists(target.pathname, outputFiles)) missingLinks.push(`${relativePath}: ${href}`);
  }
}
assert.deepEqual(missingLinks, [], `missing internal links:\n${missingLinks.join("\n")}`);

console.log(`Verified ${requiredFiles.length} documentation build artifacts.`);
