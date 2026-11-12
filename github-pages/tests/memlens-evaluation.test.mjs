import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const generatedDir = path.join(siteDir, ".generated/site");
const repoDir = path.resolve(siteDir, "..");
const figureDir = path.join(repoDir, "benchmark/memlens/docs/figure");
const methods = ["Full Context Image", "ReMe Text", "ReMe Image", "OpenViking", "Mem0"];
// Published binary verdict counts, not raw benchmark outputs or a second chart generator.
const models = [
  { name: "Qwen3.6-35B-A3B", correct: [89, 78, 101, 70, 73] },
  { name: "Qwen3.8-Max", correct: [124, 100, 137, 80, 97] },
  { name: "Kimi-K3", correct: [108, 90, 119, 91, 98] },
  { name: "DeepSeek-V4.1-Flash", correct: [114, 78, 117, 93, 93] },
];
const excluded = ["q_028cb832", "q_1d99a36c", "q_3b01c02a", "q_564a657b", "q_600a79aa", "q_c9f8e24d"];
const publicRevision = "b784468a10d98af1688b7c4ba38492dc1f9bda6f";
const reproductionLinks = [
  ...[
    "benchmark/memlens/README.md",
    "plugins/memlens/src/reme_memlens/benchmark.yaml",
    "plugins/memlens-judge/src/judge_memlens/plugin.yaml",
    "plugins/memlens-judge/src/judge_memlens/protocol.py",
  ].map((file) => `https://github.com/agentscope-ai/ReMe/blob/${publicRevision}/${file}`),
  `https://github.com/agentscope-ai/ReMe/commit/${publicRevision}`,
  "https://github.com/agentscope-ai/ReMe/pull/713",
  "https://github.com/agentscope-ai/ReMe/pull/735",
  "https://github.com/agentscope-ai/ReMe/commit/681a13831426c1a8d4f7227ef835c38d0257003a",
];

test("four-model chart displays every published MemLens score", async () => {
  const svg = await readFile(path.join(figureDir, "memlens-256k-four-models-judge.svg"), "utf8");
  const panels = [...svg.matchAll(/<g data-model="([^"]+)"[^>]*>([\s\S]*?)\n  <\/g>/g)];
  assert.equal(panels.length, models.length);
  for (const [index, model] of models.entries()) {
    assert.equal(panels[index][1], model.name);
    const bars = [...panels[index][2].matchAll(/<g data-method="([^"]+)" data-score="([\d.]+)">[\s\S]*?<rect[^>]*width="([\d.]+)"[^>]*\/>\s*<text[^>]*>([\d.]+)%<\/text>/g)];
    assert.equal(bars.length, methods.length);
    for (const [methodIndex, method] of methods.entries()) {
      const score = (100 * model.correct[methodIndex] / 189).toFixed(2);
      assert.equal(bars[methodIndex][1], method);
      assert.equal(bars[methodIndex][2], score);
      assert.equal(bars[methodIndex][4], score);
      assert.ok(Math.abs(Number(bars[methodIndex][3]) - Number(score) * 2.9) < 0.01);
    }
  }
});

function tableRows(content) {
  return content.split("\n").filter((line) => line.startsWith("| "))
    .map((line) => line.split("|").slice(1, -1).map((cell) => cell.trim().replaceAll("**", "")));
}

function assertReproduction(notes, total, results) {
  const section = notes.split("{#independent-reproduction}")[1].split("\n## ")[0];
  const expectedCount = total.toLocaleString("en-US");
  const rows = tableRows(section).filter((row) => row[1]?.endsWith(` / ${expectedCount}`));
  assert.deepEqual(rows.map((row) => row.slice(0, 4)), results.map(([model, scored, correct]) => [
    model, `${scored.toLocaleString("en-US")} / ${expectedCount}`, String(correct), (100 * correct / scored).toFixed(2),
  ]));
}

test("MemLens bilingual notes and blogs match the published scores", async () => {
  for (const language of ["en", "zh"]) {
    const blog = await readFile(path.join(generatedDir, language, "blog_20261009.md"), "utf8");
    const notes = await readFile(path.join(generatedDir, language, "benchmarks/memlens.md"), "utf8");
    const results = notes.split("{#results}")[1].split("{#evaluation-settings}")[0];
    const rows = tableRows(results).filter((row) => /^\d+ \/ 189$/.test(row[5]));
    assert.equal(rows.length, 20);
    for (const model of models) {
      for (const [index, method] of methods.entries()) {
        const correct = model.correct[index];
        const score = (100 * correct / 189).toFixed(2);
        const row = rows.find((entry) => entry[0] === model.name && entry[1] === method);
        assert.ok(row, `${language}: ${model.name}/${method}`);
        assert.equal(row[5], `${correct} / 189`);
        assert.equal(row[6], score);
      }
      assert.ok(blog.includes(`**${(100 * model.correct[2] / 189).toFixed(2)}%**`));
      const gain = (100 * (model.correct[2] - model.correct[0]) / 189).toFixed(2);
      assert.ok(blog.includes(`**${gain}**`));
    }
  }
});

test("MemLens notes disclose moderation filters, reproduction results and Judge settings", async () => {
  for (const language of ["en", "zh"]) {
    const notes = await readFile(path.join(generatedDir, language, "benchmarks/memlens.md"), "utf8");
    const filter = notes.split("{#content-moderation-filter}")[1].split("{#independent-reproduction}")[0];
    const excludedIds = tableRows(filter)
      .map(([id]) => id.replaceAll("`", ""))
      .filter((id) => /^q_[0-9a-f]{8}$/.test(id));
    assert.deepEqual(excludedIds.sort(), excluded);
    assert.match(notes, /189\/195/);
    assertReproduction(notes, 189, [
      ["Qwen3.6-35B-A3B", 189, 104],
      ["Qwen3.8-Max", 189, 133],
      ["Kimi-K3", 189, 123],
      ["DeepSeek-V4.1-Flash", 184, 119],
    ]);
    const judge = notes.split("\n\n").find((paragraph) => /(?:as|作为) Judge/.test(paragraph));
    assert.ok(judge, `${language}: Judge settings must be documented`);
    assert.match(judge, /\*\*Qwen3\.8-Max\*\*/);
    assert.match(judge, /temperature (?:为 )?\*\*0\*\*/);
    assert.match(judge, /thinking \*\*(?:disabled|关闭)\*\*/);
    assert.match(judge, /\*\*256(?:-token| tokens)\*\*/);
  }
});

test("MemLens notes pin the public evaluation revision without private audit details", async () => {
  for (const language of ["en", "zh"]) {
    const blog = await readFile(path.join(generatedDir, language, "blog_20261009.md"), "utf8");
    const notes = await readFile(path.join(generatedDir, language, "benchmarks/memlens.md"), "utf8");
    assert.ok(blog.includes(`](../../benchmark/memlens/docs/${language}/evaluation.md)`));
    assert.ok(notes.includes(`](../../../../docs/${language}/blog_20261009.md#memlens-256k-results)`));
    assert.doesNotMatch(notes, /\/mnt\/|\/Users\/|docs\/_build|file:\/\//);
    const reproduction = notes.split("{#independent-reproduction}")[1].split("\n## ")[0];
    const targets = [...reproduction.matchAll(/\]\(([^)\r\n]+)\)/g)].map(([, target]) => target);
    assert.deepEqual(targets.sort(), [...reproductionLinks].sort());
    for (const anchor of ["results", "evaluation-settings", "content-moderation-filter", "independent-reproduction"]) {
      assert.ok(notes.includes(`{#${anchor}}`));
    }
  }
});

test("MobileMem publishes seven results over 1,333 questions in both languages", async () => {
  const results = [
    ["Qwen3.8-Max", "Full Context Image", 1333, 624],
    ["Qwen3.8-Max", "ReMe Text", 1330, 511],
    ["Qwen3.8-Max", "ReMe Image", 1332, 809],
    ["Kimi-K3", "ReMe Text", 1333, 478],
    ["Kimi-K3", "ReMe Image", 1333, 682],
    ["DeepSeek-V4.1-Flash", "ReMe Text", 1333, 525],
    ["DeepSeek-V4.1-Flash", "ReMe Image", 1333, 749],
  ];
  for (const language of ["en", "zh"]) {
    const notes = await readFile(path.join(generatedDir, language, "benchmarks/mobilemem.md"), "utf8");
    const resultSection = notes.split("{#results}")[1].split("{#evaluation-settings}")[0];
    const rows = tableRows(resultSection).filter((row) => /^\d+ \/ 1,333$/.test(row[3]));
    assert.equal(rows.length, results.length);
    for (const [model, method, valid, correct] of results) {
      const row = rows.find((entry) => entry[0] === model && entry[1].startsWith(method));
      assert.ok(row, `${language}: ${model}/${method}`);
      assert.equal(Number(row[2].replaceAll(",", "")), valid);
      assert.equal(row[3], `${correct} / 1,333`);
      assert.equal(row[4], (100 * correct / 1333).toFixed(2));
    }
    assertReproduction(notes, 1333, [
      ["Qwen3.8-Max", 1329, 803],
      ["Kimi-K3", 1278, 669],
      ["DeepSeek-V4.1-Flash", 1328, 712],
    ]);
    assert.match(notes, /8_69/);
    assert.match(notes, /14_74_2/);
    assert.match(notes, /1,203/);
    assert.match(notes, /N\/A/);
    assert.doesNotMatch(notes, /\/mnt\/|\/Users\/|file:\/\//);
  }
});
