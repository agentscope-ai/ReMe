import assert from "node:assert/strict";
import test from "node:test";
import { spawn, spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { MemoryOutbox } from "../dist/outbox.js";

function directory(t) {
  const path = mkdtempSync(join(tmpdir(), "reme-outbox-"));
  t.after(() => rmSync(path, { recursive: true, force: true }));
  return path;
}

const messages = [
  {
    id: "m1",
    name: "user",
    role: "user",
    content: [{ type: "text", text: "private fact" }],
  },
];

test("preserves batch order across equal timestamps, clock rollback, and restart", (t) => {
  const now = Date.parse("2026-10-21T12:00:00Z");
  t.mock.timers.enable({ apis: ["Date"], now });
  const path = directory(t);
  let queue = new MemoryOutbox(path);
  const append = (id) =>
    queue.append(
      "http://localhost:2333",
      "session",
      "2026-10-21",
      [{ ...messages[0], id }],
      1,
    );
  try {
    append("first");
    append("second");
    const batches = readdirSync(path)
      .filter((name) => name.startsWith("ready-"))
      .map((name) => JSON.parse(readFileSync(join(path, name), "utf8")));
    assert.equal(new Set(batches.map((batch) => batch.createdAt)).size, 2);
    queue.close();
    t.mock.timers.setTime(now - 1000);
    queue = new MemoryOutbox(path);
    append("third");
    const received = [];
    for (let next = queue.next(); next; next = queue.next()) {
      received.push(next.batch.turns[0][0].id);
      queue.acknowledge(next.name);
    }
    assert.deepEqual(received, ["first", "second", "third"]);
  } finally {
    queue.close();
  }
});

test("recovers a partial batch after the owning process exits without cleanup", (t) => {
  const path = directory(t);
  const child = spawnSync(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      `
    import { MemoryOutbox } from ${JSON.stringify(
      new URL("../dist/outbox.js", import.meta.url).href,
    )};
    const queue = new MemoryOutbox(${JSON.stringify(path)});
    queue.append("http://localhost:2333", "session", "2026-10-21", ${JSON.stringify(
      messages,
    )}, 5);
  `,
    ],
    { encoding: "utf8" },
  );
  assert.equal(child.status, 0, child.stderr);
  const recovered = new MemoryOutbox(path);
  try {
    recovered.flush();
    assert.deepEqual(recovered.stats(), { turns: 1, batches: 1 });
    const { name, batch } = recovered.next();
    assert.deepEqual(batch.turns, [messages]);
    if (process.platform !== "win32")
      assert.equal(statSync(join(path, name)).mode & 0o777, 0o600);
    recovered.acknowledge(name);
    assert.deepEqual(recovered.stats(), { turns: 0, batches: 0 });
  } finally {
    recovered.close();
  }
});

test("refuses another owner without changing the queue", (t) => {
  const path = directory(t);
  const queue = new MemoryOutbox(path);
  try {
    queue.append("http://localhost:2333", "session", "2026-10-21", messages, 1);
    assert.throws(() => new MemoryOutbox(path), /in use/);
    assert.equal(queue.stats().turns, 1);
  } finally {
    queue.close();
  }
});

test("serializes concurrent stale-lock recovery without deleting a replacement owner", async (t) => {
  const path = directory(t);
  const moduleUrl = new URL("../dist/outbox.js", import.meta.url).href;
  const seed = spawnSync(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      `import { MemoryOutbox } from ${JSON.stringify(moduleUrl)};
       new MemoryOutbox(${JSON.stringify(path)});`,
    ],
    { encoding: "utf8" },
  );
  assert.equal(seed.status, 0, seed.stderr);
  const staleOwner = readFileSync(join(path, ".lock"), "utf8");
  const child = spawn(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      `
      import { existsSync, writeFileSync } from "node:fs";
      import { join } from "node:path";
      import { MemoryOutbox } from ${JSON.stringify(moduleUrl)};
      const path = ${JSON.stringify(path)};
      const kill = process.kill.bind(process);
      process.kill = (pid, signal) => {
        writeFileSync(join(path, "observed-stale"), "ready");
        const deadline = Date.now() + 5000;
        while (!existsSync(join(path, "release"))) {
          if (Date.now() > deadline) throw new Error("Recovery probe timed out");
          Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5);
        }
        return kill(pid, signal);
      };
      const queue = new MemoryOutbox(path);
      process.on("message", () => { queue.close(); process.exit(0); });
      writeFileSync(join(path, "acquired"), "ready");
      `,
    ],
    { stdio: ["ignore", "ignore", "pipe", "ipc"] },
  );
  let stderr = "";
  child.stderr.on("data", (data) => {
    stderr += data;
  });
  const exited = new Promise((resolve) => child.once("exit", resolve));
  async function waitFor(name) {
    const deadline = Date.now() + 5000;
    while (!existsSync(join(path, name))) {
      assert.equal(child.exitCode, null, stderr);
      assert.ok(
        Date.now() < deadline,
        `Timed out waiting for ${name}: ${stderr}`,
      );
      await delay(5);
    }
  }
  try {
    await waitFor("observed-stale");
    assert.throws(() => new MemoryOutbox(path), /lock recovery is busy/);
    assert.equal(readFileSync(join(path, ".lock"), "utf8"), staleOwner);
    writeFileSync(join(path, "release"), "ready");
    await waitFor("acquired");
    assert.equal(
      Number(readFileSync(join(path, ".lock"), "utf8").split(":")[0]),
      child.pid,
    );
    assert.throws(() => new MemoryOutbox(path), /in use/);
  } finally {
    writeFileSync(join(path, "release"), "ready");
    if (child.connected) child.send("close");
    const timer = setTimeout(() => child.kill(), 1000);
    await exited;
    clearTimeout(timer);
  }
});

test("fails closed on an interrupted recovery guard without touching the owner or records", (t) => {
  const path = directory(t);
  const queue = new MemoryOutbox(path);
  try {
    queue.append("http://localhost:2333", "session", "2026-10-21", messages, 1);
    const owner = readFileSync(join(path, ".lock"), "utf8");
    mkdirSync(join(path, ".lock-recovery"));
    assert.throws(
      () => new MemoryOutbox(path),
      /lock recovery is busy or interrupted/,
    );
    assert.equal(readFileSync(join(path, ".lock"), "utf8"), owner);
    assert.equal(queue.stats().turns, 1);
    assert.equal(existsSync(join(path, ".lock-recovery")), true);
  } finally {
    queue.close();
  }
});

test("preserves invalid records and keeps parser errors free of message content", (t) => {
  const path = directory(t);
  const queue = new MemoryOutbox(path);
  try {
    queue.append("http://localhost:2333", "session", "2026-10-21", messages, 1);
    const name = readdirSync(path).find((file) => file.startsWith("ready-"));
    const stored = JSON.parse(readFileSync(join(path, name), "utf8"));
    stored.id = "../outside";
    writeFileSync(join(path, name), JSON.stringify(stored));
    assert.throws(() => queue.next(), /Invalid ReMe outbox record/);
    writeFileSync(join(path, name), "PRIVATE-CONTENT");
    assert.throws(
      () => queue.next(),
      (error) => !error.message.includes("PRIVATE-CONTENT"),
    );
    assert.equal(readFileSync(join(path, name), "utf8"), "PRIVATE-CONTENT");
    queue.close();
    assert.throws(() => new MemoryOutbox(path), /Invalid ReMe outbox record/);
    assert.equal(existsSync(join(path, ".lock")), false);
    assert.equal(readFileSync(join(path, name), "utf8"), "PRIVATE-CONTENT");
  } finally {
    queue.close();
  }
});
