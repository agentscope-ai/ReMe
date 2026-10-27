import { createHash, randomUUID } from "node:crypto";
import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  opendirSync,
  readFileSync,
  renameSync,
  rmdirSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import type { ReMeMessage } from "./reme/types.js";
import { validEndpoint } from "./endpoint.js";

export interface MemoryBatch {
  version: 1;
  id: string;
  createdAt: number;
  endpoint: string;
  sessionId: string;
  date: string;
  turns: ReMeMessage[][];
}

/** Private JSON outbox: seal batches before sending and remove only after acknowledgement. */
export class MemoryOutbox {
  private readonly lock: string;
  private readonly owner = `${process.pid}:${randomUUID()}`;
  private latestCreatedAt = 0;

  constructor(readonly directory: string) {
    mkdirSync(directory, { recursive: true, mode: 0o700 });
    this.lock = join(directory, ".lock");
    // Serialize inspection, stale-owner removal, and replacement as one operation.
    // An interrupted recovery fails closed; never reclaim this guard automatically.
    const guard = join(directory, ".lock-recovery");
    try {
      mkdirSync(guard, { mode: 0o700 });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      throw new Error(
        "ReMe outbox lock recovery is busy or interrupted; retry, or inspect .lock-recovery after stopping all Hosts",
      );
    }
    try {
      if (existsSync(this.lock)) {
        const pid = Number(readFileSync(this.lock, "utf8").split(":")[0]);
        if (!Number.isSafeInteger(pid) || pid <= 0)
          throw new Error(
            "Invalid ReMe outbox lock; inspect the queue directory",
          );
        try {
          process.kill(pid, 0);
          throw new Error(
            "ReMe outbox is in use; configure a separate autoMemoryQueueDir for each Host",
          );
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
        }
        unlinkSync(this.lock);
      }
      const fd = openSync(this.lock, "wx", 0o600);
      try {
        writeFileSync(fd, this.owner);
      } finally {
        closeSync(fd);
      }
      for (const name of this.files())
        this.latestCreatedAt = Math.max(
          this.latestCreatedAt,
          this.read(name).createdAt,
        );
    } catch (error) {
      this.close();
      throw error;
    } finally {
      rmdirSync(guard);
    }
  }

  append(
    endpoint: string,
    sessionId: string,
    date: string,
    messages: ReMeMessage[],
    interval: number,
  ): void {
    const key = createHash("sha256")
      .update(JSON.stringify([endpoint, sessionId, date]))
      .digest("hex");
    const name = `partial-${key}.json`;
    const path = join(this.directory, name);
    const batch: MemoryBatch = existsSync(path)
      ? this.read(name)
      : {
          version: 1,
          id: randomUUID(),
          // Keep FIFO ordering when the clock repeats or moves backwards, including after restart.
          createdAt: (this.latestCreatedAt = Math.max(
            Date.now(),
            this.latestCreatedAt + 1,
          )),
          endpoint,
          sessionId,
          date,
          turns: [],
        };
    batch.turns.push(messages);
    this.write(name, batch);
    if (batch.turns.length >= interval) this.seal(name, batch);
    // Finish the previous workspace day without combining dates in one request.
    for (const file of this.files()) {
      if (!file.startsWith("partial-") || file === name) continue;
      const previous = this.read(file);
      if (
        previous.endpoint === endpoint &&
        previous.sessionId === sessionId &&
        previous.date !== date
      )
        this.seal(file, previous);
    }
  }

  flush(sessionId?: string): void {
    for (const name of this.files()) {
      if (!name.startsWith("partial-")) continue;
      const batch = this.read(name);
      if (sessionId === undefined || batch.sessionId === sessionId)
        this.seal(name, batch);
    }
  }

  next(
    accepts: (batch: MemoryBatch) => boolean = () => true,
  ): { name: string; batch: MemoryBatch } | undefined {
    let oldest: string | undefined;
    for (const name of this.files()) {
      if (
        name.startsWith("ready-") &&
        (oldest === undefined || name < oldest) &&
        accepts(this.read(name))
      )
        oldest = name;
    }
    return oldest === undefined
      ? undefined
      : { name: oldest, batch: this.read(oldest) };
  }

  acknowledge(name: string): void {
    unlinkSync(join(this.directory, name));
  }

  stats(): { turns: number; batches: number } {
    let turns = 0;
    let batches = 0;
    for (const name of this.files()) {
      turns += this.read(name).turns.length;
      if (name.startsWith("ready-")) batches += 1;
    }
    return { turns, batches };
  }

  close(): void {
    if (existsSync(this.lock) && readFileSync(this.lock, "utf8") === this.owner)
      unlinkSync(this.lock);
  }

  private *files(): Generator<string> {
    const dir = opendirSync(this.directory);
    try {
      for (let entry = dir.readSync(); entry; entry = dir.readSync())
        if (entry.isFile() && /^(partial|ready)-.*\.json$/.test(entry.name))
          yield entry.name;
    } finally {
      dir.closeSync();
    }
  }

  private read(name: string): MemoryBatch {
    try {
      const batch = JSON.parse(
        readFileSync(join(this.directory, name), "utf8"),
      ) as MemoryBatch;
      if (
        batch.version !== 1 ||
        typeof batch.id !== "string" ||
        !/^[a-f0-9-]{36}$/.test(batch.id) ||
        !validEndpoint(batch.endpoint) ||
        typeof batch.sessionId !== "string" ||
        typeof batch.date !== "string" ||
        !Number.isFinite(batch.createdAt) ||
        !Array.isArray(batch.turns) ||
        !batch.turns.length ||
        !batch.turns.every(
          (turn) =>
            Array.isArray(turn) &&
            turn.every(
              (message) =>
                typeof message.id === "string" &&
                ["user", "assistant"].includes(message.role) &&
                Array.isArray(message.content) &&
                message.content.every(
                  (part) =>
                    part.type === "text" && typeof part.text === "string",
                ),
            ),
        )
      )
        throw new Error(
          "Invalid ReMe outbox record; preserve the file and inspect the queue directory",
        );
      return batch;
    } catch {
      // JSON parser errors can quote message content; leave the record untouched.
      throw new Error(
        "Invalid ReMe outbox record; preserve the file and inspect the queue directory",
      );
    }
  }

  private write(name: string, batch: MemoryBatch): void {
    const temporary = join(this.directory, `${randomUUID()}.tmp`);
    const fd = openSync(temporary, "wx", 0o600);
    try {
      writeFileSync(fd, JSON.stringify(batch));
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    renameSync(temporary, join(this.directory, name));
  }

  private seal(name: string, batch: MemoryBatch): void {
    renameSync(
      join(this.directory, name),
      join(this.directory, `ready-${batch.createdAt}-${batch.id}.json`),
    );
  }
}
