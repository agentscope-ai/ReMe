import type { IncomingMessage, ServerResponse } from "node:http";

import type { OpenClawReMeConfig } from "./config.js";
import type { ReMeClient } from "./reme/client.js";
import type { OpenClawReMeRuntime } from "./runtime.js";

const STATUS_PATH = "/plugins/reme/status";

interface StatusPageOptions {
  client: ReMeClient;
  config: OpenClawReMeConfig;
  runtime: OpenClawReMeRuntime;
}

export function createReMeStatusHandler(options: StatusPageOptions) {
  return async (request: IncomingMessage, response: ServerResponse) => {
    const pathname = new URL(
      request.url || "/",
      "http://localhost",
    ).pathname.replace(/\/$/, "");
    if (request.method === "GET" && pathname === STATUS_PATH) {
      sendHtml(response, renderStatusHtml(await statusPayload(options)));
      return;
    }
    if (request.method === "GET" && pathname === `${STATUS_PATH}/api/status`) {
      const payload = await statusPayload(options);
      sendJson(
        response,
        payload.reme.connected && payload.reme.healthy ? 200 : 503,
        payload,
      );
      return;
    }
    sendJson(response, 404, { error: "Not found" });
  };
}

async function statusPayload(options: StatusPageOptions) {
  const [health, status] = await Promise.all([
    options.client.requestJob("health_check"),
    options.client.requestJob("status"),
  ]);
  const healthDetails = health.metadata?.health as
    | Record<string, unknown>
    | undefined;
  return {
    endpoint: options.config.endpoint,
    runtime: options.runtime.snapshot(),
    reme: {
      connected: health.ok,
      healthy: health.ok && healthDetails?.healthy === true,
      health: health.metadata,
      status: status.metadata,
      error: health.error || status.error || "",
    },
  };
}

function sendHtml(response: ServerResponse, body: string): void {
  response.statusCode = 200;
  response.setHeader("Content-Type", "text/html; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'",
  );
  response.end(body);
}

function sendJson(
  response: ServerResponse,
  status: number,
  body: unknown,
): void {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify(body));
}

const STATUS_HTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReMe Memory</title><style>
:root{color-scheme:dark;--bg:#0b0d12;--panel:#12151d;--panel2:#181c26;--line:#292e3b;--text:#f3f5f7;--muted:#979dab;--green:#63d4a3;--blue:#79a8ff;--amber:#f2be5c;--red:#ff7b86}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% -10%,#202c42 0,transparent 32%),var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}.shell{max-width:1240px;margin:auto;padding:32px}.top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px}.brand{display:flex;gap:14px;align-items:center}.logo{width:46px;height:46px;border-radius:14px;display:grid;place-items:center;background:linear-gradient(145deg,#678ff7,#725de3);box-shadow:0 8px 26px #506ddd44;font-size:22px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:var(--blue);font-weight:700}h1{font-size:26px;margin:1px 0 0}.sub{color:var(--muted);margin-top:3px}.status{display:flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid var(--line);border-radius:99px;background:#10131a}.dot{width:8px;height:8px;border-radius:50%;background:var(--amber)}.dot.ok{background:var(--green);box-shadow:0 0 12px #63d4a377}.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:22px}.tab{color:var(--muted);text-decoration:none;padding:11px 15px;font-weight:650;border-bottom:2px solid transparent}.view{display:none}.view:target{display:block}.shell:not(:has(.view:target)) #overview{display:block}.shell:not(:has(.view:target)) .tab:first-child,body:has(#overview:target) a[href="#overview"],body:has(#memory:target) a[href="#memory"],body:has(#dream:target) a[href="#dream"],body:has(#components:target) a[href="#components"]{color:var(--text);border-color:var(--blue)}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:linear-gradient(155deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 10px 35px #0002}.wide{grid-column:span 2}.full{grid-column:1/-1}.label{color:var(--muted);font-size:12px}.metric{font-size:25px;font-weight:720;margin-top:6px}.metric.small{font-size:15px;word-break:break-all}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}h2{font-size:15px;margin:0 0 14px}.flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.step{background:#0d1017;border:1px solid var(--line);border-radius:10px;padding:12px 15px;min-width:145px}.arrow{color:#596172}.rows{display:grid;gap:8px}.row{display:grid;grid-template-columns:1.3fr .8fr .8fr 1fr;gap:12px;align-items:center;padding:10px 12px;background:#0e1118;border:1px solid #242936;border-radius:9px}.pill{display:inline-flex;width:max-content;padding:3px 8px;border-radius:99px;background:#25382f;color:var(--green);font-size:11px;font-weight:700}.pill.failed{background:#40262c;color:var(--red)}.pill.running{background:#3c3526;color:var(--amber)}.empty{color:var(--muted);padding:20px 0}.component-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.component{padding:13px;background:#0e1118;border:1px solid var(--line);border-radius:10px}.component strong{display:block;margin-bottom:3px}.footer{margin-top:22px;color:#747b89;font-size:12px}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}.component-grid{grid-template-columns:1fr 1fr}}@media(max-width:560px){.shell{padding:20px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{gap:15px;flex-direction:column}.tabs{overflow:auto}.row{grid-template-columns:1fr 1fr}}
</style></head><body><main class="shell"><header class="top"><div class="brand"><div class="logo">◈</div><div><div class="eyebrow">OpenClaw Memory Provider</div><h1>ReMe</h1><div class="sub">File-native memory, recall, and consolidation</div></div></div><div class="status"><span class="dot" id="dot"></span><span id="connection">Connecting…</span></div></header>
<nav class="tabs"><a class="tab" href="#overview">Overview</a><a class="tab" href="#memory">Auto Memory</a><a class="tab" href="#dream">Memory Consolidation</a><a class="tab" href="#components">Components</a></nav>
<section class="view" id="overview"><div class="grid"><article class="card"><div class="label">ReMe service</div><div class="metric {{healthClass}}">{{health}}</div></article><article class="card"><div class="label">Runtime phase</div><div class="metric">{{phase}}</div></article><article class="card"><div class="label">Indexed documents</div><div class="metric">{{documents}}</div></article><article class="card"><div class="label">Active sessions</div><div class="metric">{{sessions}}</div></article><article class="card wide"><h2>Memory automation</h2><div class="flow"><div class="step"><div class="label">1 · Prompt</div>Automatic recall</div><span class="arrow">→</span><div class="step"><div class="label">2 · Response</div>Turn capture</div><span class="arrow">→</span><div class="step"><div class="label">3 · Workspace</div>File-native memory</div></div></article><article class="card wide"><h2>Connection</h2><div class="label">ReMe endpoint</div><div class="metric small">{{endpoint}}</div><div class="label" style="margin-top:14px">ReMe version</div><div>{{version}}</div></article></div></section>
<section class="view" id="memory"><div class="grid"><article class="card"><div class="label">Automatic capture</div><div class="metric">{{capture}}</div></article><article class="card"><div class="label">Batch interval</div><div class="metric">{{interval}}</div></article><article class="card"><div class="label">Queued turns</div><div class="metric">{{queued}}</div></article><article class="card"><div class="label">Active sessions</div><div class="metric">{{sessions}}</div></article><article class="card full"><h2>Recent capture activity</h2><div class="rows">{{activity}}</div></article></div></section>
<section class="view" id="dream"><div class="grid"><article class="card"><div class="label">Auto Dream</div><div class="metric">{{dreamEnabled}}</div></article><article class="card"><div class="label">Schedule</div><div class="metric small">{{cron}}</div></article><article class="card"><div class="label">Timezone</div><div class="metric small">{{timezone}}</div></article><article class="card"><div class="label">Last result</div><div class="metric small">{{dreamResult}}</div></article><article class="card wide"><h2>Next consolidation</h2><div class="metric small">{{nextDream}}</div><p class="sub">Auto Dream evolves daily notes into durable knowledge while workspace files remain the source of truth.</p></article><article class="card wide"><h2>Operator verification</h2><p class="sub">Wait for the configured schedule, then reload this page and inspect the ReMe workspace digest.</p></article></div></section>
<section class="view" id="components"><article class="card"><h2>ReMe component health</h2><div class="component-grid">{{components}}</div></article></section><div class="footer">Live diagnostics are served through OpenClaw's authenticated plugin route. No model credentials are exposed.</div></main>
</body></html>`;

function renderStatusHtml(
  payload: Awaited<ReturnType<typeof statusPayload>>,
): string {
  const healthRoot = payload.reme.health as Record<string, unknown>;
  const health =
    (healthRoot.health as Record<string, unknown> | undefined) || healthRoot;
  const components =
    (health.components as
      | Record<string, Record<string, Record<string, unknown>>>
      | undefined) || {};
  const documents = components.file_store?.default?.n_chunks ?? "—";
  const values: Record<string, unknown> = {
    health: payload.reme.connected
      ? payload.reme.healthy
        ? "Healthy"
        : "Unhealthy"
      : "Offline",
    healthClass: payload.reme.healthy ? "good" : "bad",
    phase: payload.runtime.phase,
    documents,
    sessions: payload.runtime.autoMemory.activeSessions,
    endpoint: payload.endpoint,
    version: health.version || "—",
    capture: payload.runtime.autoMemory.enabled ? "Enabled" : "Disabled",
    interval: `${payload.runtime.autoMemory.interval} turn${
      payload.runtime.autoMemory.interval === 1 ? "" : "s"
    }`,
    queued: payload.runtime.autoMemory.queuedTurns,
    dreamEnabled: payload.runtime.autoDream.enabled ? "Enabled" : "Disabled",
    cron: payload.runtime.autoDream.cron,
    timezone: payload.runtime.autoDream.timezone,
    dreamResult: payload.runtime.autoDream.running
      ? "Running…"
      : payload.runtime.autoDream.lastResult || "Not run yet",
    nextDream: payload.runtime.autoDream.nextRunAt || "—",
  };
  let html = STATUS_HTML.replace(
    'class="dot"',
    `class="dot${payload.reme.connected ? " ok" : ""}"`,
  ).replace(
    "Connecting…",
    payload.reme.connected ? "Connected" : "Unavailable",
  );
  for (const [key, value] of Object.entries(values)) {
    html = html.replaceAll(`{{${key}}}`, escapeHtml(String(value)));
  }
  const activity = payload.runtime.autoMemory.recentActivity;
  html = html.replace(
    "{{activity}}",
    activity.length
      ? activity
          .map(
            (entry) =>
              `<div class="row"><strong>Capture #${entry.id}</strong><span>${
                entry.turns
              } turn${entry.turns === 1 ? "" : "s"}</span><span class="pill ${
                entry.status
              }">${entry.status}</span><span>${escapeHtml(
                entry.completedAt || entry.startedAt,
              )}</span></div>`,
          )
          .join("")
      : '<div class="empty">Completed conversational turns will appear here.</div>',
  );
  const componentCards = Object.entries(components).flatMap(
    ([group, entries]) => {
      const instances = Object.entries(entries);
      if (!instances.length) {
        return [
          `<div class="component"><strong>${escapeHtml(
            group.replaceAll("_", " "),
          )}</strong><span class="good">Healthy</span></div>`,
        ];
      }
      return instances.map(([name, details]) => {
        const healthy =
          details.is_started === true && details.is_healthy !== false;
        const healthLabel =
          details.is_started !== true
            ? "Stopped"
            : healthy
            ? "Healthy"
            : "Unhealthy";
        const summary = Object.entries(details)
          .filter(([key]) => !["is_started", "is_healthy"].includes(key))
          .slice(0, 2)
          .map(
            ([key, value]) =>
              `${escapeHtml(key.replaceAll("_", " "))}: ${escapeHtml(
                String(value),
              )}`,
          )
          .join(" · ");
        return `<div class="component"><strong>${escapeHtml(
          group.replaceAll("_", " "),
        )} · ${escapeHtml(name)}</strong><span class="${
          healthy ? "good" : "bad"
        }">${healthLabel}</span><div class="label">${summary}</div></div>`;
      });
    },
  );
  html = html.replace(
    "{{components}}",
    componentCards.join("") ||
      '<div class="empty">No component diagnostics returned.</div>',
  );
  return html;
}

function escapeHtml(value: string): string {
  return value.replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[character] || character,
  );
}
