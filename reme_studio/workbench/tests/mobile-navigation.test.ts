import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { BrowserStorage } from "../../client/src/browser-host.ts";
import { ClientController } from "../../client/src/controller.ts";
import { agent, clientInfo } from "../../client/tests/fixtures.ts";

test("mobile pages share five bottom destinations, a central Remy and settings-owned updates", async (t) => {
  const controller = new ClientController(new BrowserStorage(), () => {});
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  t.after(async () => {
    await server.close();
    await controller.close();
  });
  const { default: Workbench } = await server.ssrLoadModule(
    "/workbench/src/workspace.tsx",
  );
  const { HostContext } = await server.ssrLoadModule("/workbench/src/host.ts");
  for (const language of ["zh", "en"] as const) {
    for (const activePage of [
      "remy",
      "tasks",
      "notebook",
      "knowledge",
      "settings",
    ] as const) {
      const state = {
        ...controller.state,
        settings: { ...controller.state.settings, language },
        connection: "connected",
        activePage,
        clientInfo,
        agents: [agent(), agent("proactive_memory")],
      };
      const html = renderToStaticMarkup(
        createElement(
          HostContext.Provider,
          { value: {} },
          createElement(Workbench, { state, version: "test", mobile: true }),
        ),
      );
      const nav = html.match(
        /<nav class="mobile-tabbar"[^>]*>(.*?)<\/nav>/s,
      )?.[1];
      assert.ok(nav);
      assert.deepEqual(
        [...nav.matchAll(/data-page="([^"]+)"/g)].map((match) => match[1]),
        ["tasks", "notebook", "remy", "knowledge", "settings"],
      );
      assert.match(
        nav,
        new RegExp(`data-page="${activePage}"[^>]*aria-current="page"`),
      );
      assert.doesNotMatch(
        html,
        /mobile-drawer|mobile-modules|mobile-sidebar-host/,
      );
      assert.match(html, /data-mobile-view="navigation"/);
      assert.match(
        html,
        language === "zh" ? /版本与更新/ : /Version &amp; updates/,
      );
      if (activePage === "remy") assert.match(html, /class="chat-composer"/);
      if (activePage === "knowledge") {
        assert.doesNotMatch(html, /knowledge-source-tabs/);
        const label = language === "zh" ? "记忆" : "Memory";
        const controls = [
          ...html.matchAll(
            new RegExp(
              `<nav class="source-tabs mobile-sections" aria-label="${label}"`,
              "g",
            ),
          ),
        ];
        assert.equal(
          controls.length,
          1,
          "memory sources share one navigation control",
        );
      }
    }
  }
});

test("the recording field distinguishes sending, sliding to cancel and canceling transcription", async (t) => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  t.after(() => server.close());
  const { default: RecordingField } = await server.ssrLoadModule(
    "/workbench/src/mobile/recording-field.tsx",
  );
  const render = (
    phase: "starting" | "recording" | "transcribing",
    canceling = false,
  ) =>
    renderToStaticMarkup(
      createElement(RecordingField, {
        phase,
        canceling,
        seconds: 3,
        level: 0.5,
        language: "zh",
        onCancel() {},
      }),
    );
  const recording = render("recording");
  assert.match(recording, /松手发送/);
  assert.match(recording, /上滑取消/);
  assert.match(recording, /0:03/);
  assert.doesNotMatch(recording, /取消发送/);
  const canceling = render("recording", true);
  assert.match(canceling, /松手取消/);
  assert.doesNotMatch(canceling, /松手发送/);
  const transcribing = render("transcribing");
  assert.match(transcribing, /正在识别并发送/);
  assert.match(transcribing, /aria-label="取消发送"/);
  assert.doesNotMatch(transcribing, /松手发送/);
  assert.match(render("starting"), /正在准备麦克风/);
});
