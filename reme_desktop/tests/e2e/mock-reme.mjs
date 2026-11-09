import http from "node:http";

export function toneWave(seconds = 2) {
  const samples = Math.round(16000 * seconds);
  const wave = Buffer.alloc(44 + samples * 2);
  wave.write("RIFF");
  wave.writeUInt32LE(wave.length - 8, 4);
  wave.write("WAVEfmt ", 8);
  wave.writeUInt32LE(16, 16);
  wave.writeUInt16LE(1, 20);
  wave.writeUInt16LE(1, 22);
  wave.writeUInt32LE(16000, 24);
  wave.writeUInt32LE(32000, 28);
  wave.writeUInt16LE(2, 32);
  wave.writeUInt16LE(16, 34);
  wave.write("data", 36);
  wave.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++)
    wave.writeInt16LE(
      Math.round(Math.sin((i * Math.PI * 2 * 440) / 16000) * 3000),
      44 + i * 2,
    );
  return wave;
}

export async function mockReMe({
  speech = false,
  workspace = false,
  ttsSeconds = 2,
  taskResults = [],
} = {}) {
  let taskIds = [...new Set(taskResults.map((item) => item.task_id))];
  const documents = {
    "digest/wiki/agent-memory.md":
      "---\nname: Agent 记忆，从记录到连接\ndescription: 让零散的信息成为可以被再次理解、关联和使用的知识。\nmemory_tags:\n  - Agent\n  - memory\n---\n## 记忆的价值，在于再次连接\n\n一份好的记忆，不只保存发生过什么，也帮助我们理解下一步可以做什么。\n\n> 文件属于你，索引可以重建。\n\n### 一个持续生长的知识库\n\n- **记录**：保留值得记住的事实与想法。\n- **关联**：通过标签与链接发现共同主题。\n- **回顾**：在需要的时候，让过去的经验重新发挥作用。\n\n参见 [[daily/2026-11-08/research.md|今天的研究笔记]]。\n\n| 来源 | 作用 |\n| --- | --- |\n| 每日记忆 | 保留上下文 |\n| 知识库 | 沉淀可复用知识 |\n",
    "daily/2026-11-08/research.md":
      "---\nname: 今天的研究笔记\nmemory_tags: [research, agent]\n---\n## 从想法开始\n\n记录下一个值得继续的问题。\n\n回到 [[digest/wiki/agent-memory.md|Agent 记忆]]。",
    "digest/personal/preferences.md":
      "---\nname: 我的工作习惯\nmemory_tags: [personal]\n---\n## 留出思考的空间\n\n喜欢清晰、安静、有条理的工作界面。",
    "daily/long-title.md":
      "---\ncreated_at: '2026-11-08T10:53:47+08:00'\nstatus: unsent\ntarget_session_id: example-session-id\nmemory_tags: [research]\ncontext:\n  source: local\n  reviewed: true\n---\n# 研究笔记：把零散记录连接成知识，从每日观察、主题标签与文档链接出发，在持续回顾中保留事实、整理线索并发现下一步值得探索的问题，让每一份记忆都能在需要的时候被重新理解与使用\n\n## 观察与记录\n\n这是一份用于验证长标题和文档属性的示例笔记。\n\n| 字段 | 说明 |\n| --- | --- |\n| 记录 | 保留上下文 |\n\n```yaml\nsource: local\n```\n\n" +
      Array.from(
        { length: 12 },
        (_, i) => `### 回顾 ${i + 1}\n\n记录事实，建立联系，留待下一次回顾。\n`,
      ).join("\n"),
    "daily/invalid-yaml.md":
      "---\nmemory_tags: [unclosed\n---\n# 属性格式有误的笔记\n\n正文依然可以阅读。",
  };
  const ttsRequests = [];
  let tagItems = [
    ["agent", 2],
    ["memory", 1],
    ["personal", 1],
    ["research", 1],
  ];
  const asrRequests = [];
  let speechError = false;
  let speechErrorText = "";
  let asrText = "语音输入的研究计划。";
  let speechDelay = 0;
  let replyDelay = 300;
  const sockets = new Set();
  const events = [];
  const cursors = [];
  let sessionId = "session-1";
  let messages = [];
  let status = "idle";
  let activities = {};
  let taskActivities = {};
  let notifications = {
    enabled: true,
    starts_at: new Date(Date.now() - 3_600_000).toISOString(),
    ends_at: new Date(Date.now() + 3_600_000).toISOString(),
    closing_message_id: null,
  };
  let turn = 0;
  const frame = (response, chunk) =>
    response.write(`data: ${JSON.stringify(chunk)}\n\n`);
  const emit = (event) => {
    const value = structuredClone({
      ...event,
      task_activities: taskActivities,
      notifications,
      session_id: sessionId,
      cursor: `test-epoch:${events.length + 1}`,
    });
    events.push(value);
    for (const socket of sockets)
      frame(socket, { chunk_type: "data", chunk: value });
    return value;
  };
  const snapshot = () => ({
    session_id: sessionId,
    state: { session_id: sessionId, status, active_flags: [] },
    messages,
  });
  const upsert = (message) => {
    const index = messages.findIndex((entry) => entry.id === message.id);
    if (index < 0) messages = [...messages, message];
    else messages = messages.map((entry, i) => (i === index ? message : entry));
    emit({ type: "message_upsert", message });
  };
  const push = (
    id,
    content = "你关注的 Agent 记忆论文有更新，想看看重点吗？",
    completed = true,
  ) => {
    upsert({
      id,
      role: "assistant",
      source: "proactive",
      content,
      delivery: "idle",
      notification_eligible: true,
      completed,
      chunks: [],
      created_at: new Date().toISOString(),
    });
  };
  // Historical reminders establish a quiet baseline on the first connection.
  messages.push({
    id: "historical",
    role: "assistant",
    source: "proactive",
    content: "历史提醒不会在启动时补弹。",
    delivery: "idle",
    notification_eligible: true,
    completed: true,
    chunks: [],
  });
  const server = http.createServer(async (request, response) => {
    const buffers = [];
    for await (const chunk of request) buffers.push(chunk);
    const body = JSON.parse(Buffer.concat(buffers).toString() || "{}");
    const json = (answer, metadata = {}) => {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ success: true, answer, metadata }));
    };
    const job = request.url;
    if (job === "/app_config")
      return json({
        workspace_dir: "/mock/user-workspace",
        daily_dir: "daily",
        digest_dir: "digest",
        jobs: Object.fromEntries(
          [
            "pa_events",
            "pa_chat",
            "pa_session",
            "pa_new_session",
            "session_read",
            "proactive_inbox_view",
            "task_results",
            ...(speech ? ["tts_demo", "asr_demo"] : []),
            ...(workspace
              ? ["list", "load", "list_tags", "list_tag_files", "search"]
              : []),
          ].map((name) => [
            name,
            {
              backend:
                name === "pa_events" || name === "pa_chat" ? "stream" : "base",
              enabled: true,
              enable_serve: true,
            },
          ]),
        ),
      });
    if (job === "/task_results") {
      const item = taskResults.find((item) => item.path === body.path);
      if (body.mark_read) item.user_read_at = new Date().toISOString();
      return json({ user_read_at: item.user_read_at });
    }
    if (job === "/proactive_inbox_view") {
      if (body.action === "read") {
        const item = taskResults.find((item) => item.id === body.entry_id);
        return json({
          item,
          content: item.preview,
          received_at: null,
          warnings: [],
        });
      }
      const items = taskResults.filter(
        (item) => !body.task_id || item.task_id === body.task_id,
      );
      const offset = body.offset ?? 0;
      const limit = body.limit ?? 5;
      return json({
        items: items.slice(offset, offset + limit),
        total: items.length,
        offset,
        limit,
        task_ids: taskIds,
        pending: 0,
        complete: true,
        warnings: [],
      });
    }
    if (workspace && job === "/list")
      return json("Files", {
        items: Object.keys(documents).filter(
          (path) => !body.path || path.startsWith(body.path + "/"),
        ),
      });
    if (workspace && job === "/load")
      return json(documents[body.path] ?? "", {
        mtime: "2026-11-08T10:30:00",
        path: body.path,
      });
    if (workspace && job === "/list_tags")
      return json({
        total_tags: tagItems.length,
        total_pages: Math.ceil(tagItems.length / body.page_size),
        page: body.page,
        items: tagItems.slice(
          (body.page - 1) * body.page_size,
          body.page * body.page_size,
        ),
      });
    if (workspace && (job === "/list_tag_files" || job === "/search")) {
      const items = Object.keys(documents)
        .filter(
          (path) =>
            (!body.path || path.startsWith(body.path + "/")) &&
            (!body.tags?.length ||
              body.tags.some((tag) =>
                documents[path].toLowerCase().includes(tag.toLowerCase()),
              )),
        )
        .map((path) => ({ path, tags: [], text: "Memory" }));
      if (job === "/search")
        return json("Search results", {
          results: items,
          tag_filter: {
            applied: body.query !== "unavailable",
            tags: body.tags.map((tag) => tag.toLowerCase()),
          },
        });
      return json(
        {
          total_files: items.length,
          total_pages: 1,
          page: 1,
          items,
        },
        {
          tag_filter: {
            applied: true,
            tags: body.tags.map((tag) => tag.toLowerCase()),
          },
        },
      );
    }
    if (speech && (job === "/tts_demo" || job === "/asr_demo")) {
      (job === "/tts_demo" ? ttsRequests : asrRequests).push(body);
      const fail =
        speechError ||
        (job === "/tts_demo" &&
          speechErrorText &&
          body.text.includes(speechErrorText));
      const finish = () => {
        if (response.destroyed) return;
        if (fail) {
          response.writeHead(200, { "Content-Type": "application/json" });
          response.end(
            JSON.stringify({
              success: false,
              answer: "Mock speech provider unavailable",
            }),
          );
        } else
          json(
            job === "/asr_demo"
              ? asrText
              : {
                  audio: toneWave(ttsSeconds).toString("base64"),
                  format: "wav",
                  sample_rate: 16000,
                  is_last: true,
                },
          );
      };
      if (speechDelay) {
        const timer = setTimeout(finish, speechDelay);
        response.on("close", () => clearTimeout(timer));
      } else finish();
      return;
    }
    if (job === "/pa_session") return json({ session_id: sessionId, status });
    if (job === "/session_read") return json(snapshot());
    if (job === "/pa_new_session") {
      sessionId = `session-${Number(sessionId.split("-").at(-1)) + 1}`;
      messages = [];
      activities = {};
      status = "idle";
      emit({ type: "session_changed" });
      emit({ type: "status_changed", status, activities });
      return json(sessionId);
    }
    if (job === "/pa_events") {
      cursors.push(body.cursor);
      response.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      });
      sockets.add(response);
      if (body.cursor) {
        const index = events.findIndex((event) => event.cursor === body.cursor);
        if (index >= 0)
          for (const event of events.slice(index + 1))
            frame(response, { chunk_type: "data", chunk: event });
        else {
          frame(response, {
            chunk_type: "data",
            chunk: { type: "reset", reason: "cursor_expired" },
          });
          frame(response, {
            chunk_type: "data",
            chunk: {
              type: "snapshot",
              session_id: sessionId,
              status,
              activities,
              task_activities: taskActivities,
              notifications,
              snapshot: snapshot(),
              cursor: `test-epoch:${events.length}`,
            },
          });
        }
      } else
        frame(response, {
          chunk_type: "data",
          chunk: {
            type: "snapshot",
            session_id: sessionId,
            status,
            activities,
            task_activities: taskActivities,
            notifications,
            snapshot: snapshot(),
            cursor: `test-epoch:${events.length}`,
          },
        });
      const heartbeat = setInterval(
        () =>
          frame(response, {
            chunk_type: "data",
            chunk: { type: "heartbeat", session_id: sessionId },
          }),
        200,
      );
      response.on("close", () => {
        clearInterval(heartbeat);
        sockets.delete(response);
      });
      return;
    }
    if (job === "/pa_chat") {
      const turnId = `turn-${++turn}`;
      status = "active";
      activities = {
        chat: {
          session_id: sessionId,
          turn_id: turnId,
          source: "chat",
          phase: "answering",
          tool: null,
        },
      };
      emit({ type: "status_changed", status, activities });
      const user = {
        id: `user-${turn}`,
        role: "user",
        source: "chat",
        content: body.query,
        delivery: "idle",
        notification_eligible: true,
        completed: true,
        turn_id: turnId,
        chunks: [],
      };
      upsert(user);
      response.writeHead(200, { "Content-Type": "text/event-stream" });
      const send = (chunk) =>
        frame(response, { session_id: sessionId, turn_id: turnId, ...chunk });
      send({ chunk_type: "reply_start", status: "running", chunk: null });
      send({
        chunk_type: "content",
        block_id: "answer",
        event: "delta",
        chunk: "旧的增量",
      });
      if (body.query.includes("停止")) {
        response.on("close", () => {
          upsert({
            id: `assistant-${turn}`,
            role: "assistant",
            source: "chat",
            content: "旧的增量",
            delivery: "idle",
            notification_eligible: true,
            completed: true,
            turn_id: turnId,
            chunks: [],
          });
          status = "idle";
          activities.chat = { ...activities.chat, phase: "interrupted" };
          emit({ type: "status_changed", status, activities });
        });
        return;
      }
      setTimeout(() => {
        const text = "我记住了。我们可以继续聊聊你的研究。";
        send({
          chunk_type: "content",
          block_id: "answer",
          event: "end",
          chunk: text,
        });
        upsert({
          id: `assistant-${turn}`,
          role: "assistant",
          source: "chat",
          content: text,
          delivery: "idle",
          notification_eligible: true,
          completed: true,
          turn_id: turnId,
          chunks: [],
        });
        send({
          chunk_type: "reply_end",
          status: "completed",
          chunk: { status: "completed", result: text, error: null },
        });
        send({ chunk_type: "done", done: true, chunk: "" });
        response.end();
        status = "idle";
        activities.chat = { ...activities.chat, phase: "completed" };
        emit({ type: "status_changed", status, activities });
      }, replyDelay);
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    notificationWindow(value) {
      notifications = value;
      emit({ type: "status_changed", status, activities });
    },
    url: `http://127.0.0.1:${server.address().port}`,
    push,
    upsert,
    setTaskIds(value) {
      taskIds = value;
    },
    setReplyDelay(value) {
      replyDelay = value;
    },
    cursors,
    ttsRequests,
    asrRequests,
    setSpeechError(value) {
      speechError = value;
    },
    setSpeechErrorText(value) {
      speechErrorText = value;
    },
    setAsrText(value) {
      asrText = value;
    },
    setTags(value) {
      tagItems = value;
    },
    setDocument(path, content) {
      documents[path] = content;
    },
    setSpeechDelay(value) {
      speechDelay = value;
    },
    activity(phase, source = "chat", turnId = "remote-turn") {
      activities = {
        ...activities,
        [source]: {
          session_id: sessionId,
          turn_id: turnId,
          source,
          phase,
          tool: null,
        },
      };
      if (source === "chat")
        status = ["completed", "failed", "interrupted"].includes(phase)
          ? "idle"
          : "active";
      emit({ type: "status_changed", status, activities });
    },
    taskActivity(phase, taskId = "research", runId = taskId) {
      if (["completed", "failed", "interrupted"].includes(phase))
        delete taskActivities[runId];
      else
        taskActivities[runId] = {
          session_id: `task-session:${runId}`,
          turn_id: `task-turn:${runId}`,
          source: "task",
          phase,
          run_id: runId,
          task_id: taskId,
          tool: null,
        };
      emit({ type: "status_changed", status, activities });
    },
    disconnect() {
      for (const socket of sockets) socket.destroy();
      sockets.clear();
    },
    async close() {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}
