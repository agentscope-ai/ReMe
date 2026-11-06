import http from "node:http";

export async function mockReMe() {
  const sockets = new Set();
  const events = [];
  const cursors = [];
  let sessionId = "session-1";
  let messages = [];
  let status = "idle";
  let turn = 0;
  const frame = (response, chunk) =>
    response.write(`data: ${JSON.stringify(chunk)}\n\n`);
  const emit = (event) => {
    const value = {
      ...event,
      session_id: sessionId,
      cursor: `test-epoch:${events.length + 1}`,
    };
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
    completed: true,
    chunks: [],
  });
  const server = http.createServer(async (request, response) => {
    const buffers = [];
    for await (const chunk of request) buffers.push(chunk);
    const body = JSON.parse(Buffer.concat(buffers).toString() || "{}");
    const json = (answer) => {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ success: true, answer, metadata: {} }));
    };
    const job = request.url;
    if (job === "/app_config")
      return json({
        workspace_dir: "/mock/user-workspace",
        jobs: Object.fromEntries(
          [
            "pa_events",
            "pa_chat",
            "pa_session",
            "pa_new_session",
            "session_read",
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
    if (job === "/pa_session") return json({ session_id: sessionId, status });
    if (job === "/session_read") return json(snapshot());
    if (job === "/pa_new_session") {
      sessionId = `session-${Number(sessionId.split("-").at(-1)) + 1}`;
      messages = [];
      status = "idle";
      emit({ type: "session_changed" });
      emit({ type: "status_changed", status });
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
      emit({ type: "status_changed", status });
      const user = {
        id: `user-${turn}`,
        role: "user",
        source: "chat",
        content: body.query,
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
            completed: true,
            turn_id: turnId,
            chunks: [],
          });
          status = "idle";
          emit({ type: "status_changed", status });
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
        emit({ type: "status_changed", status });
      }, 300);
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    push,
    cursors,
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
