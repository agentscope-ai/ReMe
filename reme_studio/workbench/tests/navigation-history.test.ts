import assert from "node:assert/strict";
import test from "node:test";
import {
  sameLocation,
  travel,
  visit,
  type NavigationHistory,
  type NavigationLocation,
} from "../src/navigation-history.ts";

const chat: NavigationLocation = { page: "remy", values: {} };
const reminders: NavigationLocation = {
  page: "remy",
  values: { section: "reminders" },
};
const document: NavigationLocation = {
  page: "knowledge",
  values: { selected: "MEMORY.md" },
};
const initial = (): NavigationHistory => ({ entries: [chat], index: 0 });

test("back and forward restore page and subpage destinations without adding visits", () => {
  const history = visit(visit(initial(), reminders), document);
  const back = travel(history, -1);
  assert.equal(back.entries[back.index], reminders);
  assert.equal(travel(back, -1).entries[0], chat);
  assert.deepEqual(travel(back, 1), history);
  assert.equal(back.entries, history.entries);
  assert.equal(visit(back, { ...reminders }), back);
});

test("a new destination after going back drops the forward branch", () => {
  const back = travel(visit(visit(initial(), reminders), document), -1);
  const next = visit(back, {
    page: "settings",
    values: { section: "appearance" },
  });
  assert.deepEqual(
    next.entries.map((entry) => entry.page),
    ["remy", "remy", "settings"],
  );
  assert.equal(travel(next, 1), next);
  assert.equal(travel(initial(), -1).index, 0);
});

test("destination identity ignores field insertion order and captures item and filter changes", () => {
  assert.ok(
    sameLocation(
      { page: "knowledge", values: { source: "tags", tags: ["work"] } },
      { page: "knowledge", values: { tags: ["work"], source: "tags" } },
    ),
  );
  assert.equal(sameLocation(chat, reminders), false);
  assert.equal(
    sameLocation(document, { ...document, values: { selected: "other.md" } }),
    false,
  );
  assert.equal(
    sameLocation(
      { ...chat, values: { filter: "all" } },
      { ...chat, values: { other: "all" } },
    ),
    false,
  );
});

test("the renderer history remains bounded while retaining the newest destinations", () => {
  let history = initial();
  for (let index = 0; index < 150; index++)
    history = visit(history, {
      page: "knowledge",
      values: { selected: `${index}.md` },
    });
  assert.equal(history.entries.length, 100);
  assert.equal(history.index, 99);
  assert.equal(history.entries[0].values.selected, "50.md");
  assert.equal(history.entries[99].values.selected, "149.md");
});
