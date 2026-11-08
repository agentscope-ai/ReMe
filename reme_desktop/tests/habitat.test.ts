import assert from "node:assert/strict";
import test from "node:test";
import { CompanionHabitat, type HabitatInput } from "../src/main/habitat.ts";

const primary = {
  id: 42,
  bounds: { x: 0, y: 0, width: 1728, height: 1117 },
  workArea: { x: 0, y: 25, width: 1728, height: 1060 },
};
const left = {
  id: 7,
  bounds: { x: -1920, y: -300, width: 1920, height: 1080 },
  workArea: { x: -1920, y: -275, width: 1920, height: 1030 },
};
const home = { x: 1600, y: 1060 };
const input = (overrides: Partial<HabitatInput> = {}): HabitatInput => ({
  displays: [primary, left],
  slot: { x: 1300, y: 180, width: 260, height: 680 },
  window: { x: 100, y: 100, width: 1500, height: 840 },
  scale: 1,
  reducedMotion: false,
  ...overrides,
});
const arrived = (habitat: CompanionHabitat, value = input()) => {
  habitat.update(value, 0);
  habitat.update(value, 300);
  return habitat.update(value, 2000);
};

test("a stable free slot starts a same-display walk and hands off once", () => {
  const habitat = new CompanionHabitat(home);
  assert.equal(habitat.update(input(), 0).motion, undefined);
  assert.equal(habitat.update(input(), 299).host, "desktop");
  const walking = habitat.update(input(), 300);
  assert.equal(walking.host, "desktop");
  assert.equal(walking.motion?.direction, -1);
  const middle = habitat.update(input(), 500);
  assert.ok(middle.anchor.x < home.x && middle.anchor.x > 1430);
  const end = habitat.update(input(), 2000);
  assert.equal(end.host, "app");
  assert.equal(end.motion, undefined);
  assert.equal(habitat.pending, false);
});

test("cross-display travel fades out then walks locally, including negative origins and gaps", () => {
  const habitat = new CompanionHabitat(home);
  const value = input({
    slot: { x: -700, y: -100, width: 260, height: 650 },
    window: { x: -1800, y: -200, width: 1400, height: 900 },
  });
  habitat.update(value, 0);
  habitat.update(value, 300);
  const fading = habitat.update({ ...value, displays: [left, primary] }, 375);
  assert.equal(fading.opacity, 0.5);
  assert.deepEqual(fading.anchor, home);
  const walking = habitat.update(value, 500);
  assert.ok(walking.anchor.x < 0);
  assert.ok(walking.motion);
  assert.equal(habitat.update(value, 2000).host, "app");
});

test("an app on a vertically separated monitor never walks through empty desktop", () => {
  const upper = {
    id: 99,
    bounds: { x: 500, y: -1600, width: 1920, height: 1080 },
    workArea: { x: 500, y: -1575, width: 1920, height: 1030 },
  };
  const value = input({
    displays: [primary, upper],
    slot: { x: 1500, y: -1450, width: 300, height: 700 },
    window: { x: 600, y: -1500, width: 1300, height: 900 },
  });
  const habitat = new CompanionHabitat(home);
  habitat.update(value, 0);
  habitat.update(value, 300);
  for (let now = 300; now <= 2000; now += 20) {
    const frame = habitat.update(value, now);
    assert.ok(frame.anchor.y >= 0 || frame.anchor.y <= -520);
  }
});

test("resident follows its app and tolerates a small space reduction without restarting travel", () => {
  const habitat = new CompanionHabitat(home);
  arrived(habitat);
  const shifted = habitat.update(
    input({
      slot: { x: 1200, y: 150, width: 215, height: 680 },
    }),
    2200,
  );
  assert.equal(shifted.host, "app");
  assert.equal(shifted.motion, undefined);
  assert.equal(shifted.anchor.x, 1308);
  const leaving = habitat.update(
    input({ slot: { x: 1200, y: 150, width: 200, height: 680 } }),
    2300,
  );
  assert.equal(leaving.host, "desktop");
  assert.ok(leaving.motion);
  assert.equal(habitat.update(input({ slot: null }), 5000).motion, undefined);
});

test("closing an app on another screen returns to that screen instead of the old home", () => {
  const habitat = new CompanionHabitat(home);
  arrived(
    habitat,
    input({
      slot: { x: -700, y: -100, width: 260, height: 650 },
      window: { x: -1800, y: -200, width: 1400, height: 900 },
    }),
  );
  const closed = input({ slot: null, window: null });
  habitat.update(closed, 2100);
  const end = habitat.update(closed, 4000);
  assert.equal(end.host, "desktop");
  assert.ok(end.anchor.x < 0);
});

test("dragging pins the desktop position until the next window opening", () => {
  const habitat = new CompanionHabitat(home);
  arrived(habitat);
  habitat.place({ x: 1100, y: 900 });
  assert.equal(habitat.update(input(), 3000).host, "desktop");
  assert.equal(habitat.pending, false);
  habitat.unpin();
  habitat.update(input(), 4000);
  assert.ok(habitat.update(input(), 4300).motion);
});

test("monitor removal cancels travel and clamps the role into a remaining work area", () => {
  const habitat = new CompanionHabitat({ x: -500, y: 700 });
  const frame = habitat.update(input({ displays: [primary], slot: null }), 0);
  assert.equal(frame.host, "desktop");
  assert.ok(frame.anchor.x >= 106);
  assert.ok(
    frame.anchor.y <= primary.workArea.y + primary.workArea.height - 16,
  );
});

test("a tall bubble releases a slot that cannot fit both the pet and its message", () => {
  const habitat = new CompanionHabitat(home);
  const small = input({ slot: { x: 1300, y: 400, width: 260, height: 300 } });
  arrived(habitat, small);
  assert.equal(habitat.update(small, 2100).host, "app");
  assert.equal(
    habitat.update({ ...small, surfaceHeight: 180 }, 2200).host,
    "desktop",
  );
});

test("unplugging a destination monitor cancels a trip or releases its resident", () => {
  const secondary = input({
    slot: { x: -700, y: -100, width: 260, height: 650 },
    window: { x: -1800, y: -200, width: 1400, height: 900 },
  });
  for (const resident of [false, true]) {
    const habitat = new CompanionHabitat(home);
    if (resident) arrived(habitat, secondary);
    else {
      habitat.update(secondary, 0);
      habitat.update(secondary, 300);
    }
    const frame = habitat.update(
      { ...secondary, displays: [primary] },
      resident ? 2100 : 350,
    );
    assert.equal(frame.host, "desktop");
    assert.equal(frame.motion, undefined);
    assert.ok(frame.anchor.x >= 106 && frame.anchor.x <= 1622);
    assert.ok(frame.anchor.y >= 221 && frame.anchor.y <= 1069);
    assert.equal(habitat.pending, false);
  }
});

test("reduced motion changes hosts directly and an unstable or clipped slot never attracts the pet", () => {
  const habitat = new CompanionHabitat(home);
  const noMotion = input({ reducedMotion: true });
  habitat.update(noMotion, 0);
  assert.equal(habitat.update(noMotion, 300).host, "app");
  habitat.place(home, false);
  for (let now = 0; now < 1000; now += 100)
    assert.equal(
      habitat.update(
        input({
          slot: { x: 1300 + now / 100, y: 180, width: 260, height: 680 },
        }),
        now,
      ).motion,
      undefined,
    );
  assert.equal(
    habitat.update(
      input({ slot: { x: 1700, y: 180, width: 260, height: 680 } }),
      2000,
    ).host,
    "desktop",
  );
});
