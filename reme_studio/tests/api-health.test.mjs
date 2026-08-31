import assert from "node:assert/strict";
import test from "node:test";
import { healthFromResponse } from "../app/health-status.ts";

test("health metadata is returned from the existing health_check response", () => {
  const health = {
    version: "0.4.1.8",
    healthy: true,
    components: { file_store: { default: { is_started: true } } },
  };

  const result = healthFromResponse({ metadata: { health } });

  assert.deepEqual(result, health);
});

test("invalid health metadata keeps the settings fallback available", () => {
  assert.equal(healthFromResponse({ metadata: {} }), undefined);
  assert.equal(healthFromResponse({ metadata: { health: null } }), undefined);
  assert.equal(
    healthFromResponse({ metadata: { health: "unknown" } }),
    undefined,
  );
});
