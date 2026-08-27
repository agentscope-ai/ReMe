import assert from "node:assert/strict";
import test from "node:test";
import {
  healthComponentEntries,
  isComponentHealthy,
} from "../app/health-status.ts";

test("component health honors explicit failures and started fallbacks", () => {
  assert.equal(isComponentHealthy({ is_healthy: true }), true);
  assert.equal(
    isComponentHealthy({ is_started: true, is_healthy: null }),
    true,
  );
  assert.equal(
    isComponentHealthy({ is_started: true, is_healthy: false }),
    false,
  );
  assert.equal(isComponentHealthy({ is_started: false }), false);
});

test("health components are flattened with matching memory usage", () => {
  const components = healthComponentEntries(
    {
      version: "0.4.1.8",
      healthy: true,
      components: {
        embedding_store: {
          default: { is_started: true, dimensions: 1024 },
        },
        file_graph: {
          memory: { is_healthy: false, n_nodes: 12 },
        },
      },
    },
    {
      embedding_store: { default: { human: "12 MiB" } },
    },
  );

  assert.deepEqual(components, [
    {
      type: "embedding_store",
      name: "default",
      component: { is_started: true, dimensions: 1024 },
      memory: "12 MiB",
    },
    {
      type: "file_graph",
      name: "memory",
      component: { is_healthy: false, n_nodes: 12 },
      memory: undefined,
    },
  ]);
  assert.deepEqual(healthComponentEntries(), []);
});
