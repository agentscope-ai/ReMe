import assert from "node:assert/strict";
import test from "node:test";
import {
  displayReMeApiEndpoint,
  normalizeReMeApiUrl,
} from "../app/api-endpoint.ts";

test("same-origin API requests retain a visible endpoint", () => {
  const apiUrl = normalizeReMeApiUrl("/");

  assert.equal(apiUrl, "");
  assert.equal(
    displayReMeApiEndpoint(apiUrl, "http://127.0.0.1:2333"),
    "http://127.0.0.1:2333",
  );
  assert.equal(displayReMeApiEndpoint(apiUrl), "/");
});

test("configured API endpoint is preferred over the browser origin", () => {
  const apiUrl = normalizeReMeApiUrl("http://localhost:8181/");

  assert.equal(apiUrl, "http://localhost:8181");
  assert.equal(
    displayReMeApiEndpoint(apiUrl, "http://localhost:3000"),
    "http://localhost:8181",
  );
});
