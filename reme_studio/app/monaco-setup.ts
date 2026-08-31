/// <reference types="vite/client" />

/** QwenPaw-compatible offline Monaco setup: bundle workers and never fetch editor code from a CDN. */
import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import cssWorker from "monaco-editor/esm/vs/language/css/css.worker?worker";
import htmlWorker from "monaco-editor/esm/vs/language/html/html.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";

if (typeof self !== "undefined") {
  self.MonacoEnvironment = {
    ...self.MonacoEnvironment,
    getWorker(_workerId: string, label: string) {
      if (label === "json") return new jsonWorker();
      if (["css", "scss", "less"].includes(label)) return new cssWorker();
      if (["html", "handlebars", "razor"].includes(label))
        return new htmlWorker();
      if (["typescript", "javascript"].includes(label)) return new tsWorker();
      return new editorWorker();
    },
  };
  loader.config({ monaco });
}
