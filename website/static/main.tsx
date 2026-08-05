import React from "react";
import ReactDOM from "react-dom/client";

import "../app/globals.css";
import { ReMeWorkspace } from "../app/workspace";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ReMeWorkspace />
  </React.StrictMode>,
);
