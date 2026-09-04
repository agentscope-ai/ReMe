import { h } from "vue";
import DefaultTheme from "vitepress/theme";
import CopyMarkdownButton from "./CopyMarkdownButton.vue";
import HomePage from "./HomePage.vue";
import SourceLink from "./SourceLink.vue";
import TrafficPage from "./TrafficPage.vue";
import "./custom.css";

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component("HomePage", HomePage);
    app.component("TrafficPage", TrafficPage);
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      "doc-before": () => h(CopyMarkdownButton),
      "doc-after": () => h(SourceLink),
    });
  },
};
