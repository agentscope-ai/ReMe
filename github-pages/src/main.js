import DOMPurify from "dompurify";
import { marked } from "marked";
import "./styles.css";

const baseUrl = import.meta.env.BASE_URL;
const repositoryUrl = "https://github.com/agentscope-ai/ReMe";

const copy = {
  zh: {
    docs: "文档",
    home: "首页",
    search: "搜索文档…",
    noResults: "没有找到匹配的文档",
    menu: "打开导航",
    toc: "本页目录",
    edit: "在 GitHub 查看源文件",
    quickStart: "快速开始",
    groups: {
      overview: "项目介绍",
      start: "开始使用",
      fundamentals: "核心原理",
      automation: "自动化能力",
      concepts: "架构与场景",
      integration: "Agent 集成",
      workspace: "工作区",
      cookbooks: "研究工作流",
      benchmarks: "评测",
      development: "开发者规范",
    },
  },
  en: {
    docs: "Documentation",
    home: "Home",
    search: "Search documentation…",
    noResults: "No matching documents",
    menu: "Open navigation",
    toc: "On this page",
    edit: "View source on GitHub",
    quickStart: "Quick start",
    groups: {
      overview: "Introduction",
      start: "Get started",
      fundamentals: "Fundamentals",
      automation: "Automation",
      concepts: "Architecture & scenarios",
      integration: "Agent integration",
      workspace: "Workspace",
      cookbooks: "Research workflows",
      benchmarks: "Benchmarks",
      development: "Development",
    },
  },
};

const state = {
  language: localStorage.getItem("reme-docs-language") || "zh",
  documents: [],
  activeDocument: null,
  query: "",
};

const homeCopy = {
  zh: {
    eyebrow: "LOCAL-FIRST · FILE-NATIVE",
    title: "让 Agent 真正记住，\n也让记忆始终属于你。",
    description: "ReMe 将对话和资料沉淀为可读、可编辑、可检索、相互链接的 Markdown，并提供从工作区管理到主动研究的一整套工具。",
    start: "快速开始",
    project: "了解 ReMe",
    explore: "按目标探索",
    exploreDescription: "选择你现在想完成的事情。每个入口都直接连接到对应的完整文档。",
    cards: [
      { id: "studio-zh", icon: "◫", label: "管理记忆", title: "ReMe 工作台", description: "在本地 Web 工作区中浏览、编辑、搜索记忆，并探索 wikilink 图谱。", tone: "mint" },
      { id: "daily-paper-zh", icon: "◌", label: "发现与分析", title: "每日论文", description: "从论文榜单筛选值得阅读的工作，解析 PDF，并生成笔记与五分钟简报。", tone: "blue" },
      { id: "auto-fin-zh", icon: "↗", label: "主题研究", title: "财经研究", description: "连接最新财联社新闻和本地历史记忆，生成可追溯的研究报告。", tone: "amber" },
    ],
    benchmark: "验证记忆能力",
    benchmarkDescription: "从检索规模、跨会话问答、个人智能体到工具经验，查看 ReMe 的四套评测与复现实验。",
    benchmarkAction: "从 BEAM 开始",
  },
  en: {
    eyebrow: "LOCAL-FIRST · FILE-NATIVE",
    title: "Memory that works for agents.\nFiles that remain yours.",
    description: "ReMe turns conversations and resources into readable, editable, searchable, interconnected Markdown—with tools spanning workspace management and proactive research.",
    start: "Quick start",
    project: "Meet ReMe",
    explore: "Explore by goal",
    exploreDescription: "Start with what you want to accomplish. Every entry opens the complete guide.",
    cards: [
      { id: "studio-en", icon: "◫", label: "Manage memory", title: "ReMe Studio", description: "Browse, edit, and search memory in a local web workspace, then explore its wikilink graph.", tone: "mint" },
      { id: "daily-paper-en", icon: "◌", label: "Discover & analyze", title: "Daily Paper", description: "Select useful papers from rankings, analyze PDFs, and create notes plus a five-minute brief.", tone: "blue" },
      { id: "auto-fin-en", icon: "↗", label: "Research topics", title: "Auto Fin", description: "Connect recent CLS news with historical local memory to produce traceable research reports.", tone: "amber" },
    ],
    benchmark: "Validate memory systems",
    benchmarkDescription: "Explore four reproducible evaluations covering retrieval scale, cross-session QA, personal agents, and tool-use experience.",
    benchmarkAction: "Start with BEAM",
  },
};

const app = document.querySelector("#app");

app.innerHTML = `
  <header class="topbar">
    <a class="brand" href="${baseUrl}" aria-label="ReMe documentation home">
      <span class="brand-mark">R</span>
      <span>ReMe</span>
      <span class="brand-divider"></span>
      <span class="brand-section" data-copy="docs"></span>
    </a>
    <nav class="top-actions" aria-label="Global navigation">
      <div class="language-switch" role="group" aria-label="Language">
        <button type="button" data-language="zh">中</button>
        <button type="button" data-language="en">EN</button>
      </div>
      <a class="quick-start-link" href="?doc=zh-quick_start" data-doc="zh-quick_start" data-copy="quickStart"></a>
      <a class="github-link" href="${repositoryUrl}" target="_blank" rel="noreferrer">GitHub ↗</a>
      <button class="menu-button" type="button" aria-expanded="false" data-action="menu"></button>
    </nav>
  </header>
  <div class="docs-shell">
    <aside class="sidebar" aria-label="Documentation navigation">
      <label class="search-box">
        <span aria-hidden="true">⌕</span>
        <input type="search" autocomplete="off" />
        <kbd>⌘K</kbd>
      </label>
      <nav class="document-nav"></nav>
      <div class="sidebar-footer">
        <span class="status-dot"></span>
        Local-first · File-native
      </div>
    </aside>
    <main class="article-wrap">
      <article class="article"><div class="loading-line"></div></article>
    </main>
    <aside class="toc-panel"><nav class="toc"></nav></aside>
  </div>
  <button class="sidebar-backdrop" type="button" aria-label="Close navigation"></button>
`;

const sidebar = app.querySelector(".sidebar");
const docsShell = app.querySelector(".docs-shell");
const documentNav = app.querySelector(".document-nav");
const article = app.querySelector(".article");
const toc = app.querySelector(".toc");
const searchInput = app.querySelector("input[type='search']");
const menuButton = app.querySelector(".menu-button");
const backdrop = app.querySelector(".sidebar-backdrop");

function slugify(value) {
  return value
    .toLowerCase()
    .trim()
    .replace(/<[^>]+>/g, "")
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-|-$/g, "");
}

function resolveDocumentPath(currentPath, target) {
  const cleanTarget = target.split("#")[0].split("?")[0];
  const currentParts = currentPath.split("/");
  currentParts.pop();
  for (const part of cleanTarget.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") currentParts.pop();
    else currentParts.push(part);
  }
  return currentParts.join("/");
}

function configureMarkdown(document) {
  const renderer = new marked.Renderer();
  const headingIds = new Map();

  renderer.heading = ({ tokens, depth }) => {
    const text = tokens.map((token) => token.text || token.raw || "").join("");
    const baseSlug = slugify(text) || "section";
    const count = headingIds.get(baseSlug) || 0;
    headingIds.set(baseSlug, count + 1);
    const id = count ? `${baseSlug}-${count + 1}` : baseSlug;
    return `<h${depth} id="${id}">${text}</h${depth}>`;
  };

  renderer.image = ({ href, title, text }) => {
    const url = /^(https?:|data:)/.test(href)
      ? href
      : `${baseUrl}content/${resolveDocumentPath(document.path, href)}`;
    const titleAttribute = title ? ` title="${title}"` : "";
    return `<img src="${url}" alt="${text}" loading="lazy"${titleAttribute}>`;
  };

  renderer.link = ({ href, title, tokens }) => {
    const label = tokens.map((token) => token.text || token.raw || "").join("");
    const titleAttribute = title ? ` title="${title}"` : "";
    if (href.startsWith("#")) return `<a href="${href}"${titleAttribute}>${label}</a>`;
    if (!/^(https?:|mailto:)/.test(href)) {
      const resolved = resolveDocumentPath(document.path, href);
      const localDocument = state.documents.find((item) => item.path === resolved);
      if (localDocument) return `<a href="?doc=${localDocument.id}" data-doc="${localDocument.id}">${label}</a>`;
      return `<a href="${repositoryUrl}/blob/main/${resolved}" target="_blank" rel="noreferrer">${label}</a>`;
    }
    return `<a href="${href}" target="_blank" rel="noreferrer"${titleAttribute}>${label}</a>`;
  };

  marked.use({ renderer, gfm: true, breaks: false });
}

function availableDocuments() {
  return state.documents.filter(
    (document) => document.language === state.language || document.language === "shared",
  );
}

function documentTitle(document) {
  return document.titles?.[state.language] || document.title;
}

function renderChrome() {
  const labels = copy[state.language];
  app.querySelector("[data-copy='docs']").textContent = labels.docs;
  const quickStartLink = app.querySelector("[data-copy='quickStart']");
  quickStartLink.textContent = `${labels.quickStart} →`;
  quickStartLink.href = `?doc=${state.language}-quick_start`;
  quickStartLink.dataset.doc = `${state.language}-quick_start`;
  searchInput.placeholder = labels.search;
  menuButton.textContent = labels.menu;
  document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
  app.querySelectorAll("[data-language]").forEach((button) => {
    button.classList.toggle("active", button.dataset.language === state.language);
  });
}

function renderNavigation() {
  const labels = copy[state.language];
  const query = state.query.trim().toLocaleLowerCase();
  const filtered = availableDocuments().filter((document) =>
    `${documentTitle(document)} ${document.title || ""} ${document.description}`.toLocaleLowerCase().includes(query),
  );
  const groups = [...new Set(filtered.map((document) => document.group))];

  if (!filtered.length) {
    documentNav.innerHTML = `
      <a href="${baseUrl}" data-home class="home-link ${state.activeDocument ? "" : "active"}">${labels.home}</a>
      <p class="empty-state">${labels.noResults}</p>
    `;
    return;
  }

  documentNav.innerHTML = `
    <a href="${baseUrl}" data-home class="home-link ${state.activeDocument ? "" : "active"}">${labels.home}</a>
  ` + groups
    .map(
      (group) => `
        <section class="nav-group">
          <h2>${labels.groups[group]}</h2>
          ${filtered
            .filter((document) => document.group === group)
            .map(
              (document) => `
                <a href="?doc=${document.id}" data-doc="${document.id}" class="${state.activeDocument?.id === document.id ? "active" : ""}">
                  <span>${documentTitle(document)}</span>
                </a>`,
            )
            .join("")}
        </section>`,
    )
    .join("");
}

function renderHome(pushHistory = true) {
  const labels = homeCopy[state.language];
  state.activeDocument = null;
  docsShell.classList.add("home-view");
  article.dataset.group = "home";
  renderNavigation();
  article.innerHTML = `
    <section class="home-hero">
      <p class="home-eyebrow">${labels.eyebrow}</p>
      <h1>${labels.title.replace("\n", "<br>")}</h1>
      <p class="home-lead">${labels.description}</p>
      <div class="home-actions">
        <a href="${baseUrl}?doc=${state.language}-quick_start" class="primary-action">${labels.start} →</a>
        <a href="${baseUrl}?doc=readme-${state.language}" class="secondary-action">${labels.project}</a>
      </div>
    </section>
    <section class="home-explore">
      <p class="section-kicker">01 / PRODUCT & WORKFLOWS</p>
      <h2>${labels.explore}</h2>
      <p class="section-lead">${labels.exploreDescription}</p>
      <div class="feature-grid">
        ${labels.cards.map((card) => `
          <a href="${baseUrl}?doc=${card.id}" class="feature-card ${card.tone}">
            <span class="feature-icon">${card.icon}</span>
            <span class="feature-label">${card.label}</span>
            <strong>${card.title}</strong>
            <span class="feature-description">${card.description}</span>
            <span class="feature-arrow">→</span>
          </a>`).join("")}
      </div>
    </section>
    <section class="benchmark-callout">
      <div>
        <p class="section-kicker">02 / BENCHMARKS</p>
        <h2>${labels.benchmark}</h2>
        <p>${labels.benchmarkDescription}</p>
      </div>
      <a href="${baseUrl}?doc=beam-${state.language}">${labels.benchmarkAction} →</a>
    </section>
  `;
  toc.innerHTML = "";
  closeMenu();
  if (pushHistory) history.pushState({ home: true }, "", baseUrl);
  window.scrollTo({ top: 0, behavior: "instant" });
}

function renderToc() {
  const headings = [...article.querySelectorAll("h2, h3")];
  if (!headings.length) {
    toc.innerHTML = "";
    return;
  }
  toc.innerHTML = `
    <h2>${copy[state.language].toc}</h2>
    ${headings
      .map(
        (heading) => `<a class="toc-${heading.tagName.toLowerCase()}" href="#${heading.id}">${heading.childNodes[0]?.textContent || heading.textContent}</a>`,
      )
      .join("")}
  `;
}

function rewriteRenderedUrls(document) {
  article.querySelectorAll("img[src]").forEach((image) => {
    const source = image.getAttribute("src");
    if (source && !/^(https?:|data:|\/)/.test(source)) {
      image.src = `${baseUrl}content/${resolveDocumentPath(document.path, source)}`;
    }
  });

  article.querySelectorAll("a[href]").forEach((link) => {
    const href = link.getAttribute("href");
    if (!href || /^(https?:|mailto:|#|\/)/.test(href) || link.dataset.doc) return;
    const resolved = resolveDocumentPath(document.path, href);
    const localDocument = state.documents.find((item) => item.path === resolved);
    if (localDocument) {
      link.href = `?doc=${localDocument.id}`;
      link.dataset.doc = localDocument.id;
      link.removeAttribute("target");
      return;
    }
    link.href = `${repositoryUrl}/blob/main/${resolved}`;
    link.target = "_blank";
    link.rel = "noreferrer";
  });
}

async function openDocument(id, pushHistory = true) {
  const fallbackId = state.language === "zh" ? "readme-zh" : "readme-en";
  const document = state.documents.find((item) => item.id === id) || state.documents.find((item) => item.id === fallbackId);
  if (document.language !== "shared" && document.language !== state.language) {
    state.language = document.language;
    localStorage.setItem("reme-docs-language", state.language);
    renderChrome();
  }
  state.activeDocument = document;
  docsShell.classList.remove("home-view");
  article.dataset.group = document.group;
  renderNavigation();
  article.innerHTML = `<div class="loading-line"></div>`;

  const response = await fetch(`${baseUrl}content/${document.path}`);
  if (!response.ok) throw new Error(`Unable to load ${document.path}`);
  configureMarkdown(document);
  const markdown = await response.text();
  const body = DOMPurify.sanitize(await marked.parse(markdown), {
    ADD_ATTR: ["target"],
  });
  article.innerHTML = `
    <div class="article-meta">
      <span>${copy[state.language].groups[document.group]}</span>
      <span>·</span>
      <span>${document.sourcePath}</span>
    </div>
    <div class="markdown-body">${body}</div>
    <footer class="article-footer">
      <a href="${repositoryUrl}/blob/main/${document.sourcePath}" target="_blank" rel="noreferrer">${copy[state.language].edit} ↗</a>
    </footer>
  `;
  rewriteRenderedUrls(document);
  renderToc();
  closeMenu();
  if (pushHistory) history.pushState({ doc: document.id }, "", `?doc=${document.id}`);
  window.scrollTo({ top: 0, behavior: "instant" });
}

function closeMenu() {
  sidebar.classList.remove("open");
  backdrop.classList.remove("visible");
  menuButton.setAttribute("aria-expanded", "false");
}

function toggleMenu() {
  const open = !sidebar.classList.contains("open");
  sidebar.classList.toggle("open", open);
  backdrop.classList.toggle("visible", open);
  menuButton.setAttribute("aria-expanded", String(open));
}

app.addEventListener("click", (event) => {
  const homeLink = event.target.closest("[data-home]");
  if (homeLink) {
    event.preventDefault();
    renderHome();
    return;
  }
  const documentLink = event.target.closest("[data-doc]");
  if (documentLink) {
    event.preventDefault();
    openDocument(documentLink.dataset.doc);
  }
  if (event.target.closest("[data-action='menu']")) toggleMenu();
  const languageButton = event.target.closest("[data-language]");
  if (languageButton && languageButton.dataset.language !== state.language) {
    state.language = languageButton.dataset.language;
    localStorage.setItem("reme-docs-language", state.language);
    state.query = "";
    searchInput.value = "";
    renderChrome();
    renderHome();
  }
});

searchInput.addEventListener("input", () => {
  state.query = searchInput.value;
  renderNavigation();
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    searchInput.focus();
  }
  if (event.key === "Escape") closeMenu();
});

backdrop.addEventListener("click", closeMenu);
window.addEventListener("popstate", (event) => {
  const id = event.state?.doc || new URLSearchParams(location.search).get("doc");
  if (id) openDocument(id, false);
  else renderHome(false);
});

const manifest = await fetch(`${baseUrl}content/manifest.json`).then((response) => response.json());
state.documents = manifest.documents;
renderChrome();
const initialDocument = new URLSearchParams(location.search).get("doc");
if (initialDocument) await openDocument(initialDocument, false);
else renderHome(false);
