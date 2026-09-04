<script setup lang="ts">
import { computed, onMounted, reactive } from "vue";
import { withBase } from "vitepress";

const props = defineProps<{ lang: "zh" | "en" }>();

const repository = "https://github.com/agentscope-ai/ReMe";
const trafficShareUrl = "https://cloud.umami.is/analytics/us/share/S1OZK1PSDLEpyiU5?date=30day&page=1";
const stats = reactive({ stars: "3.4K+", forks: "293" });

const translations = {
  zh: {
    eyebrow: "LOCAL-FIRST · FILE-NATIVE",
    title: "让 Agent 真正记住，\n让记忆始终属于你。",
    lead: "ReMe 将对话、资料与经验沉淀为可读、可编辑、可检索、相互链接的本地文件，让不同 Agent 共享同一套长期记忆。",
    quickStart: "快速开始",
    learnMore: "了解 ReMe",
    stars: "GitHub Stars",
    forks: "Forks",
    mapLabel: "SHARED MEMORY WORKSPACE",
    mapTitle: "一层记忆，连接所有 Agent",
    capabilities: [
      { icon: "▱", title: "文件即记忆", detail: "用户拥有的文件是持久事实源", href: "/zh/memory_as_file", tone: "mint" },
      { icon: "✦", title: "自动记忆", detail: "从 daily 到长期 digest 持续演化", href: "/zh/auto_memory", tone: "cyan" },
      { icon: "⌕", title: "检索与图谱", detail: "关键词、向量与 wikilink 联合召回", href: "/zh/memory_search", tone: "blue" },
      { icon: "⌁", title: "Agent 集成", detail: "通过 CLI、HTTP、MCP 与宿主适配器接入", href: "/zh/integrations", tone: "amber" },
    ],
    benchmarkLabel: "02 / BENCHMARKS",
    benchmarkTitle: "用真实评测，\n验证长期记忆",
    benchmarkLead: "从跨会话检索到百万级上下文，ReMe 用可复现的公开基准验证长期记忆。",
    benchmarkAction: "查看全部评测",
    benchmarkNote: "仓库已发布参考结果 · Agentic score",
    benchmarks: [
      { name: "LongMemEval", setting: "500 题 · cleaned-s", score: 89.4 },
      { name: "BEAM 100K", setting: "20 cases · 400 题", score: 66.1 },
      { name: "BEAM 1M", setting: "35 cases · 700 题", score: 65.0 },
    ],
    piLabel: "π-Bench 主动性",
    piDetail: "5 类用户画像的平均 PROC 得分",
    piDelta: "较 NanoBot +2.4%",
    sectionLabel: "03 / PRODUCTS & PLUGINS",
    sectionTitle: "从记忆工作区，到自动研究",
    sectionLead: "三个完整入口，把 ReMe 用到真实工作流中。",
    products: [
      { mark: "▣", label: "WORKSPACE", title: "ReMe Studio", detail: "在本地 Web 工作区中浏览、编辑、搜索记忆，并探索 wikilink 图谱。", href: "/zh/workspace/studio", tone: "mint" },
      { mark: "◌", label: "DISCOVER", title: "Daily Paper", detail: "筛选值得阅读的论文，分析 PDF，并生成文件化笔记与五分钟简报。", href: "/zh/plugins/daily-paper", tone: "cyan" },
      { mark: "↗", label: "RESEARCH", title: "Auto Fin", detail: "连接最新财联社新闻与本地历史记忆，生成带 wikilink 的研究报告。", href: "/zh/plugins/auto-fin", tone: "amber" },
    ],
    trafficLabel: "04 / OPEN METRICS",
    trafficTitle: "公开、透明的访问趋势",
    trafficDetail: "最近 30 天的页面浏览量与访问趋势，由 Umami 提供匿名统计。",
    trafficAction: "打开完整数据页",
    trafficFrameTitle: "ReMe 最近 30 天访问数据",
  },
  en: {
    eyebrow: "LOCAL-FIRST · FILE-NATIVE",
    title: "Memory for AI agents.\nFiles that remain yours.",
    lead: "ReMe turns conversations, resources, and experience into readable, editable, searchable, interconnected local files—a shared long-term memory layer for every agent.",
    quickStart: "Quick Start",
    learnMore: "Meet ReMe",
    stars: "GitHub Stars",
    forks: "Forks",
    mapLabel: "SHARED MEMORY WORKSPACE",
    mapTitle: "One memory layer for every agent",
    capabilities: [
      { icon: "▱", title: "Memory as files", detail: "User-owned files remain the durable source of truth", href: "/en/memory_as_file", tone: "mint" },
      { icon: "✦", title: "Memory workflows", detail: "Evolve daily records into durable, connected digests", href: "/en/auto_memory", tone: "cyan" },
      { icon: "⌕", title: "Search and graph", detail: "Combine keyword, vector, and wikilink retrieval", href: "/en/memory_search", tone: "blue" },
      { icon: "⌁", title: "Agent integrations", detail: "Connect through CLI, HTTP, MCP, and host adapters", href: "/en/integrations", tone: "amber" },
    ],
    benchmarkLabel: "02 / BENCHMARKS",
    benchmarkTitle: "Memory that holds up\nunder pressure",
    benchmarkLead: "From cross-session retrieval to million-token context, ReMe validates long-term memory with reproducible public benchmarks.",
    benchmarkAction: "Explore all benchmarks",
    benchmarkNote: "Published reference runs · Agentic score",
    benchmarks: [
      { name: "LongMemEval", setting: "500 questions · cleaned-s", score: 89.4 },
      { name: "BEAM 100K", setting: "20 cases · 400 questions", score: 66.1 },
      { name: "BEAM 1M", setting: "35 cases · 700 questions", score: 65.0 },
    ],
    piLabel: "π-Bench proactivity",
    piDetail: "Average PROC score across five personas",
    piDelta: "+2.4% over NanoBot",
    sectionLabel: "03 / PRODUCTS & PLUGINS",
    sectionTitle: "From memory workspace to automated research",
    sectionLead: "Three complete paths for putting ReMe into real workflows.",
    products: [
      { mark: "▣", label: "WORKSPACE", title: "ReMe Studio", detail: "Browse, edit, and search memory in a local web workspace, then explore its wikilink graph.", href: "/en/workspace/studio", tone: "mint" },
      { mark: "◌", label: "DISCOVER", title: "Daily Paper", detail: "Select useful papers, analyze PDFs, and create file-native notes plus a five-minute brief.", href: "/en/plugins/daily-paper", tone: "cyan" },
      { mark: "↗", label: "RESEARCH", title: "Auto Fin", detail: "Connect recent CLS news with local memory to create traceable, wikilink-backed reports.", href: "/en/plugins/auto-fin", tone: "amber" },
    ],
    trafficLabel: "04 / OPEN METRICS",
    trafficTitle: "Public, transparent traffic",
    trafficDetail: "Page views and traffic trends from the last 30 days, measured anonymously with Umami.",
    trafficAction: "Open the full report",
    trafficFrameTitle: "ReMe traffic for the last 30 days",
  },
} as const;

const text = computed(() => translations[props.lang]);
const localLink = (href: string) => withBase(href);

function normalizeCompactCount(value: string) {
  const normalized = value.trim().toUpperCase();
  return /[KMB]$/.test(normalized) ? `${normalized}+` : normalized;
}

async function readBadge(metric: "stars" | "forks") {
  const response = await fetch(`https://img.shields.io/github/${metric}/agentscope-ai/ReMe.json`);
  if (!response.ok) throw new Error(`Unable to load ${metric}`);
  const payload = await response.json();
  return normalizeCompactCount(String(payload.message || payload.value || ""));
}

onMounted(async () => {
  const [stars, forks] = await Promise.allSettled([readBadge("stars"), readBadge("forks")]);
  if (stars.status === "fulfilled" && stars.value) stats.stars = stars.value;
  if (forks.status === "fulfilled" && forks.value) stats.forks = forks.value;
});
</script>

<template>
  <div class="reme-home" :class="{ 'is-zh': lang === 'zh' }">
    <section class="home-stage">
      <div class="hero-copy">
        <p class="eyebrow">{{ text.eyebrow }}</p>
        <h1>{{ text.title }}</h1>
        <p class="hero-lead">{{ text.lead }}</p>

        <div class="hero-actions">
          <a class="action primary" :href="localLink(`/${lang}/quick_start`)">{{ text.quickStart }} <span>→</span></a>
          <a class="action secondary" :href="localLink(`/${lang}/memory_as_file`)">{{ text.learnMore }} <span>↗</span></a>
        </div>

        <div class="repo-stats" aria-live="polite">
          <a :href="`${repository}/stargazers`" target="_blank" rel="noreferrer" :aria-label="`${stats.stars} ${text.stars}`">
            <span class="stat-icon">☆</span>
            <span><strong>{{ stats.stars }}</strong><small>{{ text.stars }}</small></span>
          </a>
          <a :href="`${repository}/forks`" target="_blank" rel="noreferrer" :aria-label="`${stats.forks} ${text.forks}`">
            <span class="stat-icon fork-icon">⑂</span>
            <span><strong>{{ stats.forks }}</strong><small>{{ text.forks }}</small></span>
          </a>
        </div>
      </div>

      <div class="capability-map">
        <div class="map-heading">
          <div>
            <span>{{ text.mapLabel }}</span>
            <strong>{{ text.mapTitle }}</strong>
          </div>
          <img :src="localLink('/reme-icon.svg')" alt="" aria-hidden="true" />
        </div>
        <div class="map-grid">
          <a
            v-for="(capability, index) in text.capabilities"
            :key="capability.title"
            class="capability-card"
            :class="capability.tone"
            :href="localLink(capability.href)"
          >
            <span class="capability-index">0{{ index + 1 }}</span>
            <span class="capability-icon" aria-hidden="true">{{ capability.icon }}</span>
            <div class="capability-copy">
              <strong>{{ capability.title }}</strong>
              <small>{{ capability.detail }}</small>
            </div>
            <span class="card-arrow">↗</span>
          </a>
        </div>
      </div>
    </section>

    <section class="benchmark-section page-panel">
      <div class="benchmark-intro">
        <p class="section-label">{{ text.benchmarkLabel }}</p>
        <h2>{{ text.benchmarkTitle }}</h2>
        <p>{{ text.benchmarkLead }}</p>
        <a :href="localLink(`/${lang}/benchmarks/longmemeval`)">{{ text.benchmarkAction }} <span>→</span></a>
      </div>

      <div class="benchmark-board">
        <div class="benchmark-board-head">
          <span>{{ text.benchmarkNote }}</span>
          <span>0—100%</span>
        </div>
        <div class="benchmark-chart">
          <div v-for="benchmark in text.benchmarks" :key="benchmark.name" class="benchmark-row">
            <div class="benchmark-name">
              <strong>{{ benchmark.name }}</strong>
              <small>{{ benchmark.setting }}</small>
            </div>
            <div class="benchmark-track" aria-hidden="true">
              <span :style="{ width: `${benchmark.score}%` }"></span>
            </div>
            <strong class="benchmark-score">{{ benchmark.score.toFixed(1) }}%</strong>
          </div>
        </div>
        <a class="pi-score" :href="localLink(`/${lang}/benchmarks/pibench`)" :aria-label="`${text.piLabel}: 0.580`">
          <span class="pi-symbol">π</span>
          <span>
            <small>{{ text.piLabel }}</small>
            <strong>0.580</strong>
            <em>{{ text.piDetail }}</em>
          </span>
          <b>{{ text.piDelta }} ↗</b>
        </a>
      </div>
    </section>

    <section class="product-section">
      <div class="section-heading">
        <p class="section-label">{{ text.sectionLabel }}</p>
        <div>
          <h2>{{ text.sectionTitle }}</h2>
          <p>{{ text.sectionLead }}</p>
        </div>
      </div>
      <div class="product-grid">
        <a
          v-for="product in text.products"
          :key="product.title"
          class="product-card"
          :class="product.tone"
          :href="localLink(product.href)"
        >
          <span class="product-mark">{{ product.mark }}</span>
          <span class="product-label">{{ product.label }}</span>
          <strong>{{ product.title }}</strong>
          <p>{{ product.detail }}</p>
          <span class="product-arrow">→</span>
        </a>
      </div>
    </section>

    <section class="traffic-section page-panel">
      <div class="traffic-heading">
        <p class="section-label">{{ text.trafficLabel }}</p>
        <h2>{{ text.trafficTitle }}</h2>
        <p>{{ text.trafficDetail }}</p>
        <a :href="localLink(`/${lang}/traffic`)">{{ text.trafficAction }} <span>→</span></a>
      </div>
      <div class="traffic-window">
        <div class="traffic-window-bar" aria-hidden="true">
          <span></span><span></span><span></span><b>reme.agentscope.io · 30 days</b>
        </div>
        <iframe :src="trafficShareUrl" :title="text.trafficFrameTitle" loading="lazy" referrerpolicy="no-referrer" />
      </div>
    </section>
  </div>
</template>

<style scoped>
.reme-home {
  --home-ink: #17231e;
  --home-muted: #66736d;
  --home-line: #d8e2dc;
  --section-light: #f8faf8;
  --section-tint: #edf4f1;
  max-width: 1720px;
  margin: 0 auto;
  padding: 0 clamp(24px, 4.5vw, 72px) 80px;
  color: var(--home-ink);
}
.home-stage {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(460px, 1fr);
  gap: clamp(42px, 4.5vw, 78px);
  align-items: center;
  min-height: calc(100vh - 64px);
  padding: 72px 0 82px;
}
.home-stage::before {
  position: absolute;
  z-index: -1;
  inset: 0 calc(50% - 50vw);
  background:
    radial-gradient(ellipse 70% 105% at -8% 18%, rgba(21, 158, 126, 0.14), transparent 72%),
    radial-gradient(ellipse 68% 105% at 108% 10%, rgba(77, 103, 211, 0.13), transparent 73%),
    linear-gradient(115deg, #f7fbf8 0%, #fbfaf6 49%, #f7f8fd 100%);
  content: "";
}
.eyebrow, .section-label { margin: 0; color: #0a7f70; font: 750 13px/1.4 var(--vp-font-family-mono); letter-spacing: 0.16em; }
.hero-copy h1 { max-width: 100%; margin: 23px 0 0; color: #121b17; font: 760 clamp(52px, 4.2vw, 78px)/1.04 Georgia, "Times New Roman", serif; white-space: pre; letter-spacing: -0.052em; }
.is-zh .hero-copy h1 { max-width: 760px; font-size: clamp(52px, 3.6vw, 64px); white-space: pre-line; word-break: keep-all; }
.hero-lead { max-width: 650px; margin: 28px 0 0; color: #5e6963; font-size: clamp(17px, 1.3vw, 20px); line-height: 1.75; }
.hero-actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 34px; }
.action { display: inline-flex; align-items: center; justify-content: space-between; gap: 28px; min-width: 166px; min-height: 54px; padding: 0 19px; border: 1px solid #bfcac4; border-radius: 12px; color: #1b2721; background: rgba(255, 255, 255, 0.72); text-decoration: none; font-weight: 720; box-shadow: 0 8px 22px rgba(28, 57, 45, 0.06); transition: transform 160ms ease, box-shadow 160ms ease; }
.action.primary { border-color: #17241e; color: white; background: #17241e; box-shadow: 0 12px 26px rgba(20, 34, 28, 0.2); }
.action:hover { transform: translateY(-2px); box-shadow: 0 15px 28px rgba(28, 57, 45, 0.13); }
.repo-stats { display: flex; flex-wrap: wrap; gap: 34px; margin-top: 40px; }
.repo-stats a { display: flex; gap: 12px; align-items: flex-start; color: inherit; text-decoration: none; }
.stat-icon { color: #087f6a; font-size: 30px; line-height: 1; }
.fork-icon { transform: rotate(90deg); }
.repo-stats strong { display: block; font: 740 28px/1 var(--vp-font-family-mono); letter-spacing: -0.04em; }
.repo-stats small { display: block; margin-top: 8px; color: var(--home-muted); font-size: 13px; }
.capability-map { position: relative; padding: 21px; border: 1px solid rgba(33, 73, 57, 0.16); border-radius: 26px; background: rgba(255, 255, 255, 0.58); box-shadow: 0 30px 70px rgba(36, 67, 54, 0.12); backdrop-filter: blur(18px); }
.map-heading { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 20px; }
.map-heading > div { display: flex; min-width: 0; flex-direction: column; gap: 8px; }
.map-heading span { color: #1c8071; font: 700 10px/1.4 var(--vp-font-family-mono); letter-spacing: 0.12em; }
.map-heading strong { font-size: 18px; line-height: 1.3; white-space: nowrap; }
.map-heading img { width: 42px; height: 42px; flex: none; margin: 0; padding: 6px; border: 1px solid var(--home-line); border-radius: 13px; background: rgba(255, 255, 255, 0.82); box-shadow: 0 8px 18px rgba(25, 55, 43, 0.09); }
.map-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
.capability-card { position: relative; display: grid; grid-template-columns: 25px 34px minmax(0, 1fr) 16px; gap: 11px; align-items: center; min-height: 96px; padding: 17px 18px; overflow: hidden; border: 1px solid color-mix(in srgb, var(--card-accent) 32%, #d9e1dc); border-radius: 15px; color: var(--home-ink); background: radial-gradient(circle at 92% 8%, color-mix(in srgb, var(--card-accent) 15%, transparent), transparent 42%), rgba(255, 255, 255, 0.88); text-decoration: none; transition: transform 170ms ease, box-shadow 170ms ease; }
.capability-card:hover { transform: translateY(-3px); box-shadow: 0 16px 30px color-mix(in srgb, var(--card-accent) 13%, transparent); }
.mint { --card-accent: #059b7f; }.cyan { --card-accent: #169cc4; }.blue { --card-accent: #536bd8; }.amber { --card-accent: #c48324; }.violet { --card-accent: #7654c2; }
.capability-index { color: #8b9690; font: 650 10px/1 var(--vp-font-family-mono); letter-spacing: 0.12em; }
.capability-copy { min-width: 0; }
.capability-icon { display: grid; place-items: center; width: 32px; height: 32px; flex: none; border-radius: 9px; color: var(--card-accent); background: color-mix(in srgb, var(--card-accent) 11%, white); font: 600 21px/1 var(--vp-font-family-mono); }
.capability-card strong { display: block; overflow: hidden; font-size: 17px; line-height: 1.25; text-overflow: ellipsis; white-space: nowrap; }
.capability-card small { display: block; overflow: hidden; margin-top: 6px; color: var(--home-muted); font-size: 12px; line-height: 1.45; text-overflow: ellipsis; white-space: nowrap; }
.card-arrow { color: var(--card-accent); font-size: 16px; }
.page-panel { position: relative; min-height: calc(100vh - 64px); }
.benchmark-section { display: grid; grid-template-columns: minmax(310px, 0.72fr) minmax(560px, 1.28fr); gap: clamp(54px, 7vw, 110px); align-items: center; padding: 112px 0 120px; color: var(--home-ink); }
.benchmark-section::before { position: absolute; z-index: -1; inset: 0 calc(50% - 50vw); border-top: 1px solid rgba(27, 67, 52, 0.08); background: var(--section-tint); content: ""; }
.benchmark-intro .section-label { color: #087f6a; }
.benchmark-intro h2 { max-width: 620px; margin: 23px 0 0; color: var(--home-ink); font-size: clamp(42px, 4.2vw, 68px); line-height: 1.06; white-space: pre-line; letter-spacing: -0.052em; }
.is-zh .benchmark-intro h2 { word-break: keep-all; }
.benchmark-intro > p:not(.section-label) { max-width: 560px; margin: 24px 0 0; color: var(--home-muted); font-size: 17px; line-height: 1.75; }
.benchmark-intro > a, .traffic-heading > a { display: inline-flex; align-items: center; gap: 32px; min-height: 50px; margin-top: 34px; padding: 0 18px; border: 1px solid color-mix(in srgb, var(--home-ink) 24%, transparent); border-radius: 11px; color: var(--home-ink); text-decoration: none; font-weight: 720; transition: background 160ms ease, transform 160ms ease; }
.benchmark-intro > a:hover, .traffic-heading > a:hover { background: color-mix(in srgb, var(--home-ink) 6%, transparent); transform: translateY(-2px); }
.benchmark-board { padding: clamp(24px, 3vw, 38px); border: 1px solid rgba(255, 255, 255, 0.14); border-radius: 26px; color: white; background: #15352b; box-shadow: 0 28px 64px rgba(24, 58, 45, 0.2); }
.benchmark-board-head { display: flex; justify-content: space-between; gap: 20px; padding-bottom: 22px; border-bottom: 1px solid rgba(255, 255, 255, 0.13); color: #94aca3; font: 700 11px/1.4 var(--vp-font-family-mono); letter-spacing: 0.1em; text-transform: uppercase; }
.benchmark-chart { display: grid; gap: 28px; padding: 32px 0; }
.benchmark-row { display: grid; grid-template-columns: 150px minmax(140px, 1fr) 72px; gap: 20px; align-items: center; }
.benchmark-name strong { display: block; color: white; font-size: 16px; }
.benchmark-name small { display: block; margin-top: 5px; color: #8fa69d; font-size: 12px; }
.benchmark-track { height: 10px; overflow: hidden; border-radius: 99px; background: rgba(255, 255, 255, 0.09); }
.benchmark-track span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #38d3ae, #85ead2); box-shadow: 0 0 22px rgba(66, 220, 182, 0.28); }
.benchmark-score { color: white; font: 740 20px/1 var(--vp-font-family-mono); text-align: right; }
.pi-score { display: grid; grid-template-columns: 58px minmax(0, 1fr) auto; gap: 18px; align-items: center; padding: 20px; border: 1px solid rgba(142, 160, 255, 0.27); border-radius: 18px; color: white; background: linear-gradient(110deg, rgba(82, 105, 216, 0.23), rgba(82, 105, 216, 0.08)); text-decoration: none; }
.pi-symbol { display: grid; place-items: center; width: 58px; height: 58px; border-radius: 15px; color: #b9c6ff; background: rgba(107, 129, 236, 0.18); font: 700 31px/1 Georgia, serif; }
.pi-score small, .pi-score em { display: block; color: #aabbb4; font-size: 12px; font-style: normal; }
.pi-score strong { display: block; margin: 4px 0; font: 750 25px/1 var(--vp-font-family-mono); }
.pi-score b { color: #a9b7ff; font-size: 13px; white-space: nowrap; }
.product-section { position: relative; display: flex; min-height: calc(100vh - 64px); flex-direction: column; justify-content: center; padding: 112px 0 120px; }
.product-section::before { position: absolute; z-index: -1; inset: 0 calc(50% - 50vw); border-top: 1px solid rgba(27, 67, 52, 0.08); background: var(--section-light); content: ""; }
.section-heading { display: grid; grid-template-columns: minmax(190px, 0.38fr) minmax(0, 1fr); gap: 40px; align-items: start; margin-bottom: 36px; }
.section-heading h2, .traffic-heading h2 { max-width: 820px; margin: 0; color: var(--home-ink); font-size: clamp(38px, 3.6vw, 58px); line-height: 1.1; letter-spacing: -0.045em; }
.section-heading p:not(.section-label) { margin: 15px 0 0; color: var(--home-muted); font-size: 16px; }
.product-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
.product-card { position: relative; display: flex; min-height: 360px; flex-direction: column; padding: 30px; overflow: hidden; border: 1px solid color-mix(in srgb, var(--card-accent) 30%, var(--home-line)); border-radius: 22px; color: var(--home-ink); background: radial-gradient(circle at 80% 0, color-mix(in srgb, var(--card-accent) 14%, transparent), transparent 50%), linear-gradient(150deg, #fff, color-mix(in srgb, var(--card-accent) 5%, #f7f9f7)); text-decoration: none; transition: transform 170ms ease, box-shadow 170ms ease; }
.product-card:hover { transform: translateY(-4px); box-shadow: 0 20px 38px color-mix(in srgb, var(--card-accent) 14%, transparent); }
.product-mark { display: grid; place-items: center; width: 46px; height: 46px; border-radius: 13px; color: var(--card-accent); background: color-mix(in srgb, var(--card-accent) 13%, white); font: 650 24px/1 var(--vp-font-family-mono); }
.product-label { margin-top: 50px; color: var(--card-accent); font: 720 11px/1.3 var(--vp-font-family-mono); letter-spacing: 0.12em; }
.product-card strong { margin-top: 10px; font-size: 23px; }
.product-card p { max-width: 390px; margin: 13px 0 0; color: var(--home-muted); font-size: 15px; line-height: 1.65; }
.product-arrow { position: absolute; right: 23px; bottom: 20px; color: var(--card-accent); font-size: 22px; }
.traffic-section { display: grid; grid-template-columns: minmax(300px, 0.64fr) minmax(600px, 1.36fr); gap: clamp(48px, 6vw, 94px); align-items: center; padding: 112px 0 116px; color: var(--home-ink); }
.traffic-section::before { position: absolute; z-index: -1; inset: 0 calc(50% - 50vw); border-top: 1px solid rgba(27, 67, 52, 0.08); background: var(--section-tint); content: ""; }
.traffic-heading .section-label { color: #087f6a; }
.traffic-heading h2 { margin-top: 22px; color: var(--home-ink); }
.traffic-heading > p:not(.section-label) { max-width: 460px; margin: 22px 0 0; color: var(--home-muted); font-size: 17px; line-height: 1.7; }
.traffic-window { overflow: hidden; height: min(650px, calc(100vh - 170px)); min-height: 520px; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 24px; background: white; box-shadow: 0 34px 80px rgba(0, 0, 0, 0.3); }
.traffic-window-bar { display: flex; align-items: center; gap: 8px; height: 46px; padding: 0 16px; border-bottom: 1px solid #e5e8e7; background: #f6f8f7; }
.traffic-window-bar span { width: 9px; height: 9px; border-radius: 50%; background: #a8b4af; }
.traffic-window-bar span:first-child { background: #f08d78; }
.traffic-window-bar span:nth-child(2) { background: #e6bf67; }
.traffic-window-bar span:nth-child(3) { background: #69bd9a; }
.traffic-window-bar b { margin-left: 8px; color: #7b8983; font: 650 11px/1 var(--vp-font-family-mono); }
.traffic-window iframe { width: 100%; height: calc(100% - 46px); border: 0; }
:global(.dark) .reme-home { --home-ink: #edf7f3; --home-muted: #a8bbb3; --home-line: #2d4038; --section-light: #0d1512; --section-tint: #14201b; }
:global(.dark) .home-stage::before { background: radial-gradient(ellipse 70% 105% at -8% 18%, rgba(24, 169, 143, 0.13), transparent 72%), radial-gradient(ellipse 68% 105% at 108% 10%, rgba(74, 100, 218, 0.15), transparent 73%), linear-gradient(115deg, #0d1713 0%, #101713 49%, #10131c 100%); }
:global(.dark) .hero-copy h1 { color: var(--home-ink); }
:global(.dark) .action.secondary { border-color: #3b4c45; color: var(--home-ink); background: rgba(20, 32, 27, 0.76); }
:global(.dark) .capability-map { border-color: #2c4037; background: rgba(17, 28, 23, 0.72); }
:global(.dark) .capability-card, :global(.dark) .product-card { background: radial-gradient(circle at 92% 8%, color-mix(in srgb, var(--card-accent) 16%, transparent), transparent 42%), #14201b; }
:global(.dark) .benchmark-section::before { background: var(--section-tint); }
:global(.dark) .product-section::before { background: var(--section-light); }
:global(.dark) .traffic-section::before { background: var(--section-tint); }
:global(.dark) .benchmark-board { background: #0b1712; box-shadow: 0 28px 64px rgba(0, 0, 0, 0.28); }
@media (max-width: 1320px) {
  .home-stage { grid-template-columns: 1fr; min-height: auto; }
  .hero-copy { max-width: 800px; padding-top: 26px; }
  .hero-copy h1 { white-space: normal; }
  .capability-map { max-width: 850px; }
  .map-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .benchmark-section, .traffic-section { grid-template-columns: 1fr; min-height: auto; }
  .benchmark-intro, .traffic-heading { max-width: 720px; }
  .traffic-window { width: 100%; max-width: 1000px; }
}
@media (max-width: 1000px) {
  .product-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 700px) {
  .reme-home { padding-right: 20px; padding-left: 20px; }
  .home-stage { gap: 42px; padding: 52px 0 62px; }
  .hero-copy h1 { font-size: clamp(43px, 13vw, 62px); }
  .hero-lead { font-size: 16px; }
  .map-heading strong { white-space: normal; }
  .map-grid { grid-template-columns: 1fr; }
  .capability-card { min-height: 96px; }
  .section-heading { grid-template-columns: 1fr; gap: 18px; }
  .product-grid { grid-template-columns: 1fr; }
  .product-card { min-height: 280px; }
  .benchmark-section, .product-section, .traffic-section { padding-top: 78px; padding-bottom: 84px; }
  .benchmark-row { grid-template-columns: minmax(0, 1fr) auto; gap: 12px; }
  .benchmark-track { grid-column: 1 / -1; grid-row: 2; }
  .benchmark-score { grid-column: 2; grid-row: 1; }
  .pi-score { grid-template-columns: 48px minmax(0, 1fr); }
  .pi-symbol { width: 48px; height: 48px; }
  .pi-score b { grid-column: 2; }
  .traffic-window { height: 660px; min-height: 0; border-radius: 17px; }
}
</style>
