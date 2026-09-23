<script setup lang="ts">
import { computed, onMounted, reactive } from "vue";
import { useData, withBase } from "vitepress";

const props = defineProps<{ lang: "zh" | "en" }>();

const repository = "https://github.com/agentscope-ai/ReMe";
const trafficShareBase = "https://cloud.umami.is/analytics/us/share/S1OZK1PSDLEpyiU5?date=30day&page=1";
const { isDark } = useData();
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
    mapLabel: "OPEN ECOSYSTEM",
    mapTitle: "ReMe 与开源生态",
    capabilityTitle: "核心能力",
    capabilities: [
      { mark: "▣", title: "文件即记忆", href: "/zh/memory_as_file", tone: "mint" },
      { mark: "✦", title: "自动记忆", href: "/zh/auto_memory", tone: "cyan" },
      { mark: "⌕", title: "混合检索", href: "/zh/memory_search", tone: "blue" },
      { mark: "⌁", title: "图谱关联", href: "/zh/auto_link", tone: "amber" },
    ],
    agentTitle: "ReMe 接入 Agent",
    agentGuide: "接入指南",
    backendTitle: "检索引擎接入 ReMe",
    backendGuide: "配置指南",
    backendNote: "可选向量索引",
    integrations: [
      { logo: "/ecosystem/qwenpaw.png", title: "QwenPaw", href: "https://github.com/agentscope-ai/QwenPaw", tone: "blue" },
      { logo: "/ecosystem/deepseek-harness.svg", title: "DeepSeek Harness", href: "https://github.com/deepseek-ai/deepseek-harness", tone: "mint" },
      { logo: "/ecosystem/openclaw.svg", title: "OpenClaw", href: "https://github.com/openclaw/openclaw", tone: "cyan" },
      { logo: "/ecosystem/claude-code.png", title: "Claude Code", href: "https://github.com/anthropics/claude-code", tone: "violet" },
      { logo: "/ecosystem/hermes.svg", title: "Hermes Agent", href: "https://github.com/NousResearch/hermes-agent", tone: "amber" },
    ],
    backends: [
      { logo: "/ecosystem/zvec.ico", title: "Zvec", href: "https://github.com/alibaba/zvec", tone: "mint" },
      { logo: "/ecosystem/faiss.png", title: "FAISS", href: "https://github.com/facebookresearch/faiss", tone: "blue" },
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
    mapLabel: "OPEN ECOSYSTEM",
    mapTitle: "ReMe and the open ecosystem",
    capabilityTitle: "CORE CAPABILITIES",
    capabilities: [
      { mark: "▣", title: "Memory as files", href: "/en/memory_as_file", tone: "mint" },
      { mark: "✦", title: "Auto memory", href: "/en/auto_memory", tone: "cyan" },
      { mark: "⌕", title: "Hybrid search", href: "/en/memory_search", tone: "blue" },
      { mark: "⌁", title: "Linked graph", href: "/en/auto_link", tone: "amber" },
    ],
    agentTitle: "ReMe for agents",
    agentGuide: "Integration guide",
    backendTitle: "Retrieval for ReMe",
    backendGuide: "Configuration guide",
    backendNote: "Optional vector indexes",
    integrations: [
      { logo: "/ecosystem/qwenpaw.png", title: "QwenPaw", href: "https://github.com/agentscope-ai/QwenPaw", tone: "blue" },
      { logo: "/ecosystem/deepseek-harness.svg", title: "DeepSeek Harness", href: "https://github.com/deepseek-ai/deepseek-harness", tone: "mint" },
      { logo: "/ecosystem/openclaw.svg", title: "OpenClaw", href: "https://github.com/openclaw/openclaw", tone: "cyan" },
      { logo: "/ecosystem/claude-code.png", title: "Claude Code", href: "https://github.com/anthropics/claude-code", tone: "violet" },
      { logo: "/ecosystem/hermes.svg", title: "Hermes Agent", href: "https://github.com/NousResearch/hermes-agent", tone: "amber" },
    ],
    backends: [
      { logo: "/ecosystem/zvec.ico", title: "Zvec", href: "https://github.com/alibaba/zvec", tone: "mint" },
      { logo: "/ecosystem/faiss.png", title: "FAISS", href: "https://github.com/facebookresearch/faiss", tone: "blue" },
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
const trafficShareUrl = computed(() => `${trafficShareBase}&theme=${isDark.value ? "dark" : "light"}`);
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
          <a class="action secondary" :href="localLink(`/${lang}/overview`)">{{ text.learnMore }} <span>↗</span></a>
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

      <div class="ecosystem-map">
        <div class="map-heading">
          <div>
            <span>{{ text.mapLabel }}</span>
            <strong>{{ text.mapTitle }}</strong>
          </div>
        </div>
        <div class="ecosystem-network">
          <div class="network-brands">
            <div class="network-group-label">
              <strong>{{ text.agentTitle }}</strong>
              <a :href="localLink(`/${lang}/integrations`)">{{ text.agentGuide }} ↗</a>
            </div>
            <div class="brand-viewport">
              <div class="brand-reel">
                <a
                  v-for="integration in text.integrations"
                  :key="integration.title"
                  class="brand-link"
                  :class="integration.tone"
                  :href="integration.href"
                  target="_blank"
                  rel="noopener noreferrer"
                  :aria-label="`${integration.title} GitHub`"
                >
                  <span class="brand-mark" aria-hidden="true"><img :src="localLink(integration.logo)" alt="" /></span>
                  <strong>{{ integration.title }}</strong>
                </a>
                <div v-for="integration in text.integrations" :key="`${integration.title}-clone`" class="brand-link reel-clone" :class="integration.tone" aria-hidden="true">
                  <span class="brand-mark"><img :src="localLink(integration.logo)" alt="" /></span>
                  <strong>{{ integration.title }}</strong>
                </div>
              </div>
            </div>
            <div class="network-group-label backend-label">
              <strong>{{ text.backendTitle }}</strong>
              <a :href="localLink(`/${lang}/memory_search#${lang === 'zh' ? '向量索引后端' : 'vector-index-backends'}`)">{{ text.backendGuide }} ↗</a>
            </div>
            <div class="backend-links">
              <a v-for="backend in text.backends" :key="backend.title" class="brand-link" :class="backend.tone" :href="backend.href" target="_blank" rel="noopener noreferrer" :aria-label="`${backend.title} GitHub`">
                <span class="brand-mark" aria-hidden="true"><img :src="localLink(backend.logo)" alt="" /></span>
                <strong>{{ backend.title }}</strong>
              </a>
            </div>
          </div>
          <div class="network-center" aria-hidden="true">
            <span class="network-ring"></span>
            <span class="network-core"><img :src="localLink('/reme-icon.svg')" alt="" /></span>
            <strong>ReMe</strong>
          </div>
          <div class="network-capabilities">
            <strong class="capability-label">{{ text.capabilityTitle }}</strong>
            <a v-for="capability in text.capabilities" :key="capability.title" class="capability-link" :class="capability.tone" :href="localLink(capability.href)">
              <span aria-hidden="true">{{ capability.mark }}</span>
              <strong>{{ capability.title }}</strong>
              <span aria-hidden="true">↗</span>
            </a>
          </div>
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
  --home-accent: #087f6a;
  --home-surface: #ffffff;
  --home-surface-soft: #f7f9f7;
  --home-glass: rgba(255, 255, 255, 0.72);
  --home-tile: rgba(255, 255, 255, 0.88);
  --home-primary-bg: #17241e;
  --home-primary-text: #ffffff;
  --home-shadow: rgba(28, 57, 45, 0.12);
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
  grid-template-columns: minmax(0, 1fr) 650px;
  gap: clamp(42px, 4vw, 68px);
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
.eyebrow, .section-label { margin: 0; color: var(--home-accent); font: 750 13px/1.4 var(--vp-font-family-mono); letter-spacing: 0.16em; }
.hero-copy h1 { max-width: 100%; margin: 23px 0 0; color: var(--home-ink); font: 760 clamp(52px, 4.2vw, 76px)/1.04 Georgia, "Times New Roman", serif; white-space: pre-wrap; letter-spacing: -0.052em; }
.is-zh .hero-copy h1 { max-width: 760px; font-size: clamp(52px, 3.6vw, 64px); white-space: pre-line; word-break: keep-all; }
.hero-lead { max-width: 650px; margin: 28px 0 0; color: var(--home-muted); font-size: clamp(17px, 1.3vw, 20px); line-height: 1.75; }
.hero-actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 34px; }
.action { display: inline-flex; align-items: center; justify-content: space-between; gap: 28px; min-width: 166px; min-height: 54px; padding: 0 19px; border: 1px solid var(--home-line); border-radius: 12px; color: var(--home-ink); background: var(--home-glass); text-decoration: none; font-weight: 720; box-shadow: 0 8px 22px color-mix(in srgb, var(--home-shadow) 50%, transparent); transition: transform 160ms ease, box-shadow 160ms ease; }
.action.primary { border-color: var(--home-primary-bg); color: var(--home-primary-text); background: var(--home-primary-bg); box-shadow: 0 12px 26px color-mix(in srgb, var(--home-primary-bg) 28%, transparent); }
.action:hover { transform: translateY(-2px); box-shadow: 0 15px 28px var(--home-shadow); }
.repo-stats { display: flex; flex-wrap: wrap; gap: 34px; margin-top: 40px; }
.repo-stats a { display: flex; gap: 12px; align-items: flex-start; color: inherit; text-decoration: none; }
.stat-icon { color: var(--home-accent); font-size: 30px; line-height: 1; }
.fork-icon { transform: rotate(90deg); }
.repo-stats strong { display: block; font: 740 28px/1 var(--vp-font-family-mono); letter-spacing: -0.04em; }
.repo-stats small { display: block; margin-top: 8px; color: var(--home-muted); font-size: 13px; }
.ecosystem-map { position: relative; min-width: 0; }
.map-heading { margin-bottom: 25px; }
.map-heading > div { display: flex; min-width: 0; flex-direction: column; gap: 8px; }
.map-heading span { color: var(--home-accent); font: 700 10px/1.4 var(--vp-font-family-mono); letter-spacing: 0.12em; }
.map-heading strong { font-size: 18px; line-height: 1.3; white-space: nowrap; }
.ecosystem-network { position: relative; display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(100px, 0.62fr) minmax(0, 1fr); gap: 12px; align-items: center; min-height: 370px; }
.ecosystem-network::before, .ecosystem-network::after { position: absolute; z-index: 0; top: 50%; width: 19%; border-top: 1px dashed color-mix(in srgb, var(--home-accent) 58%, var(--home-line)); content: ""; }
.ecosystem-network::before { left: 30%; }.ecosystem-network::after { right: 27%; }
.network-brands, .network-center, .network-capabilities { position: relative; z-index: 1; min-width: 0; }
.network-group-label { display: flex; align-items: baseline; justify-content: space-between; gap: 5px; margin-bottom: 7px; }
.network-group-label strong, .capability-label { color: var(--home-muted); font: 750 10px/1.3 var(--vp-font-family-mono); letter-spacing: 0.04em; }
.network-group-label a { flex: none; color: var(--home-accent); font-size: 10px; font-weight: 700; text-decoration: none; white-space: nowrap; }
.network-group-label a:hover { text-decoration: underline; }
.brand-viewport { height: 184px; overflow: hidden; mask-image: linear-gradient(transparent, #000 12%, #000 88%, transparent); }
.brand-reel { display: grid; grid-auto-rows: 46px; gap: 6px; animation: brand-scroll 19s linear infinite; }
.brand-viewport:hover .brand-reel { animation-play-state: paused; }
.brand-viewport:focus-within { overflow-y: auto; mask-image: none; }
.brand-viewport:focus-within .brand-reel { animation: none; }
.brand-link { display: flex; min-width: 0; height: 46px; align-items: center; gap: 7px; padding: 5px; border: 1px solid color-mix(in srgb, var(--card-accent) 26%, var(--home-line)); border-radius: 9px; color: var(--home-ink); background: var(--home-tile); text-decoration: none; transition: border-color 160ms ease, transform 160ms ease; }
.brand-link:hover { border-color: var(--card-accent); transform: translateX(2px); }
.brand-mark { display: grid; width: 32px; height: 32px; flex: none; place-items: center; overflow: hidden; border: 1px solid var(--home-line); border-radius: 7px; background: #fff; }
.brand-mark img { width: 27px; height: 27px; margin: 0; object-fit: contain; }
.brand-link strong { min-width: 0; font-size: 13px; line-height: 1.15; overflow-wrap: anywhere; }
.reel-clone { pointer-events: none; }
.backend-label { margin-top: 13px; }
.backend-links { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 5px; }
.backend-links .brand-link { height: 42px; gap: 5px; }
.backend-links .brand-mark { width: 28px; height: 28px; }
.backend-links .brand-mark img { width: 23px; height: 23px; }
.network-center { display: flex; min-height: 164px; flex-direction: column; align-items: center; justify-content: center; gap: 16px; }
.network-ring { position: absolute; top: 50%; left: 50%; width: 108px; height: 108px; border: 1px solid color-mix(in srgb, var(--home-accent) 30%, var(--home-line)); border-radius: 50%; transform: translate(-50%, -64%); animation: hub-pulse 3.6s ease-in-out infinite; }
.network-ring::after { position: absolute; inset: 10px; border: 1px solid var(--home-line); border-radius: 50%; content: ""; }
.network-core { z-index: 1; display: grid; width: 64px; height: 64px; place-items: center; border: 1px solid var(--home-line); border-radius: 50%; background: var(--home-surface); box-shadow: 0 10px 28px var(--home-shadow); }
.network-core img { width: 42px; height: 42px; margin: 0; }
.network-center strong { z-index: 1; color: var(--home-accent); font: 750 12px/1 var(--vp-font-family-mono); }
.network-capabilities { display: grid; gap: 8px; }
.capability-label { display: block; margin-bottom: 2px; }
.capability-link { display: flex; min-height: 53px; align-items: center; gap: 7px; padding: 7px; border: 1px solid color-mix(in srgb, var(--card-accent) 30%, var(--home-line)); border-radius: 10px; color: var(--home-ink); background: radial-gradient(circle at 100% 0, color-mix(in srgb, var(--card-accent) 11%, transparent), transparent 70%), var(--home-tile); text-decoration: none; transition: transform 160ms ease, border-color 160ms ease; }
.capability-link:hover { border-color: var(--card-accent); transform: translateX(2px); }
.capability-link span:first-child { display: grid; width: 27px; height: 27px; flex: none; place-items: center; border-radius: 7px; color: var(--card-accent); background: color-mix(in srgb, var(--card-accent) 12%, var(--home-surface)); font-size: 16px; }
.capability-link strong { min-width: 0; font-size: 13px; line-height: 1.2; }
.capability-link span:last-child { margin-left: auto; color: var(--card-accent); font-size: 13px; }
.brand-link:focus-visible, .capability-link:focus-visible, .network-group-label a:focus-visible { outline: 2px solid var(--home-accent); outline-offset: 2px; }
.mint { --card-accent: #059b7f; }.cyan { --card-accent: #169cc4; }.blue { --card-accent: #536bd8; }.amber { --card-accent: #c48324; }.violet { --card-accent: #7654c2; }
@keyframes brand-scroll { to { transform: translateY(-260px); } }
@keyframes hub-pulse { 50% { transform: translate(-50%, -64%) scale(1.12); opacity: 0.55; } }
.page-panel { position: relative; min-height: calc(100vh - 64px); }
.benchmark-section { display: grid; grid-template-columns: minmax(310px, 0.72fr) minmax(560px, 1.28fr); gap: clamp(54px, 7vw, 110px); align-items: center; padding: 112px 0 120px; color: var(--home-ink); }
.benchmark-section::before { position: absolute; z-index: -1; inset: 0 calc(50% - 50vw); border-top: 1px solid var(--home-line); background: var(--section-tint); content: ""; }
.benchmark-intro .section-label { color: var(--home-accent); }
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
.product-section::before { position: absolute; z-index: -1; inset: 0 calc(50% - 50vw); border-top: 1px solid var(--home-line); background: var(--section-light); content: ""; }
.section-heading { display: grid; grid-template-columns: minmax(190px, 0.38fr) minmax(0, 1fr); gap: 40px; align-items: start; margin-bottom: 36px; }
.section-heading h2, .traffic-heading h2 { max-width: 820px; margin: 0; color: var(--home-ink); font-size: clamp(38px, 3.6vw, 58px); line-height: 1.1; letter-spacing: -0.045em; }
.section-heading p:not(.section-label) { margin: 15px 0 0; color: var(--home-muted); font-size: 16px; }
.product-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
.product-card { position: relative; display: flex; min-height: 360px; flex-direction: column; padding: 30px; overflow: hidden; border: 1px solid color-mix(in srgb, var(--card-accent) 30%, var(--home-line)); border-radius: 22px; color: var(--home-ink); background: radial-gradient(circle at 80% 0, color-mix(in srgb, var(--card-accent) 14%, transparent), transparent 50%), linear-gradient(150deg, var(--home-surface), color-mix(in srgb, var(--card-accent) 5%, var(--home-surface-soft))); text-decoration: none; transition: transform 170ms ease, box-shadow 170ms ease; }
.product-card:hover { transform: translateY(-4px); box-shadow: 0 20px 38px color-mix(in srgb, var(--card-accent) 14%, transparent); }
.product-mark { display: grid; place-items: center; width: 46px; height: 46px; border-radius: 13px; color: var(--card-accent); background: color-mix(in srgb, var(--card-accent) 13%, var(--home-surface)); font: 650 24px/1 var(--vp-font-family-mono); }
.product-label { margin-top: 50px; color: var(--card-accent); font: 720 11px/1.3 var(--vp-font-family-mono); letter-spacing: 0.12em; }
.product-card strong { margin-top: 10px; font-size: 23px; }
.product-card p { max-width: 390px; margin: 13px 0 0; color: var(--home-muted); font-size: 15px; line-height: 1.65; }
.product-arrow { position: absolute; right: 23px; bottom: 20px; color: var(--card-accent); font-size: 22px; }
.traffic-section { display: grid; grid-template-columns: minmax(300px, 0.64fr) minmax(600px, 1.36fr); gap: clamp(48px, 6vw, 94px); align-items: center; padding: 112px 0 116px; color: var(--home-ink); }
.traffic-section::before { position: absolute; z-index: -1; inset: 0 calc(50% - 50vw); border-top: 1px solid var(--home-line); background: var(--section-tint); content: ""; }
.traffic-heading .section-label { color: var(--home-accent); }
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
:global(html.dark .reme-home) { --home-ink: #edf7f3; --home-muted: #a8bbb3; --home-line: #2d4038; --home-accent: #57dfc3; --home-surface: #14201b; --home-surface-soft: #101a16; --home-glass: rgba(17, 28, 23, 0.78); --home-tile: rgba(20, 32, 27, 0.92); --home-primary-bg: #57dfc3; --home-primary-text: #07120e; --home-shadow: rgba(0, 0, 0, 0.3); --section-light: #0d1512; --section-tint: #14201b; color-scheme: dark; }
:global(html.dark .home-stage::before) { background: radial-gradient(ellipse 70% 105% at -8% 18%, rgba(24, 169, 143, 0.13), transparent 72%), radial-gradient(ellipse 68% 105% at 108% 10%, rgba(74, 100, 218, 0.15), transparent 73%), linear-gradient(115deg, #0d1713 0%, #101713 49%, #10131c 100%); }
:global(html.dark .benchmark-board) { background: #0b1712; box-shadow: 0 28px 64px rgba(0, 0, 0, 0.32); }
@media (max-width: 1680px) {
  .home-stage { grid-template-columns: 1fr; min-height: auto; }
  .hero-copy { max-width: 800px; padding-top: 26px; }
  .hero-copy h1 { white-space: pre-wrap; }
  .ecosystem-map { max-width: 850px; }
}
@media (max-width: 1320px) {
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
@media (max-width: 520px) {
  .ecosystem-network { grid-template-columns: minmax(0, 1.1fr) 62px minmax(0, 1fr); gap: 4px; }
  .network-group-label { flex-wrap: wrap; }
  .brand-link strong, .capability-link strong { font-size: 10px; }
  .network-ring { width: 78px; height: 78px; }
  .network-core { width: 52px; height: 52px; }
  .network-core img { width: 34px; height: 34px; }
  .capability-link { gap: 3px; padding: 5px; }
  .capability-link span:first-child { width: 22px; height: 22px; font-size: 13px; }
  .backend-links { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .brand-viewport { height: auto; mask-image: none; }
  .brand-reel, .network-ring { animation: none; }
  .reel-clone { display: none; }
}
</style>
