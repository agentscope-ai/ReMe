<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{ lang: "zh" | "en" }>();
const shareUrl = "https://cloud.umami.is/analytics/us/share/S1OZK1PSDLEpyiU5?date=30day&page=1";
const text = computed(() => props.lang === "zh" ? {
  eyebrow: "OPEN METRICS",
  title: "ReMe 访问数据",
  description: "最近 30 天的页面浏览量与访问趋势，由 Umami 提供隐私友好的匿名统计。",
  action: "在 Umami 中打开完整页面",
  frameTitle: "ReMe 最近 30 天访问数据",
} : {
  eyebrow: "OPEN METRICS",
  title: "ReMe traffic",
  description: "Page views and traffic trends from the last 30 days, measured anonymously with privacy-friendly Umami analytics.",
  action: "Open the full report in Umami",
  frameTitle: "ReMe traffic for the last 30 days",
});
</script>

<template>
  <main class="traffic-page">
    <header>
      <div>
        <p>{{ text.eyebrow }}</p>
        <h1>{{ text.title }}</h1>
        <span>{{ text.description }}</span>
      </div>
      <a :href="shareUrl" target="_blank" rel="noreferrer">{{ text.action }} ↗</a>
    </header>
    <div class="traffic-frame-wrap">
      <iframe :src="shareUrl" :title="text.frameTitle" loading="eager" referrerpolicy="no-referrer" />
    </div>
  </main>
</template>

<style scoped>
.traffic-page { max-width: 1440px; margin: 0 auto; padding: clamp(54px, 7vw, 96px) clamp(22px, 5vw, 74px) 90px; }
.traffic-page header { display: flex; align-items: flex-end; justify-content: space-between; gap: 40px; margin-bottom: 34px; }
.traffic-page header p { margin: 0 0 15px; color: var(--vp-c-brand-1); font: 750 12px/1.4 var(--vp-font-family-mono); letter-spacing: 0.16em; }
.traffic-page h1 { margin: 0; color: var(--vp-c-text-1); font-size: clamp(42px, 5.5vw, 68px); line-height: 1.05; letter-spacing: -0.05em; }
.traffic-page header span { display: block; max-width: 720px; margin-top: 17px; color: var(--vp-c-text-2); font-size: 17px; line-height: 1.65; }
.traffic-page header a { flex: none; padding: 11px 15px; border: 1px solid var(--vp-c-divider); border-radius: 10px; color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); text-decoration: none; font-size: 14px; font-weight: 700; }
.traffic-page header a:hover { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); }
.traffic-frame-wrap { height: min(880px, calc(100vh - 210px)); min-height: 650px; overflow: hidden; border: 1px solid var(--vp-c-divider); border-radius: 20px; background: white; box-shadow: 0 24px 58px rgba(26, 62, 47, 0.12); }
.traffic-frame-wrap iframe { width: 100%; height: 100%; border: 0; }
@media (max-width: 700px) {
  .traffic-page { padding-top: 42px; }
  .traffic-page header { align-items: flex-start; flex-direction: column; gap: 20px; }
  .traffic-frame-wrap { height: 720px; min-height: 0; border-radius: 14px; }
}
</style>
