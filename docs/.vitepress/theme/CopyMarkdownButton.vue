<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { useData } from "vitepress";

const { frontmatter, lang } = useData();
const copied = ref(false);
const failed = ref(false);
const markdown = computed(() => String(frontmatter.value._rawMarkdown || ""));
let timer: ReturnType<typeof setTimeout> | undefined;
let request = 0;
const label = computed(() => {
  if (copied.value) return lang.value.startsWith("zh") ? "已复制" : "Copied";
  if (failed.value) return lang.value.startsWith("zh") ? "复制失败，请重试" : "Copy failed. Try again";
  return lang.value.startsWith("zh") ? "复制 Markdown" : "Copy Markdown";
});

async function copyMarkdown() {
  if (!markdown.value) return;
  reset();
  const current = request;
  try {
    await navigator.clipboard.writeText(markdown.value);
    if (current !== request) return;
    copied.value = true;
    timer = setTimeout(() => { copied.value = false; }, 1800);
  } catch {
    if (current === request) failed.value = true;
  }
}

function reset() {
  request++;
  clearTimeout(timer);
  copied.value = false;
  failed.value = false;
}
watch(markdown, reset, { flush: "sync" });
onBeforeUnmount(reset);
</script>

<template>
  <div class="copy-markdown-wrap">
    <button class="copy-markdown" type="button" :class="{ copied }" @click="copyMarkdown">
      <svg v-if="!copied" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="9" y="9" width="13" height="13" rx="2" />
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
      </svg>
      <svg v-else viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6" /></svg>
      {{ label }}
    </button>
  </div>
</template>
