<script setup lang="ts">
import { computed, ref } from "vue";
import { useData } from "vitepress";

const { frontmatter, lang } = useData();
const copied = ref(false);
const label = computed(() => {
  if (copied.value) return lang.value.startsWith("zh") ? "已复制" : "Copied";
  return lang.value.startsWith("zh") ? "复制 Markdown" : "Copy Markdown";
});

async function copyMarkdown() {
  const markdown = String(frontmatter.value._rawMarkdown || "");
  if (!markdown) return;
  await navigator.clipboard.writeText(markdown);
  copied.value = true;
  window.setTimeout(() => { copied.value = false; }, 1800);
}
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
