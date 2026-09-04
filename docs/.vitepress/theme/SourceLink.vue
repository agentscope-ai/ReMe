<script setup lang="ts">
import { computed } from "vue";
import { useData } from "vitepress";

const { frontmatter, lang } = useData();
const sourcePath = computed(() => String(frontmatter.value._sourcePath || ""));
const label = computed(() => lang.value.startsWith("zh") ? "在 GitHub 查看源文件" : "View source on GitHub");
const href = computed(() => `https://github.com/agentscope-ai/ReMe/blob/main/${sourcePath.value}`);
</script>

<template>
  <div v-if="sourcePath" class="source-link-wrap">
    <a :href="href" target="_blank" rel="noreferrer">{{ label }} ↗</a>
  </div>
</template>
