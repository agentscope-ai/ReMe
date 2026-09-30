import { memoryTags } from "./workspace";
import styles from "./demo.module.css";

export const DEMO_SEARCH = "reme-demo-search";
export default function DemoTags({ content }: { content: string }) {
  return (
    <span className={styles.tags}>
      {memoryTags(content).map((tag) => (
        <button
          key={tag}
          className={styles.tag}
          onClick={() =>
            window.dispatchEvent(new CustomEvent(DEMO_SEARCH, { detail: tag }))
          }
        >
          {tag}
        </button>
      ))}
    </span>
  );
}
