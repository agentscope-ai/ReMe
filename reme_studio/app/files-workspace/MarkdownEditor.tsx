import Editor from "@monaco-editor/react";
import "../monaco-setup";
import { useThemeStore } from "../theme";
import { getLanguage } from "./get-language";

export default function MarkdownEditor({
  path,
  content,
  onChange,
}: {
  path: string;
  content: string;
  onChange: (content: string) => void;
}) {
  const theme = useThemeStore((state) => state.resolved);
  return (
    <Editor
      path={path}
      language={getLanguage(path)}
      value={content}
      theme={theme === "dark" ? "vs-dark" : "vs"}
      onChange={(value) => onChange(value ?? "")}
      options={{
        minimap: { enabled: false },
        fontSize: 13,
        wordWrap: "on",
        scrollBeyondLastLine: false,
        automaticLayout: true,
      }}
    />
  );
}
