import { useContext, type CSSProperties } from "react";
import type { LucideIcon } from "lucide-react";
import { WorkspaceLayout } from "../workspace-titlebar.tsx";

export default function MobileSections<T extends string>({
  label,
  items,
  value,
  onSelect,
  disabled = false,
}: {
  label: string;
  items: readonly { value: T; label: string; icon?: LucideIcon }[];
  value: T;
  onSelect: (value: T) => void;
  disabled?: boolean;
}) {
  if (!useContext(WorkspaceLayout)?.mobile) return null;
  return (
    <nav
      className="source-tabs mobile-sections"
      aria-label={label}
      style={
        {
          "--source-tab-count": items.length,
          "--source-tab-index": Math.max(
            0,
            items.findIndex((item) => item.value === value),
          ),
        } as CSSProperties
      }
    >
      {items.map(({ value: id, label, icon: Icon }) => (
        <button
          type="button"
          key={id}
          aria-current={id === value ? "page" : undefined}
          disabled={disabled}
          onClick={() => onSelect(id)}
        >
          {Icon && <Icon size={16} aria-hidden />}
          <span>{label}</span>
        </button>
      ))}
    </nav>
  );
}
