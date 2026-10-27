"use client";

import { useEffect, useRef } from "react";
import { Check, Monitor, Moon, Sun, SunMoon } from "lucide-react";
import { useI18n } from "./i18n";
import { type ThemePreference, useThemeStore } from "./theme";

export default function ThemeMenu() {
  const { t } = useI18n();
  const menuRef = useRef<HTMLDetailsElement>(null);
  const preference = useThemeStore((state) => state.preference);
  const setPreference = useThemeStore((state) => state.setPreference);
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node))
        menuRef.current?.removeAttribute("open");
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);
  const options: Array<[ThemePreference, string, React.ReactNode]> = [
    ["light", t("lightTheme"), <Sun key="light" size={15} />],
    ["dark", t("darkTheme"), <Moon key="dark" size={15} />],
    ["system", t("systemTheme"), <Monitor key="system" size={15} />],
  ];
  return (
    <details className="theme-menu" ref={menuRef}>
      <summary aria-label={t("appearance")} title={t("appearance")}>
        <SunMoon size={18} />
      </summary>
      <div role="menu">
        {options.map(([value, label, icon]) => (
          <button
            key={value}
            role="menuitemradio"
            aria-checked={preference === value}
            className={preference === value ? "active" : ""}
            onClick={() => {
              setPreference(value);
              menuRef.current?.removeAttribute("open");
            }}
          >
            {icon}
            <span>{label}</span>
            {preference === value && <Check size={14} />}
          </button>
        ))}
      </div>
    </details>
  );
}
