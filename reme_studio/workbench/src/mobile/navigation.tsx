import { useEffect } from "react";
import { BookOpen, Leaf, Settings, Sparkles } from "lucide-react";
import type { StudioPage, StudioState } from "../../../client/src/protocol.ts";
import { translator } from "../../../client/src/i18n.ts";
import { useNavigation } from "../navigation.tsx";

export function MobileTabBar({
  state,
  onSelect,
}: {
  state: StudioState;
  onSelect: (page: StudioPage) => void;
}) {
  const t = translator(state.settings.language);
  const tabs = [
    ["tasks", t("proactiveAi"), Sparkles],
    ["notebook", t("notebook"), Leaf],
    ["remy", "Remy", null],
    ["knowledge", t("memory"), BookOpen],
    ["settings", t("settings"), Settings],
  ] as const;
  return (
    <nav className="mobile-tabbar" aria-label={t("navigation")}>
      {tabs.map(([page, label, Icon]) => (
        <button
          type="button"
          key={page}
          data-page={page}
          aria-label={label}
          aria-current={state.activePage === page ? "page" : undefined}
          onClick={() => onSelect(page)}
        >
          <span className="mobile-tab-icon" aria-hidden>
            {Icon ? <Icon size={22} /> : <span className="mobile-remy-mark" />}
            {page === "remy" && state.unread > 0 && (
              <i className="mobile-unread" />
            )}
          </span>
          <span>{label}</span>
        </button>
      ))}
    </nav>
  );
}

/** More specific handlers (dialogs, timelines, recording) consume Back first. */
export function MobileBackNavigation() {
  const navigation = useNavigation();
  useEffect(() => {
    const back = (event: Event) => {
      if (event.defaultPrevented) return;
      if (navigation?.blocked) {
        event.preventDefault();
        return;
      }
      if (!navigation?.canBack) return;
      event.preventDefault();
      navigation.back();
    };
    window.addEventListener("reme-mobile-back", back);
    return () => window.removeEventListener("reme-mobile-back", back);
  }, [navigation]);
  return null;
}
