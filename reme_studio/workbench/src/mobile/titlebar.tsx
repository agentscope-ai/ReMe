import { ChevronDown, ChevronLeft } from "lucide-react";
import { errorText, translator } from "../../../client/src/i18n.ts";
import { useNavigation } from "../navigation.tsx";
import type { TitlebarProps } from "../workspace-titlebar.tsx";

export default function MobileTitlebar({
  path,
  sidebar,
  collapsed,
  settings,
  setSidebarCollapsed,
  children,
  status,
}: TitlebarProps) {
  const t = translator(settings.language);
  const navigation = useNavigation();
  const current = path.at(-1);
  const root = path[0];
  const isList =
    path.length === 1 ||
    current?.onSelect ||
    [t("conversationTab"), t("proactiveOverview"), t("flowAllAgents")].includes(
      current?.label ?? "",
    );
  const back = !isList ? root?.onSelect : undefined;
  const title = isList ? root?.label : current?.label;
  return (
    <header className="workspace-titlebar mobile-titlebar">
      {back && (
        <button
          type="button"
          className="titlebar-button"
          aria-label={t("goBack")}
          onClick={back}
        >
          <ChevronLeft size={23} aria-hidden />
        </button>
      )}
      <h1>
        {sidebar ? (
          <button
            type="button"
            className="mobile-page-picker"
            aria-label={t("sectionActions", {
              name: root?.label ?? title ?? "",
            })}
            aria-controls={sidebar}
            aria-expanded={!collapsed}
            onClick={() => setSidebarCollapsed(!collapsed)}
          >
            <span>{title}</span>
            <ChevronDown size={15} aria-hidden />
          </button>
        ) : (
          title
        )}
      </h1>
      {status}
      <div className="mobile-titlebar-actions">{children}</div>
      {navigation?.error != null && (
        <p className="titlebar-error action-error" role="alert">
          {errorText(navigation.error, settings.language)}
        </p>
      )}
    </header>
  );
}
