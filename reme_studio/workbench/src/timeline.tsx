import {
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { ArrowLeft, ChevronRight, X } from "lucide-react";
import { translator, type Language } from "../../client/src/i18n.ts";
import SplitPane from "./split-pane.tsx";
import { WorkspaceLayout } from "./workspace-titlebar.tsx";
import { useMobileContentMotion } from "./mobile/motion.ts";
import "./timeline.css";

const detailTransitionMs = 250;
const dateOnly = /^\d{4}-\d{2}-\d{2}$/;

function localDay(date: Date) {
  if (!Number.isFinite(date.getTime())) return "";
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
}

export function timelineDay(value?: string | null) {
  if (!value) return "";
  if (!dateOnly.test(value)) return localDay(new Date(value));
  const date = new Date(value + "T12:00:00");
  return localDay(date) === value ? value : "";
}

export function TimelineHeading({
  title,
  icon,
  action,
  controls,
}: {
  title: string;
  icon: ReactNode;
  action?: ReactNode;
  controls?: ReactNode;
}) {
  return (
    <header className="timeline-heading">
      <div className="timeline-heading-title">
        <span className="timeline-heading-icon" aria-hidden>
          {icon}
        </span>
        <h1>{title}</h1>
        {action}
      </div>
      {controls && <div className="timeline-controls">{controls}</div>}
    </header>
  );
}

export function TimelinePane({
  name,
  variant,
  language,
  heading,
  list,
  listKey,
  detailKey,
  onClose,
  children,
}: {
  name: string;
  variant?: "todo" | "idea" | "inbox" | "outbox" | "reminders";
  language: Language;
  heading: ReactNode;
  list: ReactNode;
  listKey?: string;
  detailKey?: string;
  onClose?: () => void;
  children?: ReactNode;
}) {
  const t = translator(language);
  const listRef = useRef<HTMLDivElement>(null);
  const mobile = Boolean(useContext(WorkspaceLayout)?.mobile);
  useMobileContentMotion(listRef, listKey ?? "", mobile && !detailKey);
  const detailRef = useRef<HTMLDivElement>(null);
  const back = useRef<HTMLButtonElement>(null);
  const scrollTop = useRef(0);
  const opener = useRef<HTMLElement | null>(null);
  const open = Boolean(detailKey);
  const [lastDetail, setLastDetail] = useState<ReactNode>();
  useLayoutEffect(() => {
    if (open) setLastDetail(children);
  }, [open, children]);
  useLayoutEffect(() => {
    if (open) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setLastDetail(undefined);
      return;
    }
    const timer = window.setTimeout(
      () => setLastDetail(undefined),
      detailTransitionMs,
    );
    return () => window.clearTimeout(timer);
  }, [open]);
  const wasOpen = useRef(false);
  useLayoutEffect(() => {
    scrollTop.current = 0;
    if (listRef.current) listRef.current.scrollTop = 0;
  }, [listKey]);
  useLayoutEffect(() => {
    const listElement = listRef.current;
    if (detailKey) {
      if (detailRef.current) detailRef.current.scrollTop = 0;
      if (back.current?.getClientRects().length) back.current.focus();
    } else if (wasOpen.current && listElement) {
      listElement.scrollTop = scrollTop.current;
      opener.current?.focus({ preventScroll: true });
    }
    wasOpen.current = Boolean(detailKey);
  }, [detailKey]);
  useEffect(() => {
    if (!detailKey) return;
    const close = (event: Event) => {
      const detail = detailRef.current;
      if (
        event.defaultPrevented ||
        !detail?.getClientRects().length ||
        (onClose && !back.current?.getClientRects().length)
      )
        return;
      event.preventDefault();
      onClose?.();
    };
    window.addEventListener("reme-mobile-back", close);
    return () => window.removeEventListener("reme-mobile-back", close);
  }, [detailKey, onClose]);
  return (
    <section
      className="timeline-pane"
      data-variant={variant}
      data-detail={open}
      style={
        {
          "--timeline-duration": `${detailTransitionMs}ms`,
        } as React.CSSProperties
      }
    >
      <SplitPane
        key={name}
        name={name}
        language={language}
        className="timeline-split"
        defaultWidth={360}
        defaultRatio={0.5}
        minWidth={250}
        maxWidth={Infinity}
        minContentWidth={320}
        resizeLabel={t("resizeResourceList")}
        sidebar={
          <div className="timeline-list-pane">
            {heading}
            <div
              className="timeline-list-scroll"
              ref={listRef}
              onScroll={(event) => {
                if (event.currentTarget.getClientRects().length)
                  scrollTop.current = event.currentTarget.scrollTop;
              }}
              onFocusCapture={(event) => {
                opener.current = event.target;
              }}
            >
              {list}
            </div>
          </div>
        }
      >
        <div className="timeline-detail-pane" inert={!open}>
          {onClose && (open || lastDetail != null) && (
            <div className="timeline-detail-nav">
              <button
                type="button"
                ref={back}
                className="text-action timeline-back"
                onClick={onClose}
              >
                <ArrowLeft size={15} aria-hidden />
                {t("backToList")}
              </button>
              <button
                type="button"
                className="icon-button timeline-close"
                aria-label={t("closeDetails")}
                title={t("closeDetails")}
                onClick={onClose}
              >
                <X size={16} aria-hidden />
              </button>
            </div>
          )}
          <div ref={detailRef} className="timeline-detail-scroll">
            {open ? children : lastDetail}
          </div>
        </div>
      </SplitPane>
    </section>
  );
}

export interface TimelineItem {
  id: string;
  title: string;
  date?: string | null;
  preview?: string;
  metadata?: ReactNode;
  leading?: ReactNode;
  unread?: boolean;
}

export function TimelineBadge({
  children,
  tone,
}: {
  children: ReactNode;
  tone?: "accent" | "warning";
}) {
  return (
    <span className="timeline-badge" data-tone={tone}>
      {children}
    </span>
  );
}

export function TimelineList({
  items,
  language,
  selected,
  onSelect,
  onItemMenu,
  ascending = false,
  relative = true,
  unknownDate,
  undatedTogether = false,
  holdNew = false,
  disabled = false,
}: {
  items: TimelineItem[];
  language: Language;
  selected?: string;
  onSelect: (id: string) => void;
  onItemMenu?: (id: string, x: number, y: number) => boolean | void;
  ascending?: boolean;
  relative?: boolean;
  unknownDate?: string;
  undatedTogether?: boolean;
  holdNew?: boolean;
  disabled?: boolean;
}) {
  const t = translator(language);
  const locale = language === "zh" ? "zh-CN" : "en";
  const formatTime = new Intl.DateTimeFormat(locale, {
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
  const now = new Date();
  const today = localDay(now);
  const yesterday = localDay(
    new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1),
  );
  const list = useRef<HTMLDivElement>(null);
  const known = useRef<Set<string> | undefined>(undefined);
  const [pending, setPending] = useState<Set<string>>();
  useLayoutEffect(() => {
    if (!holdNew) return;
    const ids = new Set(items.map((entry) => entry.id));
    const previous = known.current;
    known.current = ids;
    if (!previous) return;
    const firstExisting = items.findIndex(({ id }) => previous.has(id));
    if (firstExisting < 1 || (list.current?.parentElement?.scrollTop ?? 0) < 24)
      return;
    const newer = items.slice(0, firstExisting).map(({ id }) => id);
    setPending((existing) => {
      const pendingIds = [...(existing ?? []), ...newer];
      return new Set(pendingIds.filter((id) => ids.has(id)));
    });
  }, [holdNew, items]);
  const groups = new Map<string, TimelineItem[]>();
  for (const item of items) {
    if (item.id !== selected && pending?.has(item.id)) continue;
    const key = timelineDay(item.date);
    const group = groups.get(key);
    if (group) group.push(item);
    else groups.set(key, [item]);
  }
  const days = [...groups.keys()].sort((a, b) => {
    if (!a || !b) return Number(!a) - Number(!b);
    return ascending ? a.localeCompare(b) : b.localeCompare(a);
  });
  const showUpdates = items.some(
    ({ id }) => pending?.has(id) && id !== selected,
  );
  const cards = (entries: TimelineItem[]) => (
    <ul className="timeline-cards">
      {entries.map((item) => (
        <li
          className="timeline-card"
          key={item.id}
          data-selected={selected === item.id}
          data-unread={item.unread || undefined}
        >
          {item.leading}
          <button
            type="button"
            className="timeline-card-open"
            aria-pressed={selected === item.id}
            disabled={disabled}
            onClick={() => onSelect(item.id)}
            onContextMenu={
              onItemMenu &&
              ((event) => {
                if (onItemMenu(item.id, event.clientX, event.clientY))
                  event.preventDefault();
              })
            }
          >
            <span className="timeline-card-text">
              {(item.metadata || item.unread) && (
                <span className="timeline-card-meta">
                  {item.unread && <span className="unread-dot" aria-hidden />}
                  {item.metadata}
                </span>
              )}
              <span className="timeline-card-title">{item.title}</span>
              {item.preview && (
                <span className="timeline-card-preview">{item.preview}</span>
              )}
            </span>
            <ChevronRight className="timeline-chevron" size={15} aria-hidden />
          </button>
        </li>
      ))}
    </ul>
  );
  return (
    <div className="timeline-list" ref={list}>
      {showUpdates && (
        <div className="timeline-update">
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setPending(undefined);
              list.current?.parentElement?.scrollTo({ top: 0 });
            }}
          >
            {t("timelineNewItems")}
          </button>
        </div>
      )}
      {days.map((day) => {
        const entries = groups.get(day) ?? [];
        const slots = new Map<number | null, TimelineItem[]>();
        for (const item of entries) {
          const minute =
            item.date && day && !dateOnly.test(item.date)
              ? Math.floor(new Date(item.date).getTime() / 60_000)
              : null;
          const slot = slots.get(minute);
          if (slot) slot.push(item);
          else slots.set(minute, [item]);
        }
        const timed = slots.size > 1 || !slots.has(null);
        const relativeDay =
          relative && day === today
            ? t("today")
            : relative && day === yesterday
              ? t("yesterday")
              : "";
        const caption = day
          ? new Intl.DateTimeFormat(locale, {
              month: "short",
              day: "numeric",
              weekday: relativeDay ? undefined : "short",
              year:
                day.slice(0, 4) === today.slice(0, 4) ? undefined : "numeric",
            }).format(new Date(day + "T12:00:00"))
          : (unknownDate ?? t("timelineUndated"));
        return (
          <section
            className="timeline-group"
            key={day}
            data-undated={undatedTogether && !day}
            data-timed={timed}
          >
            {(!undatedTogether || day) && (
              <h2 className="timeline-date">
                {relativeDay && (
                  <>
                    {relativeDay}{" "}
                    <span className="timeline-date-dot">·</span>{" "}
                  </>
                )}
                {day ? <time dateTime={day}>{caption}</time> : caption}
              </h2>
            )}
            {timed ? (
              <ol className="timeline-slots">
                {[...slots].map(([minute, slot]) => {
                  const time =
                    minute === null ? undefined : new Date(minute * 60_000);
                  return (
                    <li
                      key={minute ?? "date"}
                      className="timeline-slot"
                      data-selected={slot.some((item) => item.id === selected)}
                      data-unread={
                        slot.some((item) => item.unread) || undefined
                      }
                    >
                      <div className="timeline-point">
                        {time && (
                          <time dateTime={time.toISOString()}>
                            {formatTime.format(time)}
                          </time>
                        )}
                      </div>
                      {cards(slot)}
                    </li>
                  );
                })}
              </ol>
            ) : (
              cards(entries)
            )}
          </section>
        );
      })}
    </div>
  );
}

export function DetailHeading({
  title,
  metadata,
  actions,
  long,
}: {
  title: string;
  metadata?: ReactNode;
  actions?: ReactNode;
  long?: boolean;
}) {
  return (
    <header className="detail-heading">
      <div className="detail-heading-text">
        <h1 className="detail-title" data-long={long}>
          {title}
        </h1>
        {metadata && <div className="detail-meta">{metadata}</div>}
      </div>
      {actions && <div className="detail-actions">{actions}</div>}
    </header>
  );
}
