import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import type { StudioPage } from "../../client/src/protocol.ts";
import type { NavigationGuard } from "../../client/src/backend.ts";
import {
  travel,
  visit,
  type NavigationHistory,
  type NavigationLocation,
  type NavigationValues,
} from "./navigation-history.ts";

type Locations = Partial<Record<StudioPage, NavigationValues>>;
type PendingNavigation = {
  source: StudioPage;
  target: NavigationLocation;
  ready: boolean;
};

const NavigationContext = createContext<{
  locations: Locations;
  update: (
    page: StudioPage,
    key: string,
    initial: unknown,
    update: (value: unknown) => unknown,
  ) => void;
  block: (page: StudioPage, id: string, blocked: boolean) => void;
  canBack: boolean;
  canForward: boolean;
  blocked: boolean;
  back: () => void;
  forward: () => void;
  error: unknown;
} | null>(null);

export function NavigationProvider({
  page,
  onSelect,
  navigate,
  children,
}: {
  page: StudioPage;
  onSelect: (page: StudioPage) => Promise<void>;
  navigate: NavigationGuard;
  children: ReactNode;
}) {
  const [locations, setLocations] = useState<Locations>({});
  const [history, setHistory] = useState<NavigationHistory>({
    entries: [{ page, values: {} }],
    index: 0,
  });
  const [pending, setPending] = useState<PendingNavigation | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [blocked, setBlocked] = useState<Record<string, StudioPage>>({});
  const current = useRef({ page, locations, history });
  current.current = { page, locations, history };
  const request = useRef<PendingNavigation | null>(null);

  useEffect(() => {
    const location = { page, values: locations[page] ?? {} };
    if (pending) {
      if (
        pending.ready &&
        (page === pending.target.page || page !== pending.source)
      ) {
        request.current = null;
        setPending(null);
      }
      return;
    }
    // Coalesce a page change and the child effects that open its requested item.
    const timer = setTimeout(
      () => setHistory((previous) => visit(previous, location)),
      0,
    );
    return () => clearTimeout(timer);
  }, [page, locations, pending]);

  const update = useCallback(
    (
      page: StudioPage,
      key: string,
      initial: unknown,
      apply: (value: unknown) => unknown,
    ) => {
      setLocations((previous) => {
        const values = previous[page] ?? {};
        const before = key in values ? values[key] : initial;
        const value = apply(before);
        if (JSON.stringify(before) === JSON.stringify(value)) return previous;
        const next = { ...values };
        if (JSON.stringify(value) === JSON.stringify(initial)) delete next[key];
        else next[key] = value;
        return { ...previous, [page]: next };
      });
    },
    [],
  );
  const block = useCallback(
    (page: StudioPage, id: string, blocked: boolean) => {
      setBlocked((previous) => {
        const next = { ...previous };
        if (blocked) next[id] = page;
        else delete next[id];
        return next;
      });
    },
    [],
  );
  const move = (delta: -1 | 1) => {
    if (request.current || Object.values(blocked).includes(page)) return;
    navigate(() => {
      if (request.current) return;
      const previous = current.current;
      const next = travel(previous.history, delta);
      if (next === previous.history) return;
      const target = next.entries[next.index];
      const operation = { source: previous.page, target, ready: false };
      request.current = operation;
      setPending(operation);
      setError(null);
      setLocations((values) => ({ ...values, [target.page]: target.values }));
      setHistory(next);
      void (async () => {
        try {
          if (target.page !== previous.page) await onSelect(target.page);
          if (request.current === operation)
            setPending({ ...operation, ready: true });
        } catch (reason) {
          if (request.current !== operation) return;
          request.current = null;
          setPending(null);
          setLocations(previous.locations);
          setHistory(previous.history);
          setError(reason);
        }
      })();
    });
  };
  const enabled = !pending && !Object.values(blocked).includes(page);
  return (
    <NavigationContext.Provider
      value={{
        locations,
        update,
        block,
        canBack: enabled && history.index > 0,
        canForward: enabled && history.index < history.entries.length - 1,
        blocked: !enabled,
        back: () => move(-1),
        forward: () => move(1),
        error,
      }}
    >
      {children}
    </NavigationContext.Provider>
  );
}

export function useNavigation() {
  return useContext(NavigationContext);
}

/** Keep destinations in the renderer's workspace history; drafts remain component-owned. */
export function useNavigationState<T>(
  page: StudioPage,
  key: string,
  initial: T,
): [T, Dispatch<SetStateAction<T>>] {
  const navigation = useNavigation();
  const [fallback, setFallback] = useState(initial);
  const initialValue = useRef(fallback).current;
  const values = navigation?.locations[page];
  const value = values && key in values ? (values[key] as T) : initialValue;
  const update = navigation?.update;
  const setValue: Dispatch<SetStateAction<T>> = useCallback(
    (next) => {
      if (!update) return setFallback(next);
      update(page, key, initialValue, (current) =>
        typeof next === "function"
          ? (next as (value: T) => T)(current as T)
          : next,
      );
    },
    [update, page, key, initialValue],
  );
  return navigation ? [value, setValue] : [fallback, setValue];
}

export function useNavigationBlocker(page: StudioPage, blocked: boolean) {
  const block = useNavigation()?.block;
  const id = useId();
  useEffect(() => {
    if (!block || !blocked) return;
    block(page, id, true);
    return () => block(page, id, false);
  }, [block, page, id, blocked]);
}
