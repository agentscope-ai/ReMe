import { useEffect, useState } from "react";
import { deserializeError } from "../shared/i18n.ts";
import { DesktopUpdates, mergeState } from "./state.ts";
import type {
  Bootstrap,
  DesktopAction,
  DesktopPatch,
} from "../shared/protocol.ts";

export function useDesktop(): Bootstrap | null {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  useEffect(() => {
    let live = true;
    let layoutVersion = 0;
    let patches: DesktopPatch[] = [];
    let loading = true;
    let updates: DesktopUpdates | undefined;
    let latestLayout: Bootstrap["layout"] | undefined;
    const stopState = window.desktop.onState((patch) => {
      if (loading) patches.push(patch);
      else updates?.push(patch);
    });
    const stopLayout = window.desktop.onLayout((layout) => {
      layoutVersion++;
      latestLayout = layout;
      if (live)
        setBootstrap((current) => (current ? { ...current, layout } : null));
    });
    void window.desktop.bootstrap().then((value) => {
      if (live) {
        loading = false;
        const state = patches.reduce(mergeState, value.state);
        updates = new DesktopUpdates(state, (state) =>
          setBootstrap((current) => (current ? { ...current, state } : null)),
        );
        setBootstrap({
          ...value,
          state,
          layout: layoutVersion ? latestLayout! : value.layout,
        });
        patches = [];
      }
    });
    const flush = () => updates?.flush();
    window.addEventListener("focus", flush);
    document.addEventListener("visibilitychange", flush);
    return () => {
      live = false;
      updates?.close();
      window.removeEventListener("focus", flush);
      document.removeEventListener("visibilitychange", flush);
      stopState();
      stopLayout();
    };
  }, []);
  return bootstrap;
}

export async function action(value: DesktopAction): Promise<void> {
  const error = await window.desktop.action(value);
  if (error) throw deserializeError(error);
}
