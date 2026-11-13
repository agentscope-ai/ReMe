import { useEffect, useState } from "react";
import { deserializeError } from "../shared/i18n.ts";
import type {
  Bootstrap,
  DesktopAction,
  DesktopPatch,
} from "../shared/protocol.ts";

export const mergeState = (state: Bootstrap["state"], patch: DesktopPatch) =>
  patch.revision > state.revision ? { ...state, ...patch } : state;

export function useDesktop(): Bootstrap | null {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  useEffect(() => {
    let live = true;
    let layoutVersion = 0;
    let patches: DesktopPatch[] = [];
    let loading = true;
    let latestLayout: Bootstrap["layout"] | undefined;
    const stopState = window.desktop.onState((patch) => {
      if (loading) patches.push(patch);
      if (live)
        setBootstrap((current) =>
          current
            ? { ...current, state: mergeState(current.state, patch) }
            : null,
        );
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
        setBootstrap({
          ...value,
          state: patches.reduce(mergeState, value.state),
          layout: layoutVersion ? latestLayout! : value.layout,
        });
        patches = [];
      }
    });
    return () => {
      live = false;
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
