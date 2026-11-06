import { useEffect, useState } from "react";
import type { Bootstrap, DesktopAction } from "../shared/protocol.ts";

export function useDesktop(): Bootstrap | null {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  useEffect(() => {
    let live = true;
    let stateVersion = 0;
    let layoutVersion = 0;
    let latestState: Bootstrap["state"] | undefined;
    let latestLayout: Bootstrap["layout"] | undefined;
    const stopState = window.desktop.onState((state) => {
      stateVersion++;
      latestState = state;
      if (live)
        setBootstrap((current) => (current ? { ...current, state } : null));
    });
    const stopLayout = window.desktop.onLayout((layout) => {
      layoutVersion++;
      latestLayout = layout;
      if (live)
        setBootstrap((current) => (current ? { ...current, layout } : null));
    });
    void window.desktop.bootstrap().then((value) => {
      if (live)
        setBootstrap({
          ...value,
          state: stateVersion ? latestState! : value.state,
          layout: layoutVersion ? latestLayout! : value.layout,
        });
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
  await window.desktop.action(value);
}
