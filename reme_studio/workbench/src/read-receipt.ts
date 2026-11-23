/** Confirm reading after a continuous visible interval; retry until the receipt is reflected in state. */
export function watchReadReceipt(
  read: (signal: AbortSignal) => Promise<boolean>,
  onError: (error: unknown) => void = () => {},
) {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout>;
  let pending = false;
  const visible = () =>
    document.hasFocus() && document.visibilityState === "visible";
  const schedule = () => {
    clearTimeout(timer);
    if (controller.signal.aborted || pending || !visible()) return;
    timer = setTimeout(async () => {
      if (!visible()) return;
      pending = true;
      let retry = true;
      try {
        retry = await read(controller.signal);
      } catch (error) {
        if (!controller.signal.aborted) onError(error);
      } finally {
        pending = false;
        if (retry) schedule();
      }
    }, 600);
  };
  const blur = () => clearTimeout(timer);
  window.addEventListener("focus", schedule);
  window.addEventListener("blur", blur);
  document.addEventListener("visibilitychange", schedule);
  schedule();
  return {
    schedule,
    close() {
      controller.abort();
      clearTimeout(timer);
      window.removeEventListener("focus", schedule);
      window.removeEventListener("blur", blur);
      document.removeEventListener("visibilitychange", schedule);
    },
  };
}
