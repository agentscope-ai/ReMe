interface SenderEvent {
  sender: { id: number; isDestroyed(): boolean; readonly mainFrame: object };
  readonly senderFrame: { detached?: boolean } | null;
}

/** Late renderer events and subframes cannot act on desktop windows. */
export function isTrustedSender(
  event: SenderEvent,
  allowed: (id: number) => boolean,
): boolean {
  if (event.sender.isDestroyed() || !allowed(event.sender.id)) return false;
  const frame = event.senderFrame;
  return Boolean(frame && !frame.detached && frame === event.sender.mainFrame);
}
