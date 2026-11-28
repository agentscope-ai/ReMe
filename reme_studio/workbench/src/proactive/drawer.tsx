import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { Maximize2, Minimize2, X } from "lucide-react";

export default function FlowDrawer({
  title,
  closeLabel,
  expandLabel,
  collapseLabel,
  wide = false,
  onClose,
  children,
}: {
  title: string;
  closeLabel: string;
  expandLabel: string;
  collapseLabel: string;
  wide?: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const [full, setFull] = useState(false);
  const [closing, setClosing] = useState(false);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  useEffect(() => {
    if (!closing) return;
    // Also finish if an interrupted animation never emits animationend.
    const timer = setTimeout(() => dialog.current?.close(), 250);
    return () => clearTimeout(timer);
  }, [closing]);
  const close = () => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches)
      dialog.current?.close();
    else setClosing(true);
  };
  return (
    <dialog
      ref={dialog}
      className="flow-drawer"
      data-wide={wide}
      data-full={full}
      data-closing={closing}
      aria-labelledby={titleId}
      onClose={onClose}
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
      onAnimationEnd={(event) => {
        if (closing && event.target === event.currentTarget)
          event.currentTarget.close();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <header>
        <h2 id={titleId}>{title}</h2>
        <div className="flow-drawer-actions">
          {wide && (
            <button
              type="button"
              className="icon-button"
              aria-label={full ? collapseLabel : expandLabel}
              aria-expanded={full}
              onClick={() => setFull((value) => !value)}
            >
              {full ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
          )}
          <button
            type="button"
            className="icon-button"
            aria-label={closeLabel}
            onClick={close}
          >
            <X size={18} />
          </button>
        </div>
      </header>
      <div className="flow-drawer-content">{children}</div>
    </dialog>
  );
}
