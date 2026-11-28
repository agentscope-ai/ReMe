import * as SwitchPrimitive from "@radix-ui/react-switch";
import { useContext, useId, type ComponentProps, type ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import { WorkspaceLayout } from "./workspace-titlebar.tsx";

// shadcn/ui-style composition: accessible primitives, styled by the local material tokens.
export function Switch(props: ComponentProps<typeof SwitchPrimitive.Root>) {
  return (
    <SwitchPrimitive.Root className="ui-switch" {...props}>
      <SwitchPrimitive.Thumb className="ui-switch-thumb" />
    </SwitchPrimitive.Root>
  );
}

/** One disclosure primitive for group and Agent menus; selection stays separate. */
export function Disclosure({
  label,
  open,
  onOpenChange,
  className,
  triggerClassName,
  summary,
  actions,
  children,
}: {
  label: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  className?: string;
  triggerClassName?: string;
  summary?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const id = useId();
  const mobile = Boolean(useContext(WorkspaceLayout)?.mobile);
  return (
    <div className={className}>
      <div className="disclosure-heading">
        <button
          className={triggerClassName}
          aria-label={label}
          aria-expanded={open}
          aria-controls={id}
          onClick={() => onOpenChange(!open)}
        >
          <ChevronRight
            className="disclosure-chevron"
            size={13}
            style={{ rotate: open ? "90deg" : undefined }}
            aria-hidden
          />
          {summary}
        </button>
        {actions}
      </div>
      {mobile ? (
        <div
          id={id}
          className="mobile-disclosure"
          data-open={open}
          inert={!open}
        >
          <div>{children}</div>
        </div>
      ) : (
        open && <div id={id}>{children}</div>
      )}
    </div>
  );
}
