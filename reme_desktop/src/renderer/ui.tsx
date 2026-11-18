import * as SwitchPrimitive from "@radix-ui/react-switch";
import { useId, type ComponentProps, type ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

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
  children,
}: {
  label: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  className?: string;
  triggerClassName?: string;
  summary?: ReactNode;
  children: ReactNode;
}) {
  const id = useId();
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
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          {summary}
        </button>
      </div>
      {open && <div id={id}>{children}</div>}
    </div>
  );
}
