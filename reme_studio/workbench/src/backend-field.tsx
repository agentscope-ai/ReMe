import { translator, type Language } from "../../client/src/i18n.ts";

export default function TextField({
  label,
  value,
  onChange,
  required = false,
  hint,
  type = "text",
  configured = false,
  language,
}: {
  label: string;
  value: string | null;
  onChange: (value: string) => void;
  required?: boolean;
  hint?: string;
  type?: string;
  configured?: boolean;
  language: Language;
}) {
  const t = translator(language);
  return (
    <label className="backend-field">
      <span>
        {label}
        <span className="backend-field-rule">
          {t(required ? "backendRequired" : "backendOptional")}
        </span>
      </span>
      <input
        type={type}
        value={value ?? ""}
        required={required && !configured}
        autoComplete={type === "password" ? "off" : undefined}
        placeholder={configured ? t("backendKeySet") : undefined}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint && <span className="settings-note">{hint}</span>}
    </label>
  );
}
