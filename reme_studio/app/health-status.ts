import type { ReMeComponentHealth, ReMeHealth, ReMeResponse } from "./types";

export interface ComponentMemoryUsage {
  human?: string;
}

export interface HealthComponentEntry {
  type: string;
  name: string;
  component: ReMeComponentHealth;
  memory?: string;
}

export function healthFromResponse(
  response: Pick<ReMeResponse, "metadata">,
): ReMeHealth | undefined {
  const health = response.metadata.health;
  return health && typeof health === "object"
    ? (health as ReMeHealth)
    : undefined;
}

export function isComponentHealthy(component: ReMeComponentHealth): boolean {
  return (
    component.is_healthy === true ||
    (component.is_started === true && component.is_healthy !== false)
  );
}

export function healthComponentEntries(
  health?: ReMeHealth,
  memory?: Record<string, Record<string, ComponentMemoryUsage>>,
): HealthComponentEntry[] {
  return Object.entries(health?.components || {}).flatMap(([type, entries]) =>
    Object.entries(entries).map(([name, component]) => ({
      type,
      name,
      component,
      memory: memory?.[type]?.[name]?.human,
    })),
  );
}
