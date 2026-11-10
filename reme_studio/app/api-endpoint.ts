export function normalizeReMeApiUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

export function displayReMeApiEndpoint(
  apiUrl: string,
  origin?: string,
): string {
  return apiUrl || origin || "/";
}
