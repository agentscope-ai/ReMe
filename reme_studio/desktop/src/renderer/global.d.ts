import type { DesktopApi } from "../protocol.ts";
declare global {
  interface Window {
    desktop: DesktopApi;
  }
}
