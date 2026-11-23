// The optional macOS addon is absent on Windows and Linux.
declare module "electron-liquid-glass" {
  const glass: {
    isGlassSupported(): boolean;
    addView(handle: Uint8Array, options: { cornerRadius: number }): number;
  };
  export default glass;
}
