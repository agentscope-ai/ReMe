import { mkdir, readFile, writeFile } from "node:fs/promises";

// ICNS supports a PNG-backed 512px ic09 element. macOS scales it for smaller sizes.
const png = await readFile(
  new URL("../../workbench/public/reme-app-icon.png", import.meta.url),
);
if (png.readUInt32BE(16) !== 512 || png.readUInt32BE(20) !== 512)
  throw new Error("ReMe app icon source must be 512 × 512 PNG");
const header = Buffer.alloc(16);
header.write("icns", 0);
header.writeUInt32BE(png.length + 16, 4);
header.write("ic09", 8);
header.writeUInt32BE(png.length + 8, 12);
await mkdir(new URL("../out/", import.meta.url), { recursive: true });
await writeFile(
  new URL("../out/reme.icns", import.meta.url),
  Buffer.concat([header, png]),
);
