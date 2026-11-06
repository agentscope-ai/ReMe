import { mkdir, readFile, writeFile } from "node:fs/promises";

// ICNS supports a PNG-backed 512px ic09 element. Reuse the existing artwork
// byte-for-byte; macOS scales this representation for smaller icon sizes.
const png = await readFile(
  new URL(
    "../../assets/remy/anime/animations/v1/frames/idle/00.png",
    import.meta.url,
  ),
);
if (png.readUInt32BE(16) !== 512 || png.readUInt32BE(20) !== 512)
  throw new Error("Remy icon source must be 512 × 512 PNG");
const header = Buffer.alloc(16);
header.write("icns", 0);
header.writeUInt32BE(png.length + 16, 4);
header.write("ic09", 8);
header.writeUInt32BE(png.length + 8, 12);
await mkdir(new URL("../out/", import.meta.url), { recursive: true });
await writeFile(
  new URL("../out/remy.icns", import.meta.url),
  Buffer.concat([header, png]),
);
