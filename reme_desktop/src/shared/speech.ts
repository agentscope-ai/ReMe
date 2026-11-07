/** Bounded, portable audio and narration shared by the desktop processes. */
export const SAMPLE_RATE = 16_000;
export const MAX_RECORD_MS = 24_000;
export const MIN_RECORD_MS = 300;
export const MAX_AUDIO_BYTES = 768 * 1024;
export const AUTO_SPEECH_CHARS = 600;

export function encodeWave(
  pcm: Int16Array,
  sampleRate = SAMPLE_RATE,
): Uint8Array {
  const bytes = new Uint8Array(44 + pcm.length * 2);
  const view = new DataView(bytes.buffer);
  const ascii = (offset: number, value: string) => {
    [...value].forEach((char, i) => (bytes[offset + i] = char.charCodeAt(0)));
  };
  ascii(0, "RIFF");
  view.setUint32(4, bytes.length - 8, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, pcm.length * 2, true);
  pcm.forEach((value, i) => view.setInt16(44 + i * 2, value, true));
  return bytes;
}

export function validRecording(bytes: Uint8Array): boolean {
  if (bytes.byteLength < 46 || bytes.byteLength > MAX_AUDIO_BYTES) return false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const ascii = (offset: number, length: number) =>
    String.fromCharCode(...bytes.subarray(offset, offset + length));
  return (
    ascii(0, 4) === "RIFF" &&
    ascii(8, 4) === "WAVE" &&
    ascii(12, 4) === "fmt " &&
    ascii(36, 4) === "data" &&
    view.getUint32(4, true) === bytes.length - 8 &&
    view.getUint32(16, true) === 16 &&
    view.getUint16(20, true) === 1 &&
    view.getUint16(22, true) === 1 &&
    view.getUint32(24, true) === SAMPLE_RATE &&
    view.getUint32(28, true) === SAMPLE_RATE * 2 &&
    view.getUint16(32, true) === 2 &&
    view.getUint16(34, true) === 16 &&
    view.getUint32(40, true) === bytes.length - 44 &&
    (bytes.length - 44) % 2 === 0
  );
}

export function narrationText(text: string): string {
  return text
    .replace(/```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)/g, " ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/[^\s<>]+/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)/gm, "")
    .replace(/[`*_~|]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Sentence boundaries keep long narration sequential without oversized provider calls. */
export function narrationParts(text: string, automatic = false): string[] {
  const chars = [...narrationText(text)];
  if (automatic && chars.length > AUTO_SPEECH_CHARS) {
    chars.length = AUTO_SPEECH_CHARS;
    chars.push("…");
  }
  const parts: string[] = [];
  while (chars.length) {
    let end = Math.min(400, chars.length);
    if (end < chars.length) {
      for (let i = end - 1; i >= end / 2; i--)
        if (/[。！？.!?;；\n]/u.test(chars[i])) {
          end = i + 1;
          break;
        }
    }
    const part = chars.splice(0, end).join("").trim();
    if (part) parts.push(part);
  }
  return parts;
}
