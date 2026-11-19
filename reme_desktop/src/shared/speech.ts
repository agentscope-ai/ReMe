import { fromMarkdown } from "mdast-util-from-markdown";
import type { Nodes } from "mdast";

/** Bounded, portable audio and narration shared by the desktop processes. */
export const SAMPLE_RATE = 16_000;
export const MAX_RECORD_MS = 24_000;
export const MIN_RECORD_MS = 300;
export const MAX_AUDIO_BYTES = 768 * 1024;
export const NARRATION_PART_CHARS = 400;
const TARGET_PART_CHARS = 300;

/** Conservative energy gate: reject silence and isolated clicks, not quiet voices. This is not VAD. */
export function hasAudibleSignal(pcm: Int16Array): boolean {
  const window = SAMPLE_RATE / 50;
  let active = 0;
  for (let start = 0; start < pcm.length; start += window) {
    const end = Math.min(start + window, pcm.length);
    let energy = 0;
    let peak = 0;
    for (let i = start; i < end; i++) {
      const value = pcm[i] / 32768;
      energy += value * value;
      peak = Math.max(peak, Math.abs(value));
    }
    if (peak >= 0.002 && Math.sqrt(energy / (end - start)) >= 0.001)
      active += end - start;
  }
  return active >= SAMPLE_RATE * 0.06;
}

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

function narrationBlocks(text: string): string[] {
  const blocks: string[] = [];
  const plain = (node: Nodes): string => {
    if (
      node.type === "code" ||
      node.type === "html" ||
      node.type === "definition"
    )
      return "";
    if (node.type === "text" || node.type === "inlineCode") return node.value;
    if (node.type === "image" || node.type === "imageReference")
      return node.alt ?? "";
    if (node.type === "break") return " ";
    return "children" in node ? node.children.map(plain).join("") : "";
  };
  const visit = (node: Nodes): void => {
    if (node.type === "paragraph" || node.type === "heading") {
      const value = plain(node)
        .replace(/https?:\/\/[^\s<>]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      if (value) blocks.push(value);
    } else if (node.type !== "code" && "children" in node)
      node.children.forEach(visit);
  };
  visit(fromMarkdown(text));
  return blocks;
}

export function narrationText(text: string): string {
  return narrationBlocks(text).join(" ");
}

/** Keep full text, prefer paragraphs and sentences, and bound each provider request. */
export function narrationParts(text: string): string[] {
  const parts: string[] = [];
  const segmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });
  let current = "";
  const length = (value: string) => [...value].length;
  const flush = () => {
    if (current.trim()) parts.push(current.trim());
    current = "";
  };
  const append = (value: string) => {
    const size = length(current),
      added = length(value);
    if (
      size + added > NARRATION_PART_CHARS ||
      (size >= 200 && size + added > TARGET_PART_CHARS)
    )
      flush();
    current += value;
    if (length(current) >= TARGET_PART_CHARS) flush();
  };
  for (const block of narrationBlocks(text)) {
    if (length(current) >= 200) flush();
    if (current) current += " ";
    for (const { segment } of segmenter.segment(block)) {
      const chars = [...segment];
      let offset = 0;
      while (chars.length - offset > NARRATION_PART_CHARS) {
        let end = offset + NARRATION_PART_CHARS;
        for (let i = end - 1; i >= offset + 200; i--) {
          if (
            /[\s，,；;：:]/u.test(chars[i]) &&
            !(/\d/u.test(chars[i - 1]) && /\d/u.test(chars[i + 1] ?? ""))
          ) {
            end = i + 1;
            break;
          }
        }
        const token = /[A-Za-z0-9.,_%+-]/u;
        if (token.test(chars[end - 1]) && token.test(chars[end] ?? "")) {
          let boundary = end;
          while (boundary > offset && token.test(chars[boundary - 1]))
            boundary--;
          if (boundary > offset) end = boundary;
        }
        append(chars.slice(offset, end).join(""));
        offset = end;
      }
      append(chars.slice(offset).join(""));
    }
  }
  flush();
  return parts;
}
