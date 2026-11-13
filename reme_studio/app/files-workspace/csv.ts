/** Read logical CSV records, preserving quoted newlines, escaped quotes, and field whitespace. */
export function parseCsv(content: string): string[][] {
  if (!content) return [];
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  const raw = content.replace(/^\uFEFF/, "");
  for (let index = 0; index < raw.length; index++) {
    const char = raw[index];
    if (char === '"') {
      if (quoted && raw[index + 1] === '"') {
        field += '"';
        index++;
      } else quoted = !quoted;
    } else if (!quoted && (char === "," || char === "\n" || char === "\r")) {
      row.push(field);
      field = "";
      if (char !== ",") {
        rows.push(row);
        row = [];
        if (char === "\r" && raw[index + 1] === "\n") index++;
      }
    } else field += char;
  }
  if (field || row.length || !/[\r\n]$/.test(raw)) rows.push([...row, field]);
  return rows;
}
