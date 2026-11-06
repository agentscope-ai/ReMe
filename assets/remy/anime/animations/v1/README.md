# Remy animations

Ten animations share a 512 × 512 transparent canvas: `idle`, `listening`, `thinking`, `remembering`, `speaking`,
`reminding`, `happy`, `confused`, `sleeping`, and `dragged`. Each has eight original keyframes and timed holds at 24 fps.
The frames animate independently of a model; speaking has mouth motion, without audio or speech-driven lip sync.

- `reference.png` and `prompts/`: the original identity reference and generation prompts.
- `sheets/`: original eight-frame artwork in a 4 × 2 grid.
- `timeline.json`: source timing, descriptions, ambient pool, segments and render scales.
- `build.mjs`: extracts and scales frames with premultiplied colors to avoid transparent edge bleeding.
- `frames/` and `manifest.json`: desktop runtime assets, derived from the sources above.

Rebuild with Node.js, FFmpeg and ffprobe on PATH:

```bash
node build.mjs
node build.mjs idle speaking     # Rebuild selected clips
node build.mjs --metadata-only   # Refresh manifest after timing changes
```

Once-only clips return to the desktop's current state. Sleeping uses a smaller render scale to keep the seated head
consistent with standing poses. Dragged remains suspended until dragging ends. Use the desktop's animation menu to
preview the clips; no separate preview site, APNG, video, or archive is needed.

The version-2 manifest declares neutral ambient clips and optional `enter` / `hold` / `exit` frame indices. Reminding uses
frames 0–5 to raise the note, holds frame 5 while its bubble is visible, and uses frames 6–7 to lower it. Desktop behavior
owns playback mode; clips contain only timing and frame data. This reuses the same 80 PNGs without new art.
Menu labels are translated in the desktop's shared language catalog; source timeline labels are for artwork authoring.
