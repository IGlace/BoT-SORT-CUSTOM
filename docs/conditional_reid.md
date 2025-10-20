# Conditional ReID Optimization Overview

This document summarizes the real-time ReID optimizations added to the BoT-SORT tracker, along with guidance for tuning behaviour and performance trade-offs.

## What Changed
- **Lazy ReID triggering** – FastReID inference now runs only when the IoU-based association appears ambiguous. Ambiguity is detected per frame and per detection via `reid_ambiguity_thresh` and `reid_overlap_thresh`.
- **Per-stage instrumentation** – The tracker emits Loguru messages detailing which frames/stages require appearance cues, making it easier to profile ReID usage.
- **Deferred feature seeding** – Confirmed tracks queue a single appearance extraction once they have been stable for `reid_min_track_age` frames, avoiding immediate encoder calls while keeping features ready for future conflicts.
- **Partial embedding fusion** – Cost matrices fall back to IoU everywhere except the rows/columns where embeddings are available, preventing unnecessary encoder work.

### Key Code Touchpoints
- `tracker/bot_sort.py`: conditional embedding extraction, ambiguity heuristics, log statements, and queued feature captures.
- `README.md`: quick-start notes about the new behaviour and tuning knobs.

## Tuning The Behaviour

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `reid_ambiguity_thresh` | 0.1 | Minimum gap between the best/second-best IoU costs before ReID is considered necessary. Smaller values make ReID rarer. | Lower ⇒ fewer encoder calls, riskier associations. Higher ⇒ more appearance checks, safer IDs. |
| `reid_overlap_thresh` | 0.4 | IoU overlap level that flags a detection seen by multiple tracks as ambiguous. | Lower ⇒ more overlap deemed ambiguous (higher ReID load). Higher ⇒ only very tight overlaps trigger ReID. |
| `reid_min_track_age` | 3 | Number of consecutive frames a track must remain active before its first deferred appearance capture. | Lower ⇒ earlier feature snapshots (small upfront cost). Higher ⇒ minimal cost but weaker resilience to early occlusions. |

These values can be supplied on the CLI (e.g. `--reid-ambiguity-thresh 0.15`). If not provided, BoT-SORT uses the defaults above.

## Strategies For Balancing Speed & ID Stability
1. **Ambiguity gap tuning** – Start with `reid_ambiguity_thresh=0.1` and increase gradually until identity swaps disappear on your footage. Monitor the log outputs to confirm ReID frequency.
2. **Overlap sensitivity** – Indoor scenes with dense crowds may benefit from a lower `reid_overlap_thresh` (e.g. `0.3`) to force ReID earlier; sparse outdoor scenes can push this higher to save compute.
3. **Initial snapshot timing** – If early occlusions cause mismatches, reduce `reid_min_track_age` so tracks cache appearance sooner. When latency is critical, keep it at 3 or even raise it.
4. **Logging levels** – Loguru defaults to `INFO`; you can adjust via `logger.remove()` / `logger.add()` in your application to suppress or redirect the ReID diagnostics once tuned.

## Verification & Profiling
- Use `tools/demo.py --with-reid` and watch the log statements to count how often appearance cues are triggered.
- Compare per-frame latency before/after enabling the lazy ReID mode to quantify savings. Ideally, the encoder runs only on ambiguous frames.
- For regression tests, log the total number of ReID triggers across a clip and track it alongside IDF1/HOTA metrics.

## Future Customisations
- Swap the current heuristics with scene-specific signals (e.g., motion history, detection confidence entropy).
- Extend the queue to batch ambiguous detections across a short temporal window before calling the encoder once.
- Add an upper bound on ReID calls per second to keep latency predictable.
