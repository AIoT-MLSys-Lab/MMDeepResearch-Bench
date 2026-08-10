# Current-schema reference output

This directory contains a complete 140-task export for the listed system
`Gemini 3.1 Flash Lite`, generated from the retained current-schema run with
Gemini 2.5 Pro as evaluator.

This is a purpose-built implementation reference for the current schema and
aggregation formula. It was not part of the arXiv leaderboard snapshot. Do not
confuse it with Gemini 3 Flash or use it as a proxy for another historical
system.

Regenerate it from the repository root with:

```bash
python export_leaderboard.py \
  --results reports_runs/gemini31flash/results/gemini31flash.jsonl \
  --output_dir reference_outputs/gemini31flash \
  --model_name "Gemini 3.1 Flash Lite" \
  --require_complete
```

The manifest binds the exported files to the source JSONL by SHA256 and records
the number of contributing tasks for every leaderboard column. MOSAIC dimensions
are type-routed, so `Sem.`, `Acc.`, and `VQA` legitimately have different counts.

`Vef.` is the thresholded 0/100 VEF component used by the official aggregation
formula. Do not substitute `VEF_Raw`, which is retained only for prompt-level
auditing and is not a leaderboard component.
