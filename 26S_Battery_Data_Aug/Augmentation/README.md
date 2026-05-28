# Binary Layout Augmentation Template (OpenRouter)

Baseline scaffold for generating variants of 2D binary layout matrices via LLM API calls.

This is a domain-agnostic template — fork it and customize the prompts and validators for your own layout problem (e.g. floor plans, circuit masks, channel routing, packing layouts, etc.).

## What this pipeline does

For each layout matrix:

1. loads `.npy` binary layouts from `data/layouts/`
2. converts matrices to JSON-serializable form locally
3. sends strict JSON-only prompts to OpenRouter
4. parses returned JSON variants (full grids or sparse edits)
5. converts variants back to NumPy locally
6. validates shape and binary values
7. saves valid/invalid candidates as `.npy` + PNG
8. writes per-model JSON reports

No API calls are used for image handling, plotting, or format conversion.

## Files

- `config.py` - paths, models, generation settings, env-based API key
- `data_io.py` - load/save `.npy`, matrix↔JSON helpers, PNG saving
- `prompting.py` - **baseline** system/user prompt builders (customize here)
- `openrouter_client.py` - OpenRouter chat API with retry + raw logging
- `validators.py` - **baseline** binary + shape checks (extend here)
- `visualize.py` - matplotlib plotting helpers
- `utils.py` - dirs, JSON extraction, dry-run mock generator
- `augment.py` - main pipeline entrypoint

## Input format

Place your `.npy` files in `data/layouts/`. Requirements:

- 2D arrays
- Binary values only (0 or 1)
- All layouts must share the same shape

The file stem becomes the layout id (e.g. `data/layouts/example_1.npy` → id `example_1`).

## Setup

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Stage example layout(s) from `Real_Data/` into `data/layouts/`:

```bash
# from the Augmentation/ directory
python setup_data.py
```

This copies `Real_Data/Type_C/masks/battery_cells.npy` into `Augmentation/data/layouts/`.
Edit `SOURCE_MASKS` in `setup_data.py` to stage additional masks, or drop your own
`.npy` files into `data/layouts/` directly.

Set your API key:

```bash
export OPENROUTER_API_KEY=<your_api_key_here>
```

(Or copy `.env.example` → `.env` and source it.)

## Run

```bash
# Real API call
python augment.py

# Dry run (no credits, deterministic local mock)
python augment.py --dry-run

# Override variant count
python augment.py --variants-per-layout 6

# Override model list (defaults are free OpenRouter models; the `:free` suffix is required)
# Check https://openrouter.ai/models?max_price=0 for currently-available free slugs.
python augment.py --models deepseek/deepseek-chat-v3.1:free google/gemma-3-27b-it:free
```

## Output structure

All outputs land in the repo-root `outputs/` directory:

```
outputs/
  valid/<model_slug>/from_examples/         # variants that passed validation
  invalid/<model_slug>/from_examples/       # variants that failed
  reports/<model_slug>/from_examples/report.json
  visualizations/<model_slug>/from_examples/
  raw_responses/<model_slug>/from_examples/  # raw model output + meta
  run_report.json                            # aggregate summary
```

## How to customize for your domain

1. **Prompts** (`prompting.py`) — replace the "1 = foreground / 0 = background" language with your domain semantics; add constraints, few-shot examples, etc.
2. **Validators** (`validators.py`) — add domain checks (no-go masks, connectivity, port touches, area limits, ...) to `validate_candidate`.
3. **Auxiliary inputs** — if your prompts need masks or other inputs, load them in `augment.py` and pass them into `build_user_prompt` and `validate_candidate`.
4. **Edit ops** — `_apply_sparse_edits` in `augment.py` supports rectangular and point edits; extend with polylines / macros if needed.

## Notes / limitations

- Text+JSON only (no image input to the model)
- Full-grid output only as a fallback — sparse edits are the preferred format
- Model JSON compliance varies; the extractor is robust to leading/trailing noise but not perfect
- Baseline validators do not check connectivity or any domain constraints — add your own
