# Interview presentation

The final deck is `output/graph-payment-fraud-interview-v3.pptx`. It contains 10 widescreen slides, speaker notes on every slide, editable diagrams, and four native editable charts with embedded workbooks.

## Recommended use

- Rehearse with `demo_script.md` and keep the main presentation to eight minutes.
- Use Presenter View to see the slide notes.
- Run the Streamlit dashboard only after its Day 8 audit passes.
- Keep the synthetic-data and investigator-review caveats visible when discussing results.

The deck deliberately calls connected groups “suspected fraud rings.” It does not claim verified criminal networks, bank performance, calibrated probabilities, or automatic account blocking.

## Source and assets

- `build_deck.mjs` is the editable deck source.
- `assets/network-cover.png` is the generated cover illustration.
- `assets/network-cover.prompt.md` records the exact image-generation prompt and constraints.
- `output/graph-payment-fraud-interview-v3.pptx` is the visually inspected and structurally validated artifact.

The cover is decorative; all analytical charts and diagrams are native slide objects. Slides 6–8 contain editable quantitative charts. Speaker notes cite the repository documents that support each slide.

## Validation

Run the portable delivery audit from the repository root:

```powershell
python scripts/run_day10.py
```

It verifies the OOXML package, slide and notes counts, native chart/workbook counts, embedded media, required metrics, and responsible-use language. The deck was also rendered to images and inspected slide by slide. This validation does not claim that the file was opened in native Microsoft PowerPoint.

## Rebuilding the deck

The source was authored with the bundled `@oai/artifact-tool` presentation runtime. Rebuilding requires that compatible runtime plus these absolute environment variables:

- `PROJECT_ROOT`: this repository root;
- `SKILL_DIR`: the installed presentation-skill directory;
- `RUNTIME_PYTHON`: a Python executable used by the validators;
- `RUNTIME_NODE_MODULES`: the compatible bundled Node modules.

Run `node presentation/build_deck.mjs` from an environment where `@oai/artifact-tool` resolves. The finalizer will not overwrite an existing final file; increment the version and validation folder for a deliberate revision. After rebuilding, render and inspect every slide again before changing the delivered version.
