# sam

Text-prompted segmentation helpers, used to preview what a prompt selects before
committing to a batch run. Mask GENERATION at scale happens in the separate
`samask` repository; these are inspection tools.

- `sl-sam3-qs-text.py` — full SAM3, plots the masks, boxes and scores.
- `efficientsam3-teasertext.py` — EfficientSAM3, prints scores. Faster, useful
  for checking a checkpoint loads and runs.

Both take `--image` and `--prompt`, both required:

```bash
python sl-sam3-qs-text.py --image path/to/image.jpg --prompt "chair"
python sl-sam3-qs-text.py --image path/to/image.jpg --prompt "chair" --device cuda --save out.png
```

**`--prompt` has no default, deliberately.** SAM3 segments whatever concept you
name, and this pipeline is object-agnostic: it reconstructs arbitrary scenes and
objects and contains no domain-specific logic. Both scripts previously carried a
hardcoded prompt, which quietly implied the pipeline was specialized when it is
not. Naming the target every run is the honest interface.

Neither script is imported by the reconstruction pipeline. They are run by hand.
