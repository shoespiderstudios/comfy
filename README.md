# ComfyUI workflows

This repository is a version-controlled snapshot of the workflows from:

`C:\Users\mrtom\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows`

The tracked workflow files live in [`workflows/`](workflows/). Models, generated images, and input images are intentionally not included.

Locally developed nodes required by these workflows live in [`custom_nodes/`](custom_nodes/). Reproducible workflow builders and audits live in [`tools/`](tools/).

## Actor idea pipeline

- `actor-idea-sketch-batch.json` analyzes three actors once, generates a batch of concise scene cards in one 4B language-model pass, and renders 384px four-step Flux.2 Klein sketches.
- `actor-idea-sketch-ultrafast.json` combines actor inspection and batched scene direction into one 4B vision pass, then renders intentionally rough 256px two-step Flux.2 Klein thumbnails.
- `actor-finalize-direct-uplift.json` reconstructs a selected sketch at 832px with its original short scene card, an 18-step schedule, and 0.65 denoise.
- `actor-finalize-fresh-render.json` refines the selected direction and renders from fresh 1024px noise.
- `actor-finalize-hybrid.json` refines the selected direction and reconstructs the sketch at 0.85 denoise.
- `actor-progression-escalation-loop.json` plans a complete source-aware progression once, then carries each pencil-sketch stage into the next render; an optional permanent source anchor can be enabled if identity drift becomes excessive.

The sketch workflow writes portable candidates beneath `output/actor-pipeline`. Move approved candidate PNGs from `inbox` to `selected`, while leaving their matching actor files in `actors`. Each finalizer reads the same selected set and writes to its own strategy directory.

## Restore

With ComfyUI closed, copy the contents of `workflows/` back into the directory above. Review `git status` and commit the current workflows before replacing anything, so local changes remain recoverable.

## Update the snapshot

With ComfyUI closed or after saving all open workflows, copy the current workflow directory into `workflows/`, then review and commit the diff. ComfyUI JSON files can contain prompts, selected model filenames, image filenames, and other widget values, so inspect the diff before publishing changes.

Some workflows depend on third-party custom nodes and local models, which must still be installed separately. Copy the bundled `custom_nodes/comfyui-storyboard-jobs` directory into ComfyUI's `custom_nodes` directory when restoring the actor pipeline.
