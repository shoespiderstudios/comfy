# ComfyUI workflows

This repository is a version-controlled snapshot of the workflows from:

`C:\Users\mrtom\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows`

The tracked files live in [`workflows/`](workflows/). Models, generated images, input images, and custom-node installations are intentionally not included.

## Restore

With ComfyUI closed, copy the contents of `workflows/` back into the directory above. Review `git status` and commit the current workflows before replacing anything, so local changes remain recoverable.

## Update the snapshot

With ComfyUI closed or after saving all open workflows, copy the current workflow directory into `workflows/`, then review and commit the diff. ComfyUI JSON files can contain prompts, selected model filenames, image filenames, and other widget values, so inspect the diff before publishing changes.

Some workflows depend on locally installed custom nodes and models. This repository preserves the graph definitions, but those dependencies must be installed separately.
