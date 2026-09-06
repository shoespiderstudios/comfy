import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

import folder_paths


METADATA_KEY = "storyboard_job"
CANDIDATE_METADATA_KEY = "actor_candidate_job"


def _safe_root(value):
    relative = Path((value or "storyboards").strip().strip("/\\"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("storyboard_root must be a relative folder beneath ComfyUI output")
    output = Path(folder_paths.get_output_directory()).resolve()
    root = (output / relative).resolve()
    if root != output and output not in root.parents:
        raise ValueError("storyboard_root escaped ComfyUI output")
    return root


def _tensor_to_pil(image):
    if image is None:
        raise ValueError("An image is required")
    array = image[0].detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def _pil_to_tensor(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _new_job_id(source_image, director_prompt, sketch_seed):
    sample = source_image[0].detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(sample.tobytes())
    digest.update(str(director_prompt).encode("utf-8"))
    digest.update(str(sketch_seed).encode("ascii"))
    token = digest.hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"storyboard_{stamp}_{int(sketch_seed)}_{token}"


def _new_candidate_id(actor_images, scene_prompt, preview_seed):
    digest = hashlib.sha256()
    for image in actor_images:
        digest.update(image[0].detach().cpu().numpy().tobytes())
    digest.update(str(scene_prompt).encode("utf-8"))
    digest.update(str(preview_seed).encode("ascii"))
    token = digest.hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"candidate_{stamp}_{int(preview_seed)}_{token}"


def _actor_set_id(actor_images):
    """Return a stable ID for one ordered set of actor images."""
    digest = hashlib.sha256()
    for label, image in zip(("A", "B", "C"), actor_images):
        sample = image[0].detach().cpu().numpy()
        digest.update(label.encode("ascii"))
        digest.update(str(sample.shape).encode("ascii"))
        digest.update(str(sample.dtype).encode("ascii"))
        digest.update(sample.tobytes())
    return f"actor_set_{digest.hexdigest()[:16]}"


def _metadata(job, mode, render_seed=None, source_path=None):
    return {
        "version": 1,
        "job_id": job["job_id"],
        "mode": mode,
        "created_utc": job.get("created_utc"),
        "director_prompt": job.get("director_prompt", ""),
        "sketch_prompt": job.get("sketch_prompt", ""),
        "director_seed": int(job.get("director_seed", 0)),
        "sketch_seed": int(job.get("sketch_seed", 0)),
        "render_seed": None if render_seed is None else int(render_seed),
        "source_path": str(source_path or job.get("source_path", "")),
        "selected_sketch_path": str(job.get("selected_sketch_path", "")),
    }


def _pnginfo(payload, prompt=None, extra_pnginfo=None):
    info = PngInfo()
    encoded = json.dumps(payload, ensure_ascii=False)
    info.add_text(METADATA_KEY, encoded)
    info.add_text("director_prompt", payload.get("director_prompt", ""))
    info.add_text("sketch_prompt", payload.get("sketch_prompt", ""))
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            info.add_text(str(key), json.dumps(value, ensure_ascii=False))
    return info


def _candidate_pnginfo(payload, prompt=None, extra_pnginfo=None):
    info = PngInfo()
    info.add_text(CANDIDATE_METADATA_KEY, json.dumps(payload, ensure_ascii=False))
    info.add_text("actor_inventory", payload.get("actor_inventory", ""))
    info.add_text("scene_prompt", payload.get("scene_prompt", ""))
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            info.add_text(str(key), json.dumps(value, ensure_ascii=False))
    return info


def _candidate_payload(job, mode, render_seed=None, preview_path=None):
    return {
        "version": 2,
        "candidate_id": job["candidate_id"],
        "mode": mode,
        "created_utc": job.get("created_utc"),
        "actor_inventory": job.get("actor_inventory", ""),
        "scene_prompt": job.get("scene_prompt", ""),
        "director_seed": int(job.get("director_seed", 0)),
        "preview_seed": int(job.get("preview_seed", 0)),
        "render_seed": None if render_seed is None else int(render_seed),
        "settings": job.get("settings", {}),
        "actor_set_id": job.get("actor_set_id", ""),
        "actor_paths": list(job.get("actor_paths", [])),
        "preview_path": str(preview_path or job.get("preview_path", "")),
        "selected_candidate_path": str(job.get("selected_candidate_path", "")),
    }


class CreateStoryboardJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "sketch_image": ("IMAGE",),
                "director_prompt": ("STRING", {"forceInput": True}),
                "sketch_prompt": ("STRING", {"forceInput": True}),
                "director_seed": ("INT", {"forceInput": True}),
                "sketch_seed": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STORYBOARD_JOB",)
    RETURN_NAMES = ("job",)
    FUNCTION = "create"
    CATEGORY = "Storyboard Jobs"

    def create(self, source_image, sketch_image, director_prompt, sketch_prompt,
               director_seed, sketch_seed):
        created = datetime.now(timezone.utc).isoformat()
        job = {
            "job_id": _new_job_id(source_image, director_prompt, sketch_seed),
            "created_utc": created,
            "source_image": source_image,
            "sketch_image": sketch_image,
            "director_prompt": str(director_prompt),
            "sketch_prompt": str(sketch_prompt),
            "director_seed": int(director_seed),
            "sketch_seed": int(sketch_seed),
            "source_path": "",
            "selected_sketch_path": "",
        }
        return (job,)


class StoryboardDirectorySnapshot:
    @classmethod
    def INPUT_TYPES(cls):
        default = str(Path(folder_paths.get_output_directory()) / "storyboards" / "selected")
        return {"required": {"directory": ("STRING", {"default": default})}}

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("snapshot", "count", "summary")
    FUNCTION = "snapshot"
    CATEGORY = "Storyboard Jobs"

    @staticmethod
    def _files(directory):
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Selected-storyboard directory does not exist: {root}")
        return sorted(
            (path.resolve() for path in root.iterdir()
             if path.is_file() and path.suffix.lower() == ".png"),
            key=lambda path: (path.name.casefold(), path.name),
        )

    @classmethod
    def IS_CHANGED(cls, directory):
        try:
            digest = hashlib.sha256(str(Path(directory).resolve()).encode("utf-8"))
            for path in cls._files(directory):
                stat = path.stat()
                digest.update(path.name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
            return digest.hexdigest()
        except OSError:
            return float("NaN")

    def snapshot(self, directory):
        paths = self._files(directory)
        if not paths:
            raise FileNotFoundError(f"No selected storyboard PNG files found in: {Path(directory).resolve()}")
        value = json.dumps([str(path) for path in paths], ensure_ascii=False)
        return (value, len(paths), f"{len(paths)} selected storyboard job(s)")


class LoadStoryboardJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "snapshot": ("STRING", {"forceInput": True}),
                "index": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STORYBOARD_JOB",)
    RETURN_NAMES = ("job",)
    FUNCTION = "load"
    CATEGORY = "Storyboard Jobs"

    def load(self, snapshot, index):
        paths = json.loads(snapshot)
        if index < 0 or index >= len(paths):
            raise IndexError(f"Storyboard index {index} is outside a snapshot of {len(paths)} jobs")
        sketch_path = Path(paths[index]).resolve()
        with Image.open(sketch_path) as opened:
            raw = opened.info.get(METADATA_KEY)
            if not raw:
                raise ValueError(f"Storyboard metadata '{METADATA_KEY}' is missing from {sketch_path.name}")
            payload = json.loads(raw)
            sketch_image = _pil_to_tensor(opened)

        source_path = Path(payload.get("source_path", "")).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"The source image recorded by {sketch_path.name} is missing: {source_path}")
        with Image.open(source_path) as source:
            source_image = _pil_to_tensor(source)

        job = dict(payload)
        job.update({
            "source_image": source_image,
            "sketch_image": sketch_image,
            "source_path": str(source_path),
            "selected_sketch_path": str(sketch_path),
        })
        return (job,)


class UnpackStoryboardJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": ("STORYBOARD_JOB",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "source_image", "sketch_image", "director_prompt", "sketch_prompt",
        "job_id", "director_seed", "sketch_seed",
    )
    FUNCTION = "unpack"
    CATEGORY = "Storyboard Jobs"

    def unpack(self, job):
        return (
            job["source_image"], job["sketch_image"], job.get("director_prompt", ""),
            job.get("sketch_prompt", ""), job.get("job_id", ""),
            int(job.get("director_seed", 0)), int(job.get("sketch_seed", 0)),
        )


class SaveStoryboardOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_render_mode": ("BOOLEAN", {"default": False}),
                "job": ("STORYBOARD_JOB",),
                "image": ("IMAGE",),
                "render_seed": ("INT", {"forceInput": True}),
                "storyboard_root": ("STRING", {"default": "storyboards"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "saved_path")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Storyboard Jobs"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def save(self, full_render_mode, job, image, render_seed, storyboard_root,
             prompt=None, extra_pnginfo=None):
        root = _safe_root(storyboard_root)
        inbox = root / "inbox"
        selected = root / "selected"
        sources = root / "sources"
        renders = root / "renders"
        for directory in (inbox, selected, sources, renders):
            directory.mkdir(parents=True, exist_ok=True)

        job = dict(job)
        job_id = job.get("job_id") or _new_job_id(
            job["source_image"], job.get("director_prompt", ""), job.get("sketch_seed", 0)
        )
        job["job_id"] = job_id

        if full_render_mode:
            destination = renders / f"{job_id}_render.png"
            counter = 1
            while destination.exists():
                destination = renders / f"{job_id}_render_{counter:03d}.png"
                counter += 1
            payload = _metadata(job, "full_render", render_seed=render_seed)
            _tensor_to_pil(image).save(
                destination, pnginfo=_pnginfo(payload, prompt, extra_pnginfo), compress_level=4
            )
            relative_dir = renders.relative_to(Path(folder_paths.get_output_directory()).resolve())
        else:
            destination = inbox / f"{job_id}.png"
            counter = 1
            while destination.exists():
                job_id = f"{job['job_id']}_{counter:03d}"
                destination = inbox / f"{job_id}.png"
                counter += 1
            job["job_id"] = job_id
            source_path = sources / f"{job_id}_source.png"
            source_payload = _metadata(job, "source", render_seed=None, source_path=source_path)
            _tensor_to_pil(job["source_image"]).save(
                source_path, pnginfo=_pnginfo(source_payload, prompt, extra_pnginfo), compress_level=4
            )
            job["source_path"] = str(source_path)
            payload = _metadata(job, "storyboard", render_seed=None, source_path=source_path)
            _tensor_to_pil(image).save(
                destination, pnginfo=_pnginfo(payload, prompt, extra_pnginfo), compress_level=4
            )
            with (root / "storyboard-index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            relative_dir = inbox.relative_to(Path(folder_paths.get_output_directory()).resolve())

        ui_image = {"filename": destination.name, "subfolder": str(relative_dir), "type": "output"}
        return {"ui": {"images": [ui_image]}, "result": (image, str(destination))}


class CreateActorCandidateJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "actor_a": ("IMAGE",),
                "actor_b": ("IMAGE",),
                "actor_c": ("IMAGE",),
                "preview_image": ("IMAGE",),
                "actor_inventory": ("STRING", {"forceInput": True}),
                "scene_prompt": ("STRING", {"forceInput": True}),
                "director_seed": ("INT", {"forceInput": True}),
                "preview_seed": ("INT", {"forceInput": True}),
                "settings_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("ACTOR_CANDIDATE_JOB",)
    RETURN_NAMES = ("job",)
    FUNCTION = "create"
    CATEGORY = "Actor Candidate Jobs"

    def create(self, actor_a, actor_b, actor_c, preview_image, actor_inventory,
               scene_prompt, director_seed, preview_seed, settings_json):
        try:
            settings = json.loads(settings_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"settings_json is not valid JSON: {exc}") from exc
        actors = [actor_a, actor_b, actor_c]
        job = {
            "candidate_id": _new_candidate_id(actors, scene_prompt, preview_seed),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "actor_images": actors,
            "preview_image": preview_image,
            "actor_inventory": str(actor_inventory),
            "scene_prompt": str(scene_prompt),
            "director_seed": int(director_seed),
            "preview_seed": int(preview_seed),
            "settings": settings,
            "actor_paths": [],
            "preview_path": "",
            "selected_candidate_path": "",
        }
        return (job,)


class CandidateDirectorySnapshot:
    @classmethod
    def INPUT_TYPES(cls):
        default = str(Path(folder_paths.get_output_directory()) / "candidates" / "selected")
        return {"required": {"directory": ("STRING", {"default": default})}}

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("snapshot", "count", "summary")
    FUNCTION = "snapshot"
    CATEGORY = "Actor Candidate Jobs"

    @staticmethod
    def _files(directory):
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Selected-candidate directory does not exist: {root}")
        result = []
        for path in root.iterdir():
            if not path.is_file() or path.suffix.lower() != ".png":
                continue
            try:
                with Image.open(path) as opened:
                    if opened.info.get(CANDIDATE_METADATA_KEY):
                        result.append(path.resolve())
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda path: (path.name.casefold(), path.name))

    @classmethod
    def IS_CHANGED(cls, directory):
        try:
            digest = hashlib.sha256(str(Path(directory).resolve()).encode("utf-8"))
            for path in cls._files(directory):
                stat = path.stat()
                digest.update(path.name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
            return digest.hexdigest()
        except OSError:
            return float("NaN")

    def snapshot(self, directory):
        paths = self._files(directory)
        if not paths:
            raise FileNotFoundError(f"No portable candidate PNG files found in: {Path(directory).resolve()}")
        return (
            json.dumps([str(path) for path in paths], ensure_ascii=False),
            len(paths),
            f"{len(paths)} selected candidate(s)",
        )


class LoadActorCandidateJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "snapshot": ("STRING", {"forceInput": True}),
                "index": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("ACTOR_CANDIDATE_JOB",)
    RETURN_NAMES = ("job",)
    FUNCTION = "load"
    CATEGORY = "Actor Candidate Jobs"

    def load(self, snapshot, index):
        paths = json.loads(snapshot)
        if index < 0 or index >= len(paths):
            raise IndexError(f"Candidate index {index} is outside a snapshot of {len(paths)} jobs")
        preview_path = Path(paths[index]).resolve()
        with Image.open(preview_path) as opened:
            raw = opened.info.get(CANDIDATE_METADATA_KEY)
            if not raw:
                raise ValueError(f"Candidate metadata '{CANDIDATE_METADATA_KEY}' is missing from {preview_path.name}")
            payload = json.loads(raw)
            preview_image = _pil_to_tensor(opened)

        actor_paths = [Path(value).expanduser().resolve() for value in payload.get("actor_paths", [])]
        if len(actor_paths) != 3:
            raise ValueError(f"{preview_path.name} does not record exactly three actor paths")
        actor_images = []
        for actor_path in actor_paths:
            if not actor_path.is_file():
                raise FileNotFoundError(f"Actor image recorded by {preview_path.name} is missing: {actor_path}")
            with Image.open(actor_path) as opened:
                actor_images.append(_pil_to_tensor(opened))

        job = dict(payload)
        job.update({
            "actor_images": actor_images,
            "preview_image": preview_image,
            "actor_paths": [str(path) for path in actor_paths],
            "preview_path": str(preview_path),
            "selected_candidate_path": str(preview_path),
        })
        return (job,)


class UnpackActorCandidateJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": ("ACTOR_CANDIDATE_JOB",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "actor_a", "actor_b", "actor_c", "preview_image", "actor_inventory",
        "scene_prompt", "candidate_id", "director_seed", "preview_seed", "settings_json",
    )
    FUNCTION = "unpack"
    CATEGORY = "Actor Candidate Jobs"

    def unpack(self, job):
        actors = job["actor_images"]
        return (
            actors[0], actors[1], actors[2], job["preview_image"],
            job.get("actor_inventory", ""), job.get("scene_prompt", ""),
            job.get("candidate_id", ""), int(job.get("director_seed", 0)),
            int(job.get("preview_seed", 0)),
            json.dumps(job.get("settings", {}), ensure_ascii=False),
        )


class SaveActorCandidateOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "finalize_mode": ("BOOLEAN", {"default": False}),
                "job": ("ACTOR_CANDIDATE_JOB",),
                "image": ("IMAGE",),
                "render_seed": ("INT", {"forceInput": True}),
                "candidate_root": ("STRING", {"default": "candidates"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "saved_path")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Actor Candidate Jobs"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def save(self, finalize_mode, job, image, render_seed, candidate_root,
             prompt=None, extra_pnginfo=None):
        root = _safe_root(candidate_root)
        inbox = root / "inbox"
        selected = root / "selected"
        actors_dir = root / "actors"
        final_dir = root / "final"
        for directory in (inbox, selected, actors_dir, final_dir):
            directory.mkdir(parents=True, exist_ok=True)

        job = dict(job)
        candidate_id = job.get("candidate_id") or _new_candidate_id(
            job["actor_images"], job.get("scene_prompt", ""), job.get("preview_seed", 0)
        )
        job["candidate_id"] = candidate_id

        if finalize_mode:
            destination = final_dir / f"{candidate_id}_final.png"
            counter = 1
            while destination.exists():
                destination = final_dir / f"{candidate_id}_final_{counter:03d}.png"
                counter += 1
            payload = _candidate_payload(job, "final", render_seed=render_seed)
            _tensor_to_pil(image).save(
                destination,
                pnginfo=_candidate_pnginfo(payload, prompt, extra_pnginfo),
                compress_level=4,
            )
            relative_dir = final_dir.relative_to(Path(folder_paths.get_output_directory()).resolve())
        else:
            actor_set_id = _actor_set_id(job["actor_images"])
            actor_paths = []
            for label, actor_image in zip(("A", "B", "C"), job["actor_images"]):
                actor_path = actors_dir / f"{actor_set_id}_actor_{label}.png"
                if not actor_path.exists():
                    _tensor_to_pil(actor_image).save(actor_path, compress_level=4)
                actor_paths.append(str(actor_path))
            job["actor_set_id"] = actor_set_id
            job["actor_paths"] = actor_paths

            destination = inbox / f"{candidate_id}.png"
            counter = 1
            while destination.exists():
                destination = inbox / f"{candidate_id}_{counter:03d}.png"
                counter += 1
            job["preview_path"] = str(destination)
            payload = _candidate_payload(job, "preview", preview_path=destination)
            _tensor_to_pil(image).save(
                destination,
                pnginfo=_candidate_pnginfo(payload, prompt, extra_pnginfo),
                compress_level=4,
            )
            with (root / "candidate-index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            relative_dir = inbox.relative_to(Path(folder_paths.get_output_directory()).resolve())

        ui_image = {"filename": destination.name, "subfolder": str(relative_dir), "type": "output"}
        return {"ui": {"images": [ui_image]}, "result": (image, str(destination))}


class ParseActorSceneCards:
    """Turn one batched director reply into a stable JSON snapshot for a render loop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_text": ("STRING", {"forceInput": True}),
                "requested_count": ("INT", {"forceInput": True, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("snapshot", "count", "summary")
    FUNCTION = "parse"
    CATEGORY = "Actor Candidate Jobs"

    @staticmethod
    def _clean_line(value):
        value = value.strip()
        value = re.sub(r"^(?:SCENE\s*)?\d+\s*[:.)-]\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^(?:[-*]\s+|SCENE\s*:\s*)", "", value, flags=re.IGNORECASE)
        return value.strip().strip('"')

    def parse(self, raw_text, requested_count):
        text = re.sub(r"<think>.*?</think>", "", str(raw_text), flags=re.IGNORECASE | re.DOTALL)
        lines = [self._clean_line(line) for line in text.replace("|||", "\n").splitlines()]
        cards = [line for line in lines if line]
        if len(cards) <= 1:
            paragraphs = [self._clean_line(value) for value in re.split(r"\n\s*\n+", text)]
            cards = [value for value in paragraphs if value]
        count = min(int(requested_count), len(cards))
        cards = cards[:count]
        if not cards:
            raise ValueError("The director reply did not contain any usable scene cards")
        return (json.dumps(cards, ensure_ascii=False), len(cards), f"{len(cards)} scene card(s) parsed")


class BuildActorSceneBatchPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "actor_inventory": ("STRING", {"forceInput": True}),
                "direction": ("STRING", {"forceInput": True}),
                "count": ("INT", {"forceInput": True, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "Actor Candidate Jobs"

    def build(self, actor_inventory, direction, count):
        prompt = f"""ACTORS:
{actor_inventory}

DIRECTION:
{direction}

Invent exactly {int(count)} distinct, photographable scene ideas. Keep each idea short and concrete: identify ACTOR A, ACTOR B, and ACTOR C; give their positions and one clear visible action; then name the setting, camera framing, and lighting. Use simple language. Avoid tangled bodies, unclear limb ownership, impossible joints, mirrors, crowds, fantasy, science fiction, and alternatives. Return exactly one complete scene per line. Begin each line with SCENE 1:, SCENE 2:, and so on. Add no preface, analysis, headings, blank lines, or continuation lines."""
        return (prompt,)


class BuildActorFinalPromptRequest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "actor_inventory": ("STRING", {"forceInput": True}),
                "scene_prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "Actor Candidate Jobs"

    def build(self, actor_inventory, scene_prompt):
        prompt = f"""ACTORS:
{actor_inventory}

APPROVED SCENE CARD:
{scene_prompt}

Rewrite the approved card as one self-contained production prompt for a high-capacity, reference-conditioned Flux.2 image model at 1024 pixels. Preserve the approved actors, action, spatial arrangement, setting, camera viewpoint, and lighting. Add only visible details that improve photographic clarity, anatomy, materials, contact points, and unambiguous limb ownership. Keep all three actors distinct and recognizable. Do not introduce new people, actions, alternatives, backstory, fantasy, science fiction, mirrors, or crowds. Output only one concise paragraph with no heading, analysis, quotation marks, or preface."""
        return (prompt,)


class ActorSceneCardAtIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "snapshot": ("STRING", {"forceInput": True}),
                "index": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_prompt",)
    FUNCTION = "get"
    CATEGORY = "Actor Candidate Jobs"

    def get(self, snapshot, index):
        cards = json.loads(snapshot)
        if index < 0 or index >= len(cards):
            raise IndexError(f"Scene-card index {index} is outside a batch of {len(cards)} cards")
        return (str(cards[index]),)


class SetActorCandidateFinalPrompt:
    """Preserve the cheap scene card while recording the prompt used for a final render."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job": ("ACTOR_CANDIDATE_JOB",),
                "final_prompt": ("STRING", {"forceInput": True}),
                "strategy": ("STRING", {"default": "fresh-render"}),
            }
        }

    RETURN_TYPES = ("ACTOR_CANDIDATE_JOB",)
    RETURN_NAMES = ("job",)
    FUNCTION = "update"
    CATEGORY = "Actor Candidate Jobs"

    def update(self, job, final_prompt, strategy):
        updated = dict(job)
        settings = dict(updated.get("settings", {}))
        settings["original_scene_prompt"] = updated.get("scene_prompt", "")
        settings["final_prompt"] = str(final_prompt)
        settings["finishing_strategy"] = str(strategy)
        updated["settings"] = settings
        updated["scene_prompt"] = str(final_prompt)
        return (updated,)


NODE_CLASS_MAPPINGS = {
    "CreateStoryboardJob": CreateStoryboardJob,
    "StoryboardDirectorySnapshot": StoryboardDirectorySnapshot,
    "LoadStoryboardJob": LoadStoryboardJob,
    "UnpackStoryboardJob": UnpackStoryboardJob,
    "SaveStoryboardOutput": SaveStoryboardOutput,
    "CreateActorCandidateJob": CreateActorCandidateJob,
    "CandidateDirectorySnapshot": CandidateDirectorySnapshot,
    "LoadActorCandidateJob": LoadActorCandidateJob,
    "UnpackActorCandidateJob": UnpackActorCandidateJob,
    "SaveActorCandidateOutput": SaveActorCandidateOutput,
    "ParseActorSceneCards": ParseActorSceneCards,
    "BuildActorSceneBatchPrompt": BuildActorSceneBatchPrompt,
    "BuildActorFinalPromptRequest": BuildActorFinalPromptRequest,
    "ActorSceneCardAtIndex": ActorSceneCardAtIndex,
    "SetActorCandidateFinalPrompt": SetActorCandidateFinalPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateStoryboardJob": "Create Portable Storyboard Job",
    "StoryboardDirectorySnapshot": "Snapshot Selected Storyboards",
    "LoadStoryboardJob": "Load Selected Storyboard Job",
    "UnpackStoryboardJob": "Unpack Storyboard Job",
    "SaveStoryboardOutput": "Save Storyboard / Full Render",
    "CreateActorCandidateJob": "Create Portable Actor Candidate",
    "CandidateDirectorySnapshot": "Snapshot Selected Actor Candidates",
    "LoadActorCandidateJob": "Load Selected Actor Candidate",
    "UnpackActorCandidateJob": "Unpack Actor Candidate",
    "SaveActorCandidateOutput": "Save Candidate / Final Render",
    "ParseActorSceneCards": "Parse Batched Actor Scene Cards",
    "BuildActorSceneBatchPrompt": "Build Batched Actor Scene Prompt",
    "BuildActorFinalPromptRequest": "Build Final Actor Prompt Request",
    "ActorSceneCardAtIndex": "Scene Card at Index",
    "SetActorCandidateFinalPrompt": "Record Final Actor Prompt",
}
