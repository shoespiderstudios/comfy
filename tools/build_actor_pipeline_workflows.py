import copy
import json
import uuid
from pathlib import Path

import build_actor_candidate_workflow as base


WF_DIR = base.WF_DIR
OUTPUT = Path(r"C:\Users\mrtom\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
PIPELINE_ROOT = "actor-pipeline"
SELECTED = OUTPUT / PIPELINE_ROOT / "selected"


def stable_id(name):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"comfy-actor-pipeline:{name}"))


def story_node(node_type, node_id, title, pos, inputs, outputs, widgets=None, size=(390, 180)):
    node = base.custom_node(node_type, node_id, title, pos, inputs, outputs, widgets, size)
    node["properties"]["cnr_id"] = "comfyui-storyboard-jobs"
    return node


def workflow_document(name, graph, subgraphs, scale=0.22, offset=(230, 290)):
    document = copy.deepcopy(base.source_docs[0])
    document["id"] = stable_id(name)
    document["revision"] = 0
    document["last_node_id"] = max(node["id"] for node in graph.nodes)
    document["last_link_id"] = graph.next_link - 1
    document["nodes"] = graph.nodes
    document["links"] = graph.links
    document["groups"] = []
    document["definitions"] = {"subgraphs": subgraphs}
    document["config"] = {}
    document.setdefault("extra", {})["workflowRendererVersion"] = "LG"
    document["extra"]["ds"] = {"scale": scale, "offset": list(offset)}
    document["version"] = 0.4
    return document


def actor_analyzer_subgraph():
    graph_id = stable_id("actor-analyzer-4b")
    graph = base.Graph(object_links=True)
    scale_a = graph.add(base.clean_node("ImageScaleToMaxDimension", 101, "Actor A vision copy", (-420, -80), ["lanczos", 384]))
    scale_b = graph.add(base.clean_node("ImageScaleToMaxDimension", 102, "Actor B vision copy", (-420, 80), ["lanczos", 384]))
    scale_c = graph.add(base.clean_node("ImageScaleToMaxDimension", 103, "Actor C vision copy", (-420, 240), ["lanczos", 384]))
    stitch_ab = graph.add(base.clean_node("ImageStitch", 104, "A left, B middle", (-60, 20), ["right", True, 8, "white"]))
    stitch_c = graph.add(base.clean_node("ImageStitch", 105, "Add C right", (300, 80), ["right", True, 8, "white"]))
    sheet = graph.add(base.clean_node("ImageScaleToMaxDimension", 106, "Compact casting sheet", (650, 80), ["lanczos", 640]))
    clip = graph.add(base.clean_node("CLIPLoader", 107, "Qwen3-VL 4B Heretic", (-300, 500), ["qwen3-vl-4b-heretic_int8.safetensors", "krea2", "default"]))
    prompt_text = (
        "The image is a three-panel casting sheet: ACTOR A is left, ACTOR B is middle, ACTOR C is right. "
        "For each adult, record only stable visible identity traits useful to an image generator: apparent age range, "
        "face shape, skin tone, hair, build, and one distinctive feature. Ignore pose, clothes, background, and expression. "
        "Return exactly three short entries labelled ACTOR A:, ACTOR B:, ACTOR C:. No preface."
    )
    prompt = graph.add(base.clean_node("TextBox1", 108, "Compact actor inventory instruction", (650, 390), [prompt_text], (500, 230)))
    generate = graph.add(base.clean_from_template(base.text_generate_off_template, 109, "Analyze actors once", (1080, 170), ["", 180, "off", 0.1, 1], (470, 280)))
    trim = graph.add(base.clean_node("StringTrim", 110, "Trim actor inventory", (1600, 230), ["", "Both"]))
    for slot, target in enumerate((scale_a, scale_b, scale_c)):
        graph.connect(-10, slot, target["id"], "image", "IMAGE")
    graph.connect(scale_a["id"], 0, stitch_ab["id"], 0, "IMAGE")
    graph.connect(scale_b["id"], 0, stitch_ab["id"], 1, "IMAGE")
    graph.connect(stitch_ab["id"], 0, stitch_c["id"], 0, "IMAGE")
    graph.connect(scale_c["id"], 0, stitch_c["id"], 1, "IMAGE")
    graph.connect(stitch_c["id"], 0, sheet["id"], "image", "IMAGE")
    graph.connect(clip["id"], 0, generate["id"], "clip", "CLIP")
    graph.connect(sheet["id"], 0, generate["id"], "image", "IMAGE")
    graph.connect(prompt["id"], 0, generate["id"], "prompt", "STRING")
    graph.connect(generate["id"], 0, trim["id"], "string", "STRING")
    graph.connect(trim["id"], 0, -20, 0, "STRING")
    return graph_id, base.make_subgraph(
        graph_id,
        "Actor inventory - one compact 4B vision pass",
        [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE")],
        [("actor_inventory", "STRING")],
        graph,
    )


def vision_director_subgraph():
    """Inventory the actors and produce every scene card in one VLM invocation."""
    graph_id = stable_id("actor-vision-director-ultrafast")
    graph = base.Graph(object_links=True)
    scale_a = graph.add(base.clean_node("ImageScaleToMaxDimension", 101, "Actor A vision copy", (-420, -80), ["lanczos", 256]))
    scale_b = graph.add(base.clean_node("ImageScaleToMaxDimension", 102, "Actor B vision copy", (-420, 80), ["lanczos", 256]))
    scale_c = graph.add(base.clean_node("ImageScaleToMaxDimension", 103, "Actor C vision copy", (-420, 240), ["lanczos", 256]))
    stitch_ab = graph.add(base.clean_node("ImageStitch", 104, "A left, B middle", (-60, 20), ["right", True, 6, "white"]))
    stitch_c = graph.add(base.clean_node("ImageStitch", 105, "Add C right", (300, 80), ["right", True, 6, "white"]))
    sheet = graph.add(base.clean_node("ImageScaleToMaxDimension", 106, "Tiny casting sheet", (650, 80), ["lanczos", 448]))
    clip = graph.add(base.clean_node("CLIPLoader", 107, "Qwen3-VL 4B Heretic", (-300, 500), ["qwen3-vl-4b-heretic_int8.safetensors", "krea2", "default"]))
    build = graph.add(story_node(
        "BuildActorVisionSceneBatchPrompt", 108, "One-pass casting + direction request", (650, 370),
        [("direction", "STRING", False), ("count", "INT", False)], [("prompt", "STRING")], [], (500, 150)
    ))
    generate = graph.add(base.clean_node(
        "TextGenerate", 109, "Inventory and direct once", (1190, 100),
        ["", 520, "on", 0.9, 40, 0.9, 0.02, 0.5, 1, 0.1, False, True], (480, 350)
    ))
    parse = graph.add(story_node(
        "ParseActorVisionSceneBatch", 110, "Split inventory and scene cards", (1720, 130),
        [("raw_text", "STRING", False), ("requested_count", "INT", False)],
        [("actor_inventory", "STRING"), ("snapshot", "STRING"), ("count", "INT"), ("summary", "STRING")],
        [], (450, 150)
    ))
    for slot, target in enumerate((scale_a, scale_b, scale_c)):
        graph.connect(-10, slot, target["id"], "image", "IMAGE")
    graph.connect(scale_a["id"], 0, stitch_ab["id"], 0, "IMAGE")
    graph.connect(scale_b["id"], 0, stitch_ab["id"], 1, "IMAGE")
    graph.connect(stitch_ab["id"], 0, stitch_c["id"], 0, "IMAGE")
    graph.connect(scale_c["id"], 0, stitch_c["id"], 1, "IMAGE")
    graph.connect(stitch_c["id"], 0, sheet["id"], "image", "IMAGE")
    graph.connect(-10, 3, build["id"], "direction", "STRING")
    graph.connect(-10, 4, build["id"], "count", "INT")
    graph.connect(clip["id"], 0, generate["id"], "clip", "CLIP")
    graph.connect(sheet["id"], 0, generate["id"], "image", "IMAGE")
    graph.connect(build["id"], 0, generate["id"], "prompt", "STRING")
    graph.connect(-10, 5, generate["id"], "sampling_mode.seed", "INT")
    graph.connect(generate["id"], 0, parse["id"], "raw_text", "STRING")
    graph.connect(-10, 4, parse["id"], "requested_count", "INT")
    for slot in range(4):
        graph.connect(parse["id"], slot, -20, slot, ("STRING", "STRING", "INT", "STRING")[slot])
    graph.connect(generate["id"], 0, -20, 4, "STRING")
    return graph_id, base.make_subgraph(
        graph_id,
        "One-pass 4B vision director - casting inventory plus scene batch",
        [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"),
         ("direction", "STRING"), ("count", "INT"), ("seed", "INT")],
        [("actor_inventory", "STRING"), ("snapshot", "STRING"), ("count", "INT"), ("summary", "STRING"),
         ("raw_reply", "STRING")],
        graph,
        2250,
    )


def klein_sketch_subgraph(render_size=384, steps=4, variant="standard"):
    graph_id = stable_id(f"klein-sketch-{render_size}-{steps}-{variant}")
    graph = base.Graph(object_links=True)
    model = graph.add(base.clean_node("UNETLoader", 201, "Flux.2 Klein 4B FP8", (-450, 500), ["flux-2-klein-4b-fp8.safetensors", "default"]))
    clip = graph.add(base.clean_node("CLIPLoader", 202, "Qwen 3 4B Flux encoder", (-450, 650), ["qwen_3_4b.safetensors", "flux2", "default"]))
    vae = graph.add(base.clean_node("VAELoader", 203, "Flux.2 VAE", (-450, 800), ["flux2-vae.safetensors"]))
    encode = graph.add(base.clean_node("CLIPTextEncode", 204, "Encode short scene card", (0, 520), [""]))
    zero = graph.add(base.clean_node("ConditioningZeroOut", 205, "Zero negative", (300, 660)))
    graph.connect(encode["id"], 0, zero["id"], "conditioning", "CONDITIONING")
    positive, negative = encode, zero
    for index, input_slot in enumerate((0, 1, 2)):
        x = -300 + index * 350
        scale = graph.add(base.clean_node("ImageScaleToMaxDimension", 206 + index * 4, f"Actor {'ABC'[index]} max {render_size}", (x, -130), ["lanczos", render_size]))
        vae_encode = graph.add(base.clean_node("VAEEncode", 207 + index * 4, f"Encode actor {'ABC'[index]}", (x, 20)))
        pos_ref = graph.add(base.clean_node("ReferenceLatent", 208 + index * 4, f"Positive actor {'ABC'[index]}", (x, 150)))
        neg_ref = graph.add(base.clean_node("ReferenceLatent", 209 + index * 4, f"Negative actor {'ABC'[index]}", (x, 260)))
        graph.connect(-10, input_slot, scale["id"], "image", "IMAGE")
        graph.connect(scale["id"], 0, vae_encode["id"], "pixels", "IMAGE")
        graph.connect(vae["id"], 0, vae_encode["id"], "vae", "VAE")
        graph.connect(positive["id"], 0, pos_ref["id"], "conditioning", "CONDITIONING")
        graph.connect(negative["id"], 0, neg_ref["id"], "conditioning", "CONDITIONING")
        graph.connect(vae_encode["id"], 0, pos_ref["id"], "latent", "LATENT")
        graph.connect(vae_encode["id"], 0, neg_ref["id"], "latent", "LATENT")
        positive, negative = pos_ref, neg_ref
    scheduler = graph.add(base.clean_node("Flux2Scheduler", 218, f"{steps}-step {render_size}px schedule", (820, 590), [steps, render_size, render_size]))
    noise = graph.add(base.clean_node("RandomNoise", 219, "Sketch noise", (820, 740), [0, "fixed"]))
    sampler = graph.add(base.clean_node("KSamplerSelect", 220, "Euler", (820, 870), ["euler"]))
    guider = graph.add(base.clean_node("CFGGuider", 221, "Klein guidance", (1160, 500), [1.0]))
    latent = graph.add(base.clean_node("EmptyFlux2LatentImage", 222, f"{render_size} x {render_size} sketch latent", (1160, 760), [render_size, render_size, 1]))
    sample = graph.add(base.clean_node("SamplerCustomAdvanced", 223, f"{steps}-step sketch", (1480, 560)))
    decode = graph.add(base.clean_node("VAEDecode", 224, "Decode sketch", (1770, 560)))
    graph.connect(model["id"], 0, guider["id"], "model", "MODEL")
    graph.connect(positive["id"], 0, guider["id"], "positive", "CONDITIONING")
    graph.connect(negative["id"], 0, guider["id"], "negative", "CONDITIONING")
    graph.connect(clip["id"], 0, encode["id"], "clip", "CLIP")
    graph.connect(-10, 3, encode["id"], "text", "STRING")
    graph.connect(-10, 4, noise["id"], "noise_seed", "INT")
    graph.connect(noise["id"], 0, sample["id"], "noise", "NOISE")
    graph.connect(guider["id"], 0, sample["id"], "guider", "GUIDER")
    graph.connect(sampler["id"], 0, sample["id"], "sampler", "SAMPLER")
    graph.connect(scheduler["id"], 0, sample["id"], "sigmas", "SIGMAS")
    graph.connect(latent["id"], 0, sample["id"], "latent_image", "LATENT")
    graph.connect(sample["id"], 0, decode["id"], "samples", "LATENT")
    graph.connect(vae["id"], 0, decode["id"], "vae", "VAE")
    graph.connect(decode["id"], 0, -20, 0, "IMAGE")
    return graph_id, base.make_subgraph(
        graph_id,
        f"Cheap sketch - Flux.2 Klein 4B, {render_size}px, {steps} steps",
        [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("scene_prompt", "STRING"), ("seed", "INT")],
        [("sketch", "IMAGE")],
        graph,
    )


def prompt_refiner_subgraph():
    graph_id = stable_id("selected-prompt-refiner-4b")
    graph = base.Graph(object_links=True)
    clip = graph.add(base.clean_node("CLIPLoader", 301, "Qwen3-VL 4B Heretic - text only", (-400, 260), ["qwen3-vl-4b-heretic_int8.safetensors", "krea2", "default"]))
    build = graph.add(story_node(
        "BuildActorFinalPromptRequest", 302, "Build production-prompt request", (-80, 40),
        [("actor_inventory", "STRING", False), ("scene_prompt", "STRING", False)], [("prompt", "STRING")], [], (440, 130)
    ))
    generate = graph.add(base.clean_from_template(base.text_generate_off_template, 303, "Refine selected direction only", (430, 80), ["", 320, "off", 0.1, 1], (480, 280)))
    trim = graph.add(base.clean_node("StringTrim", 304, "Trim production prompt", (970, 150), ["", "Both"]))
    graph.connect(-10, 0, build["id"], "actor_inventory", "STRING")
    graph.connect(-10, 1, build["id"], "scene_prompt", "STRING")
    graph.connect(clip["id"], 0, generate["id"], "clip", "CLIP")
    graph.connect(build["id"], 0, generate["id"], "prompt", "STRING")
    graph.connect(generate["id"], 0, trim["id"], "string", "STRING")
    graph.connect(trim["id"], 0, -20, 0, "STRING")
    return graph_id, base.make_subgraph(
        graph_id,
        "Selected prompt refinement - concise 4B text pass",
        [("actor_inventory", "STRING"), ("scene_prompt", "STRING")],
        [("production_prompt", "STRING")],
        graph,
        1500,
    )


def miracle_renderer_subgraph(name, use_preview, render_size=1024, steps=26,
                              reference_size=768, clip_name="qwen_3_8b.safetensors"):
    graph_id = stable_id(name)
    graph = base.Graph(object_links=True)
    model = graph.add(base.clean_node("UNETLoader", 401, "MiracleIn 309B FP8", (-450, 550), ["miraclein309bFp8.aUKt.safetensors", "default"]))
    clip_size = "4B" if "4b" in clip_name.lower() else "8B"
    clip = graph.add(base.clean_node("CLIPLoader", 402, f"Qwen 3 {clip_size} Flux encoder", (-450, 700), [clip_name, "flux2", "default"]))
    vae = graph.add(base.clean_node("VAELoader", 403, "Flux.2 VAE", (-450, 850), ["flux2-vae.safetensors"]))
    encode = graph.add(base.clean_node("CLIPTextEncode", 404, "Encode final prompt", (0, 540), [""]))
    zero = graph.add(base.clean_node("ConditioningZeroOut", 405, "Zero negative", (300, 680)))
    graph.connect(encode["id"], 0, zero["id"], "conditioning", "CONDITIONING")
    positive, negative = encode, zero
    for index, input_slot in enumerate((0, 1, 2)):
        x = -300 + index * 360
        scale = graph.add(base.clean_node("ImageScaleToMaxDimension", 406 + index * 4, f"Actor {'ABC'[index]} max {reference_size}", (x, -130), ["lanczos", reference_size]))
        vae_encode = graph.add(base.clean_node("VAEEncode", 407 + index * 4, f"Encode actor {'ABC'[index]}", (x, 20)))
        pos_ref = graph.add(base.clean_node("ReferenceLatent", 408 + index * 4, f"Positive actor {'ABC'[index]}", (x, 150)))
        neg_ref = graph.add(base.clean_node("ReferenceLatent", 409 + index * 4, f"Negative actor {'ABC'[index]}", (x, 260)))
        graph.connect(-10, input_slot, scale["id"], "image", "IMAGE")
        graph.connect(scale["id"], 0, vae_encode["id"], "pixels", "IMAGE")
        graph.connect(vae["id"], 0, vae_encode["id"], "vae", "VAE")
        graph.connect(positive["id"], 0, pos_ref["id"], "conditioning", "CONDITIONING")
        graph.connect(negative["id"], 0, neg_ref["id"], "conditioning", "CONDITIONING")
        graph.connect(vae_encode["id"], 0, pos_ref["id"], "latent", "LATENT")
        graph.connect(vae_encode["id"], 0, neg_ref["id"], "latent", "LATENT")
        positive, negative = pos_ref, neg_ref
    scheduler = graph.add(base.clean_node("Flux2Scheduler", 418, f"{render_size}px {steps}-step schedule", (850, 610), [steps, render_size, render_size]))
    noise = graph.add(base.clean_node("RandomNoise", 419, "Final noise", (850, 760), [0, "fixed"]))
    sampler = graph.add(base.clean_node("KSamplerSelect", 420, "Euler", (850, 900), ["euler"]))
    guider = graph.add(base.clean_node("CFGGuider", 421, "MiracleIn guidance", (1220, 510), [1.1]))
    graph.connect(model["id"], 0, guider["id"], "model", "MODEL")
    graph.connect(positive["id"], 0, guider["id"], "positive", "CONDITIONING")
    graph.connect(negative["id"], 0, guider["id"], "negative", "CONDITIONING")
    graph.connect(clip["id"], 0, encode["id"], "clip", "CLIP")
    prompt_slot = 4 if use_preview else 3
    seed_slot = 5 if use_preview else 4
    graph.connect(-10, prompt_slot, encode["id"], "text", "STRING")
    graph.connect(-10, seed_slot, noise["id"], "noise_seed", "INT")
    if use_preview:
        preview_scale = graph.add(base.clean_node("ImageScaleToMaxDimension", 422, f"Selected sketch to {render_size}", (850, 40), ["lanczos", render_size]))
        latent = graph.add(base.clean_node("VAEEncode", 423, "Encode selected sketch", (1190, 60)))
        split = graph.add(story_node(
            "SplitSigmasDenoise", 424, "Partial-denoise schedule", (1190, 250),
            [("sigmas", "SIGMAS", False), ("denoise", "FLOAT", True)],
            [("high_sigmas", "SIGMAS"), ("low_sigmas", "SIGMAS")], [0.7], (330, 110)
        ))
        graph.connect(-10, 3, preview_scale["id"], "image", "IMAGE")
        graph.connect(preview_scale["id"], 0, latent["id"], "pixels", "IMAGE")
        graph.connect(vae["id"], 0, latent["id"], "vae", "VAE")
        graph.connect(scheduler["id"], 0, split["id"], "sigmas", "SIGMAS")
        graph.connect(-10, 6, split["id"], "denoise", "FLOAT")
        sigmas, sigmas_slot = split, 1
    else:
        latent = graph.add(base.clean_node("EmptyFlux2LatentImage", 422, f"Fresh {render_size} x {render_size} latent", (1210, 760), [render_size, render_size, 1]))
        sigmas, sigmas_slot = scheduler, 0
    sample = graph.add(base.clean_node("SamplerCustomAdvanced", 425, "Final sample", (1550, 580)))
    decode = graph.add(base.clean_node("VAEDecode", 426, "Decode final", (1830, 580)))
    graph.connect(noise["id"], 0, sample["id"], "noise", "NOISE")
    graph.connect(guider["id"], 0, sample["id"], "guider", "GUIDER")
    graph.connect(sampler["id"], 0, sample["id"], "sampler", "SAMPLER")
    graph.connect(sigmas["id"], sigmas_slot, sample["id"], "sigmas", "SIGMAS")
    graph.connect(latent["id"], 0, sample["id"], "latent_image", "LATENT")
    graph.connect(sample["id"], 0, decode["id"], "samples", "LATENT")
    graph.connect(vae["id"], 0, decode["id"], "vae", "VAE")
    graph.connect(decode["id"], 0, -20, 0, "IMAGE")
    inputs = [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE")]
    if use_preview:
        inputs.append(("selected_sketch", "IMAGE"))
    inputs.extend([("final_prompt", "STRING"), ("seed", "INT")])
    if use_preview:
        inputs.append(("denoise", "FLOAT"))
    label = "MiracleIn partial-denoise reconstruction" if use_preview else "MiracleIn fresh-noise render"
    return graph_id, base.make_subgraph(graph_id, label, inputs, [("image", "IMAGE")], graph, 2250)


def build_sketch_workflow():
    analyzer_id, analyzer_def = actor_analyzer_subgraph()
    sketch_id, sketch_def = klein_sketch_subgraph()
    graph = base.Graph(object_links=False)
    note = graph.add(base.clean_node("MarkdownNote", 1001, "How to use", (-1180, -360), [
        "# ACTOR IDEA + SKETCH BATCH\n\nLoads the three actors once, creates every short scene card in one 4B text pass, then switches to Flux.2 Klein for 384px four-step sketches.\n\nReview `output/actor-pipeline/inbox` and move approved PNGs to `output/actor-pipeline/selected`. Keep the matching files in `output/actor-pipeline/actors`.\n\nThe sketch is an idea/composition gate. Do not judge final faces, hands, or fine anatomy here."
    ], (690, 350)))
    actor_a = graph.add(base.clean_node("LoadImage", 1002, "Actor A", (-1180, 80), ["imagen2_00062_.png", "image"]))
    actor_b = graph.add(base.clean_node("LoadImage", 1003, "Actor B", (-1180, 390), ["imagen2_00062_.png", "image"]))
    actor_c = graph.add(base.clean_node("LoadImage", 1004, "Actor C", (-1180, 700), ["imagen2_00062_.png", "image"]))
    direction = graph.add(base.clean_node("TextBox1", 1005, "Scene boundaries", (-760, 80), [
        "Invent varied, realistic adult photographic situations. Make each composition visually decisive and substantially different. Preserve the three actors as distinct recognizable people."
    ], (560, 220)))
    count = graph.add(base.clean_node("PrimitiveInt", 1006, "Number of scene ideas", (-760, 350), [12, "fixed"]))
    director_seed = graph.add(base.clean_node("SeedNode", 1007, "One seed for the idea batch", (-760, 490), [123456789, "randomize"]))
    sketch_seed = graph.add(base.clean_node("SeedNode", 1008, "Sketch seed base", (-760, 630), [987654321, "randomize"]))
    analyzer = graph.add(base.subgraph_node(1009, analyzer_id, "Actor inventory - cached", (-100, 30), [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE")], [("actor_inventory", "STRING")], (440, 170)))
    build_prompt = graph.add(story_node(
        "BuildActorSceneBatchPrompt", 1010, "Build one batched director request", (390, 40),
        [("actor_inventory", "STRING", False), ("direction", "STRING", False), ("count", "INT", False)],
        [("prompt", "STRING")], [], (450, 150)
    ))
    director_clip = graph.add(base.clean_node("CLIPLoader", 1011, "Fast 4B text director", (390, 260), ["qwen3-vl-4b-heretic_int8.safetensors", "krea2", "default"]))
    director = graph.add(base.clean_node("TextGenerate", 1012, "Generate all short scene cards once", (890, 20), ["", 900, "on", 0.95, 80, 0.95, 0.02, 1.05, 1, 0.25, False, True], (500, 390)))
    parse = graph.add(story_node(
        "ParseActorSceneCards", 1013, "Parse scene-card batch", (1450, 30),
        [("raw_text", "STRING", False), ("requested_count", "INT", False)],
        [("snapshot", "STRING"), ("count", "INT"), ("summary", "STRING")], [], (430, 140)
    ))
    loop = graph.add(base.clean_node("easy forLoopStart", 1014, "Sketch batch loop", (1910, 30), [12]))
    card = graph.add(story_node(
        "ActorSceneCardAtIndex", 1015, "Current scene card", (2210, 50),
        [("snapshot", "STRING", False), ("index", "INT", False)], [("scene_prompt", "STRING")], [], (390, 110)
    ))
    seed = graph.add(base.clean_node("easy mathInt", 1016, "Sketch seed = base + index", (2210, 220), [0, 0, "add"]))
    sketch = graph.add(base.subgraph_node(1017, sketch_id, "Cheap 384px / four-step Klein sketch", (2650, 20), [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("scene_prompt", "STRING"), ("seed", "INT")], [("sketch", "IMAGE")], (480, 240)))
    settings_text = json.dumps({
        "pipeline": "actor-idea-sketch-batch",
        "director": {"model": "qwen3-vl-4b-heretic_int8", "thinking": False, "batched": True},
        "sketch": {"model": "flux-2-klein-4b-fp8", "width": 384, "height": 384, "steps": 4, "cfg": 1.0},
    }, indent=2)
    settings = graph.add(base.clean_node("TextBox1", 1018, "Portable sketch settings", (2660, 320), [settings_text], (470, 260)))
    create = graph.add(story_node(
        "CreateActorCandidateJob", 1019, "Create portable sketch candidate", (3190, 20),
        [("actor_a", "IMAGE", False), ("actor_b", "IMAGE", False), ("actor_c", "IMAGE", False),
         ("preview_image", "IMAGE", False), ("actor_inventory", "STRING", False), ("scene_prompt", "STRING", False),
         ("director_seed", "INT", False), ("preview_seed", "INT", False), ("settings_json", "STRING", True)],
        [("job", "ACTOR_CANDIDATE_JOB")], [settings_text], (480, 300)
    ))
    save = graph.add(story_node(
        "SaveActorCandidateOutput", 1020, "Save sketch candidate", (3740, 40),
        [("finalize_mode", "BOOLEAN", True), ("job", "ACTOR_CANDIDATE_JOB", False), ("image", "IMAGE", False),
         ("render_seed", "INT", False), ("candidate_root", "STRING", True)],
        [("image", "IMAGE"), ("saved_path", "STRING")], [False, PIPELINE_ROOT], (440, 220)
    ))
    loop_end = graph.add(base.clean_node("easy forLoopEnd", 1021, "Finish sketch batch", (4250, 190), []))
    preview = graph.add(base.clean_node("PreviewImage", 1022, "Last sketch", (4580, 150), []))
    path = graph.add(base.clean_node("PreviewAny", 1023, "Last saved path", (4250, 20), []))
    for actor, slot in zip((actor_a, actor_b, actor_c), range(3)):
        graph.connect(actor["id"], 0, analyzer["id"], slot, "IMAGE")
        graph.connect(actor["id"], 0, sketch["id"], slot, "IMAGE")
        graph.connect(actor["id"], 0, create["id"], slot, "IMAGE")
    graph.connect(analyzer["id"], 0, build_prompt["id"], "actor_inventory", "STRING")
    graph.connect(direction["id"], 0, build_prompt["id"], "direction", "STRING")
    graph.connect(count["id"], 0, build_prompt["id"], "count", "INT")
    graph.connect(director_clip["id"], 0, director["id"], "clip", "CLIP")
    graph.connect(build_prompt["id"], 0, director["id"], "prompt", "STRING")
    graph.connect(director_seed["id"], 0, director["id"], "sampling_mode.seed", "INT")
    graph.connect(director["id"], 0, parse["id"], "raw_text", "STRING")
    graph.connect(count["id"], 0, parse["id"], "requested_count", "INT")
    graph.connect(parse["id"], 1, loop["id"], 1, "INT")
    graph.connect(parse["id"], 0, card["id"], "snapshot", "STRING")
    graph.connect(loop["id"], 1, card["id"], "index", "INT")
    graph.connect(sketch_seed["id"], 0, seed["id"], 0, "INT")
    graph.connect(loop["id"], 1, seed["id"], 1, "INT")
    graph.connect(card["id"], 0, sketch["id"], 3, "STRING")
    graph.connect(seed["id"], 0, sketch["id"], 4, "INT")
    graph.connect(sketch["id"], 0, create["id"], "preview_image", "IMAGE")
    graph.connect(analyzer["id"], 0, create["id"], "actor_inventory", "STRING")
    graph.connect(card["id"], 0, create["id"], "scene_prompt", "STRING")
    graph.connect(director_seed["id"], 0, create["id"], "director_seed", "INT")
    graph.connect(seed["id"], 0, create["id"], "preview_seed", "INT")
    graph.connect(settings["id"], 0, create["id"], "settings_json", "STRING")
    graph.connect(create["id"], 0, save["id"], "job", "ACTOR_CANDIDATE_JOB")
    graph.connect(sketch["id"], 0, save["id"], "image", "IMAGE")
    graph.connect(seed["id"], 0, save["id"], "render_seed", "INT")
    graph.connect(save["id"], 1, path["id"], 0, "*")
    graph.connect(loop["id"], 0, loop_end["id"], 0, "FLOW_CONTROL")
    graph.connect(save["id"], 0, loop_end["id"], 1, "IMAGE")
    graph.connect(loop_end["id"], 0, preview["id"], 0, "IMAGE")
    return workflow_document("actor-idea-sketch-batch", graph, [analyzer_def, sketch_def], 0.2, (260, 300))


def build_ultrafast_sketch_workflow():
    director_id, director_def = vision_director_subgraph()
    sketch_id, sketch_def = klein_sketch_subgraph(256, 2, "ultrafast")
    graph = base.Graph(object_links=False)
    graph.add(base.clean_node("MarkdownNote", 3001, "How to use", (-1180, -360), [
        "# ACTOR IDEA + SKETCH — ULTRAFAST\n\nOne 4B vision-language pass inventories all three actors and writes the complete scene batch. Flux.2 Klein then makes intentionally rough 256px, two-step thumbnails.\n\nReview `output/actor-pipeline/inbox` and move promising PNGs to `output/actor-pipeline/selected`. Keep the shared files in `output/actor-pipeline/actors`.\n\nJudge only the premise, actor placement, action, framing, and broad coherence. Faces, hands, anatomy, texture, and fine identity are not reliable at this draft level."
    ], (700, 360)))
    actor_a = graph.add(base.clean_node("LoadImage", 3002, "Actor A", (-1180, 80), ["imagen2_00062_.png", "image"]))
    actor_b = graph.add(base.clean_node("LoadImage", 3003, "Actor B", (-1180, 390), ["imagen2_00062_.png", "image"]))
    actor_c = graph.add(base.clean_node("LoadImage", 3004, "Actor C", (-1180, 700), ["imagen2_00062_.png", "image"]))
    direction = graph.add(base.clean_node("TextBox1", 3005, "Scene boundaries", (-760, 80), [
        "Invent varied, realistic adult photographic situations. Make each composition visually decisive and substantially different. Preserve the three actors as distinct recognizable people."
    ], (560, 220)))
    count = graph.add(base.clean_node("PrimitiveInt", 3006, "Number of thumbnail ideas", (-760, 350), [12, "fixed"]))
    director_seed = graph.add(base.clean_node("SeedNode", 3007, "One seed for casting + ideas", (-760, 490), [135791357, "randomize"]))
    sketch_seed = graph.add(base.clean_node("SeedNode", 3008, "Thumbnail seed base", (-760, 630), [975319753, "randomize"]))
    director = graph.add(base.subgraph_node(
        3009, director_id, "One-pass actor-aware director", (-100, 30),
        [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"),
         ("direction", "STRING"), ("count", "INT"), ("seed", "INT")],
        [("actor_inventory", "STRING"), ("snapshot", "STRING"), ("count", "INT"), ("summary", "STRING"),
         ("raw_reply", "STRING")],
        (520, 230)
    ))
    loop = graph.add(base.clean_node("easy forLoopStart", 3010, "Thumbnail loop", (500, 40), [12]))
    card = graph.add(story_node(
        "ActorSceneCardAtIndex", 3011, "Current short scene card", (820, 50),
        [("snapshot", "STRING", False), ("index", "INT", False)], [("scene_prompt", "STRING")], [], (390, 110)
    ))
    seed = graph.add(base.clean_node("easy mathInt", 3012, "Thumbnail seed = base + index", (820, 220), [0, 0, "add"]))
    sketch = graph.add(base.subgraph_node(
        3013, sketch_id, "Cheapest 256px / two-step Klein sketch", (1260, 20),
        [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"),
         ("scene_prompt", "STRING"), ("seed", "INT")], [("sketch", "IMAGE")], (500, 240)
    ))
    settings_text = json.dumps({
        "pipeline": "actor-idea-sketch-ultrafast",
        "director": {
            "model": "qwen3-vl-4b-heretic_int8",
            "thinking": False,
            "combined_actor_inventory_and_scene_batch": True,
            "vision_sheet_max_dimension": 448,
        },
        "sketch": {
            "model": "flux-2-klein-4b-fp8",
            "width": 256,
            "height": 256,
            "steps": 2,
            "cfg": 1.0,
            "actor_reference_max_dimension": 256,
        },
    }, indent=2)
    settings = graph.add(base.clean_node("TextBox1", 3014, "Portable ultrafast settings", (1270, 320), [settings_text], (480, 290)))
    create = graph.add(story_node(
        "CreateActorCandidateJob", 3015, "Create portable thumbnail candidate", (1810, 20),
        [("actor_a", "IMAGE", False), ("actor_b", "IMAGE", False), ("actor_c", "IMAGE", False),
         ("preview_image", "IMAGE", False), ("actor_inventory", "STRING", False), ("scene_prompt", "STRING", False),
         ("director_seed", "INT", False), ("preview_seed", "INT", False), ("settings_json", "STRING", True)],
        [("job", "ACTOR_CANDIDATE_JOB")], [settings_text], (490, 300)
    ))
    save = graph.add(story_node(
        "SaveActorCandidateOutput", 3016, "Save ultrafast candidate", (2360, 40),
        [("finalize_mode", "BOOLEAN", True), ("job", "ACTOR_CANDIDATE_JOB", False), ("image", "IMAGE", False),
         ("render_seed", "INT", False), ("candidate_root", "STRING", True)],
        [("image", "IMAGE"), ("saved_path", "STRING")], [False, PIPELINE_ROOT], (440, 220)
    ))
    path = graph.add(base.clean_node("PreviewAny", 3017, "Last saved path", (2870, 20), []))
    loop_end = graph.add(base.clean_node("easy forLoopEnd", 3018, "Finish thumbnail batch", (2870, 190), []))
    preview = graph.add(base.clean_node("PreviewImage", 3019, "Last thumbnail", (3210, 150), []))
    raw_reply = graph.add(base.clean_node("PreviewAny", 3020, "Raw director reply", (500, 340), []))

    for actor, slot in zip((actor_a, actor_b, actor_c), range(3)):
        graph.connect(actor["id"], 0, director["id"], slot, "IMAGE")
        graph.connect(actor["id"], 0, sketch["id"], slot, "IMAGE")
        graph.connect(actor["id"], 0, create["id"], slot, "IMAGE")
    graph.connect(direction["id"], 0, director["id"], "direction", "STRING")
    graph.connect(count["id"], 0, director["id"], "count", "INT")
    graph.connect(director_seed["id"], 0, director["id"], "seed", "INT")
    graph.connect(director["id"], 4, raw_reply["id"], 0, "*")
    graph.connect(director["id"], 2, loop["id"], 1, "INT")
    graph.connect(director["id"], 1, card["id"], "snapshot", "STRING")
    graph.connect(loop["id"], 1, card["id"], "index", "INT")
    graph.connect(sketch_seed["id"], 0, seed["id"], 0, "INT")
    graph.connect(loop["id"], 1, seed["id"], 1, "INT")
    graph.connect(card["id"], 0, sketch["id"], "scene_prompt", "STRING")
    graph.connect(seed["id"], 0, sketch["id"], "seed", "INT")
    graph.connect(sketch["id"], 0, create["id"], "preview_image", "IMAGE")
    graph.connect(director["id"], 0, create["id"], "actor_inventory", "STRING")
    graph.connect(card["id"], 0, create["id"], "scene_prompt", "STRING")
    graph.connect(director_seed["id"], 0, create["id"], "director_seed", "INT")
    graph.connect(seed["id"], 0, create["id"], "preview_seed", "INT")
    graph.connect(settings["id"], 0, create["id"], "settings_json", "STRING")
    graph.connect(create["id"], 0, save["id"], "job", "ACTOR_CANDIDATE_JOB")
    graph.connect(sketch["id"], 0, save["id"], "image", "IMAGE")
    graph.connect(seed["id"], 0, save["id"], "render_seed", "INT")
    graph.connect(save["id"], 1, path["id"], 0, "*")
    graph.connect(loop["id"], 0, loop_end["id"], 0, "FLOW_CONTROL")
    graph.connect(save["id"], 0, loop_end["id"], 1, "IMAGE")
    graph.connect(loop_end["id"], 0, preview["id"], 0, "IMAGE")
    return workflow_document(
        "actor-idea-sketch-ultrafast", graph, [director_def, sketch_def], 0.22, (280, 300)
    )


def build_finalizer(strategy, refine_prompt, use_preview, denoise=None):
    renderer_options = {}
    if strategy == "direct-uplift":
        renderer_options = {
            "render_size": 832,
            "steps": 18,
            "reference_size": 512,
            # MiracleIn expects the 12288-wide conditioning produced by the
            # 8B Flux.2 encoder. The 4B encoder emits width 7680 and fails in
            # the transformer's txt_in projection at the first sampling step.
            "clip_name": "qwen_3_8b.safetensors",
        }
    renderer_id, renderer_def = miracle_renderer_subgraph(
        f"renderer-{strategy}", use_preview, **renderer_options
    )
    subgraphs = [renderer_def]
    refiner_id = None
    if refine_prompt:
        refiner_id, refiner_def = prompt_refiner_subgraph()
        subgraphs.insert(0, refiner_def)
    graph = base.Graph(object_links=False)
    descriptions = {
        "direct-uplift": "Uses the original short scene card and selected sketch as a starting latent at 0.65 denoise. The balanced 832px, 18-step configuration preserves the approved composition while substantially reducing final-render cost.",
        "fresh-render": "Refines the selected scene card, discards the sketch pixels, and renders from fresh 1024px noise. This maximizes redraw quality but permits composition drift.",
        "hybrid": "Refines the selected scene card and reconstructs from the selected sketch at 0.85 denoise. This keeps broad staging while allowing a much stronger redraw.",
    }
    note = graph.add(base.clean_node("MarkdownNote", 2001, "How to use", (-1100, -260), [
        f"# ACTOR FINALIZER - {strategy.upper()}\n\n{descriptions[strategy]}\n\nMove approved sketch PNGs into `output/actor-pipeline/selected`, then queue this workflow. It processes every selected portable candidate."
    ], (680, 300)))
    snapshot = graph.add(story_node(
        "CandidateDirectorySnapshot", 2002, "Selected sketch directory", (-1100, 100),
        [("directory", "STRING", True)], [("snapshot", "STRING"), ("count", "INT"), ("summary", "STRING")],
        [str(SELECTED)], (560, 130)
    ))
    seed_base = graph.add(base.clean_node("SeedNode", 2003, "Final render seed base", (-1100, 300), [246813579, "randomize"]))
    loop = graph.add(base.clean_node("easy forLoopStart", 2004, "Selected-candidate loop", (-430, 80), [1]))
    seed = graph.add(base.clean_node("easy mathInt", 2005, "Final seed = base + index", (-420, 260), [0, 0, "add"]))
    load = graph.add(story_node(
        "LoadActorCandidateJob", 2006, "Load selected candidate", (-80, 70),
        [("snapshot", "STRING", False), ("index", "INT", False)], [("job", "ACTOR_CANDIDATE_JOB")], [], (410, 110)
    ))
    unpack = graph.add(story_node(
        "UnpackActorCandidateJob", 2007, "Portable candidate context", (390, 20),
        [("job", "ACTOR_CANDIDATE_JOB", False)],
        [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("preview_image", "IMAGE"),
         ("actor_inventory", "STRING"), ("scene_prompt", "STRING"), ("candidate_id", "STRING"),
         ("director_seed", "INT"), ("preview_seed", "INT"), ("settings_json", "STRING")], [], (450, 310)
    ))
    x = 900
    if refine_prompt:
        refiner = graph.add(base.subgraph_node(2008, refiner_id, "Refine selected direction", (900, 20),
                                               [("actor_inventory", "STRING"), ("scene_prompt", "STRING")],
                                               [("production_prompt", "STRING")], (460, 150)))
        x = 1420
    else:
        refiner = None
    renderer_inputs = [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE")]
    if use_preview:
        renderer_inputs.append(("selected_sketch", "IMAGE"))
    renderer_inputs.extend([("final_prompt", "STRING"), ("seed", "INT")])
    if use_preview:
        renderer_inputs.append(("denoise", "FLOAT"))
    renderer_node_id = 2009
    renderer = graph.add(base.subgraph_node(renderer_node_id, renderer_id, f"MiracleIn final - {strategy}", (x, 20), renderer_inputs, [("image", "IMAGE")], (500, 270)))
    denoise_node = None
    if use_preview:
        denoise_node = graph.add(base.clean_node("PrimitiveFloat", 2010, "Reconstruction denoise", (x, 350), [denoise]))
    update = graph.add(story_node(
        "SetActorCandidateFinalPrompt", 2011, "Record final prompt and strategy", (x + 560, 40),
        [("job", "ACTOR_CANDIDATE_JOB", False), ("final_prompt", "STRING", False), ("strategy", "STRING", True)],
        [("job", "ACTOR_CANDIDATE_JOB")], [strategy], (450, 150)
    ))
    save_root = f"{PIPELINE_ROOT}/{strategy}"
    save = graph.add(story_node(
        "SaveActorCandidateOutput", 2012, f"Save {strategy} result", (x + 1070, 40),
        [("finalize_mode", "BOOLEAN", True), ("job", "ACTOR_CANDIDATE_JOB", False), ("image", "IMAGE", False),
         ("render_seed", "INT", False), ("candidate_root", "STRING", True)],
        [("image", "IMAGE"), ("saved_path", "STRING")], [True, save_root], (450, 220)
    ))
    path = graph.add(base.clean_node("PreviewAny", 2013, "Last saved path", (x + 1580, 40), []))
    loop_end = graph.add(base.clean_node("easy forLoopEnd", 2014, "Finish selected batch", (x + 1580, 260), []))
    preview = graph.add(base.clean_node("PreviewImage", 2015, "Last final image", (x + 1920, 210), []))
    graph.connect(snapshot["id"], 1, loop["id"], 1, "INT")
    graph.connect(seed_base["id"], 0, seed["id"], 0, "INT")
    graph.connect(loop["id"], 1, seed["id"], 1, "INT")
    graph.connect(snapshot["id"], 0, load["id"], "snapshot", "STRING")
    graph.connect(loop["id"], 1, load["id"], "index", "INT")
    graph.connect(load["id"], 0, unpack["id"], "job", "ACTOR_CANDIDATE_JOB")
    if refiner:
        graph.connect(unpack["id"], 4, refiner["id"], "actor_inventory", "STRING")
        graph.connect(unpack["id"], 5, refiner["id"], "scene_prompt", "STRING")
        prompt_source, prompt_slot = refiner, 0
    else:
        prompt_source, prompt_slot = unpack, 5
    graph.connect(unpack["id"], 0, renderer["id"], "actor_a", "IMAGE")
    graph.connect(unpack["id"], 1, renderer["id"], "actor_b", "IMAGE")
    graph.connect(unpack["id"], 2, renderer["id"], "actor_c", "IMAGE")
    if use_preview:
        graph.connect(unpack["id"], 3, renderer["id"], "selected_sketch", "IMAGE")
    graph.connect(prompt_source["id"], prompt_slot, renderer["id"], "final_prompt", "STRING")
    graph.connect(seed["id"], 0, renderer["id"], "seed", "INT")
    if use_preview:
        graph.connect(denoise_node["id"], 0, renderer["id"], "denoise", "FLOAT")
    graph.connect(load["id"], 0, update["id"], "job", "ACTOR_CANDIDATE_JOB")
    graph.connect(prompt_source["id"], prompt_slot, update["id"], "final_prompt", "STRING")
    graph.connect(update["id"], 0, save["id"], "job", "ACTOR_CANDIDATE_JOB")
    graph.connect(renderer["id"], 0, save["id"], "image", "IMAGE")
    graph.connect(seed["id"], 0, save["id"], "render_seed", "INT")
    graph.connect(save["id"], 1, path["id"], 0, "*")
    graph.connect(loop["id"], 0, loop_end["id"], 0, "FLOW_CONTROL")
    graph.connect(save["id"], 0, loop_end["id"], 1, "IMAGE")
    graph.connect(loop_end["id"], 0, preview["id"], 0, "IMAGE")
    return workflow_document(f"actor-finalize-{strategy}", graph, subgraphs, 0.22, (250, 280))


def main():
    workflows = {
        "actor-idea-sketch-batch.json": build_sketch_workflow(),
        "actor-idea-sketch-ultrafast.json": build_ultrafast_sketch_workflow(),
        "actor-finalize-direct-uplift.json": build_finalizer("direct-uplift", False, True, 0.65),
        "actor-finalize-fresh-render.json": build_finalizer("fresh-render", True, False),
        "actor-finalize-hybrid.json": build_finalizer("hybrid", True, True, 0.85),
    }
    for filename, workflow in workflows.items():
        path = WF_DIR / filename
        path.write_text(json.dumps(workflow, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{filename}: nodes={len(workflow['nodes'])}, links={len(workflow['links'])}, subgraphs={len(workflow['definitions']['subgraphs'])}, bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
