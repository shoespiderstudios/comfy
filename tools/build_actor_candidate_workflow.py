import copy
import json
import uuid
from pathlib import Path


COMFY = Path(r"C:\Users\mrtom\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI")
WF_DIR = COMFY / "user" / "default" / "workflows"
REPO_WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
SOURCE_NAMES = [
    "imagen2-storyboard-or-full-render-loop.json",
    "imagen2-storyboard-and-full-render-paired-loop.json",
    "imagen2-actor-candidate-quality-gate.json",
]
SOURCES = [
    live_path if (live_path := WF_DIR / name).is_file() else REPO_WORKFLOWS / name
    for name in SOURCE_NAMES
]
DESTINATION = WF_DIR / "imagen2-actor-candidate-quality-gate.json"


source_docs = [json.loads(path.read_text(encoding="utf-8")) for path in SOURCES]
templates = {}
for document in source_docs:
    for node in document.get("nodes", []):
        templates.setdefault(node["type"], node)
    for subgraph in document.get("definitions", {}).get("subgraphs", []):
        for node in subgraph.get("nodes", []):
            templates.setdefault(node["type"], node)

text_generate_off_template = next(
    node
    for subgraph in source_docs[1]["definitions"]["subgraphs"]
    for node in subgraph.get("nodes", [])
    if node.get("type") == "TextGenerate" and node.get("title") == "Pass A — literal source description"
)


def clean_from_template(template, node_id, title, pos, widgets=None, size=None):
    node = copy.deepcopy(template)
    node["id"] = node_id
    node["title"] = title
    node["pos"] = list(pos)
    node["order"] = 0
    if widgets is not None:
        node["widgets_values"] = widgets
    if size is not None:
        node["size"] = list(size)
    for item in node.get("inputs", []):
        item["link"] = None
    for item in node.get("outputs", []):
        item["links"] = []
    return node


def clean_node(node_type, node_id, title, pos, widgets=None, size=None):
    return clean_from_template(templates[node_type], node_id, title, pos, widgets, size)


def custom_node(node_type, node_id, title, pos, inputs, outputs, widgets=None, size=(360, 180)):
    storyboard_node = (
        "Candidate" in node_type
        or node_type.startswith(("ActorScene", "BuildActor", "ParseActor", "SetActor"))
    )
    return {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"localized_name": name, "name": name, "type": kind, **({"widget": {"name": name}} if widget else {}), "link": None}
            for name, kind, widget in inputs
        ],
        "outputs": [
            {"localized_name": name, "name": name, "type": kind, "links": []}
            for name, kind in outputs
        ],
        "title": title,
        "properties": {
            "Node name for S&R": node_type,
            "cnr_id": "comfyui-storyboard-jobs" if storyboard_node else "comfy-core",
        },
        "widgets_values": widgets or [],
    }


class Graph:
    def __init__(self, object_links=False):
        self.nodes = []
        self.links = []
        self.next_link = 1
        self.object_links = object_links

    def add(self, node):
        node["order"] = len(self.nodes)
        self.nodes.append(node)
        return node

    def node(self, node_id):
        return next(node for node in self.nodes if node["id"] == node_id)

    @staticmethod
    def slot(items, value):
        if isinstance(value, int):
            return value
        return next(i for i, item in enumerate(items) if item.get("name") == value)

    def connect(self, origin_id, origin_slot, target_id, target_slot, kind):
        link_id = self.next_link
        self.next_link += 1
        if origin_id >= 0:
            origin = self.node(origin_id)
            origin_slot = self.slot(origin.get("outputs", []), origin_slot)
            links = origin["outputs"][origin_slot].get("links")
            if links is None:
                links = []
                origin["outputs"][origin_slot]["links"] = links
            links.append(link_id)
        if target_id >= 0:
            target = self.node(target_id)
            target_slot = self.slot(target.get("inputs", []), target_slot)
            target["inputs"][target_slot]["link"] = link_id
        if self.object_links:
            self.links.append({
                "id": link_id,
                "origin_id": origin_id,
                "origin_slot": origin_slot,
                "target_id": target_id,
                "target_slot": target_slot,
                "type": kind,
            })
        else:
            self.links.append([link_id, origin_id, origin_slot, target_id, target_slot, kind])
        return link_id


def subgraph_node(node_id, graph_id, title, pos, inputs, outputs, size=(440, 210)):
    return {
        "id": node_id,
        "type": graph_id,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": name, "type": kind, **({"widget": {"name": name}} if kind in {"STRING", "INT", "FLOAT", "BOOLEAN"} else {}), "link": None}
            for name, kind in inputs
        ],
        "outputs": [{"name": name, "localized_name": name, "type": kind, "links": []} for name, kind in outputs],
        "title": title,
        "properties": {"cnr_id": "comfy-core", "enableTabs": False, "previewExposures": []},
        "widgets_values": ["" if kind == "STRING" else False if kind == "BOOLEAN" else 0 for _, kind in inputs if kind in {"STRING", "INT", "FLOAT", "BOOLEAN"}],
    }


def make_subgraph(graph_id, name, inputs, outputs, graph, width=2200):
    input_defs = []
    for index, (field_name, kind) in enumerate(inputs):
        used = [link["id"] for link in graph.links if link["origin_id"] == -10 and link["origin_slot"] == index]
        input_defs.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{graph_id}:input:{index}:{field_name}:{kind}")),
            "name": field_name, "type": kind, "linkIds": used,
            "pos": [-520, 74 + index * 22],
        })
    output_defs = []
    for index, (field_name, kind) in enumerate(outputs):
        used = [link["id"] for link in graph.links if link["target_id"] == -20 and link["target_slot"] == index]
        output_defs.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{graph_id}:output:{index}:{field_name}:{kind}")),
            "name": field_name, "type": kind, "linkIds": used,
            "localized_name": field_name, "pos": [width - 120, 94 + index * 22],
        })
    return {
        "id": graph_id,
        "version": 1,
        "state": {"lastGroupId": 0, "lastNodeId": max(node["id"] for node in graph.nodes), "lastLinkId": graph.next_link - 1, "lastRerouteId": 0},
        "revision": 0,
        "config": {},
        "name": name,
        "inputNode": {"id": -10, "bounding": [-650, 50, 170, max(100, 55 + len(inputs) * 22)]},
        "outputNode": {"id": -20, "bounding": [width - 150, 70, 170, max(68, 45 + len(outputs) * 22)]},
        "inputs": input_defs,
        # Comfy's workflow schema always represents subgraph ports as arrays,
        # including the common case where a subgraph has only one output.
        "outputs": output_defs,
        "widgets": [],
        "nodes": graph.nodes,
        "groups": [],
        "links": graph.links,
        "extra": {"workflowRendererVersion": "LG"},
    }


def build_actor_analyzer(graph_id):
    g = Graph(object_links=True)
    scale_a = g.add(clean_node("ImageScaleToMaxDimension", 1, "Actor A vision copy", (-420, -80), ["lanczos", 512]))
    scale_b = g.add(clean_node("ImageScaleToMaxDimension", 2, "Actor B vision copy", (-420, 80), ["lanczos", 512]))
    scale_c = g.add(clean_node("ImageScaleToMaxDimension", 3, "Actor C vision copy", (-420, 240), ["lanczos", 512]))
    stitch_ab = g.add(clean_node("ImageStitch", 4, "Vision sheet — A left, B middle", (-60, 20), ["right", True, 12, "white"]))
    stitch_c = g.add(clean_node("ImageStitch", 5, "Vision sheet — add C right", (300, 80), ["right", True, 12, "white"]))
    final_scale = g.add(clean_node("ImageScaleToMaxDimension", 6, "Vision sheet max 768", (650, 80), ["lanczos", 768]))
    clip4 = g.add(clean_node("CLIPLoader", 7, "Qwen3-VL 4B Heretic", (-360, 480), ["qwen3-vl-4b-heretic_int8.safetensors", "krea2", "default"]))
    clip8 = g.add(clean_node("CLIPLoader", 8, "Qwen3-VL 8B Heretic", (-360, 640), ["qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors", "krea2", "default"]))
    switch = g.add(clean_node("LazySwitchKJ", 9, "Choose analyzer lazily", (80, 540), [False]))
    instruction = (
        "Study the three reference panels as an actor casting sheet: ACTOR A is left, ACTOR B is middle, "
        "and ACTOR C is right. Describe each actor independently using only stable, visible identity traits useful "
        "to an image generator: apparent adult age range, face shape, skin tone, hair, build, and distinctive visible "
        "features. Do not merge traits between panels. Ignore pose, clothing, background, and temporary expression except "
        "where needed to distinguish identities. Return exactly three compact labelled entries beginning ACTOR A:, ACTOR B:, "
        "and ACTOR C:. Do not invent a scene, mention policy, or add a preface."
    )
    prompt = g.add(clean_node("TextBox1", 10, "Actor inventory instruction", (650, 420), [instruction], (470, 250)))
    generate = g.add(clean_from_template(text_generate_off_template, 11, "Analyze actors once per source set", (1050, 180), ["", 256, "off", 0.1, 1], (470, 280)))
    trim = g.add(clean_node("StringTrim", 12, "Trim actor inventory", (1570, 250), ["", "Both"]))
    for slot, target in enumerate((scale_a, scale_b, scale_c)):
        g.connect(-10, slot, target["id"], "image", "IMAGE")
    g.connect(scale_a["id"], 0, stitch_ab["id"], 0, "IMAGE")
    g.connect(scale_b["id"], 0, stitch_ab["id"], 1, "IMAGE")
    g.connect(stitch_ab["id"], 0, stitch_c["id"], 0, "IMAGE")
    g.connect(scale_c["id"], 0, stitch_c["id"], 1, "IMAGE")
    g.connect(stitch_c["id"], 0, final_scale["id"], "image", "IMAGE")
    g.connect(clip4["id"], 0, switch["id"], 0, "CLIP")
    g.connect(clip8["id"], 0, switch["id"], 1, "CLIP")
    g.connect(-10, 3, switch["id"], 2, "BOOLEAN")
    g.connect(switch["id"], 0, generate["id"], "clip", "CLIP")
    g.connect(final_scale["id"], 0, generate["id"], "image", "IMAGE")
    g.connect(prompt["id"], 0, generate["id"], "prompt", "STRING")
    g.connect(generate["id"], 0, trim["id"], "string", "STRING")
    g.connect(trim["id"], 0, -20, 0, "STRING")
    return make_subgraph(graph_id, "Actor analyzer — independent A/B/C inventory", [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("use_8b", "BOOLEAN")], [("actor_inventory", "STRING")], g)


def build_scene_director(graph_id):
    g = Graph(object_links=True)
    clip4 = g.add(clean_node("CLIPLoader", 1, "Qwen3-VL 4B Heretic", (-420, 240), ["qwen3-vl-4b-heretic_int8.safetensors", "krea2", "default"]))
    clip8 = g.add(clean_node("CLIPLoader", 2, "Qwen3-VL 8B Heretic", (-420, 400), ["qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors", "krea2", "default"]))
    switch = g.add(clean_node("LazySwitchKJ", 3, "Choose director lazily", (20, 310), [False]))
    concat = g.add(clean_node("StringConcatenate", 4, "Actor inventory + direction", (250, 30), ["", "", "\n\nSCENE-DIRECTOR REQUIREMENTS:\n"], (420, 210)))
    guidance_text = (
        "The downstream renderer is MiracleIn, a high-capacity reference-conditioned Flux.2 model. Write a self-contained "
        "description of one final photograph, not an edit procedure. Assign ACTOR A, ACTOR B, and ACTOR C explicitly to "
        "distinct positions and visible actions; never blend their traits or leave body-part ownership ambiguous. Preserve "
        "their recognizable identities while changing pose, activity, environment, wardrobe, framing, and lighting as the "
        "user direction permits. Use one coherent action and physically plausible spatial relationships, natural joints, "
        "correct limb counts, clear contact points, and uncomplicated overlaps. State actor placement and pose first, then "
        "the environment, camera viewpoint, and lighting. Output only one concise final-scene paragraph with no analysis, "
        "heading, quotation marks, alternatives, source description, or preface."
    )
    guidance = g.add(clean_node("TextBox1", 5, "MiracleIn composition guidance", (250, 300), [guidance_text], (480, 300)))
    concat2 = g.add(clean_node("StringConcatenate", 6, "Add renderer constraints", (720, 120), ["", "", "\n\nRENDERER CONSTRAINTS:\n"], (420, 210)))
    generate = g.add(clean_node("TextGenerate", 7, "Direct one candidate scene", (1120, 110), ["", 240, "on", 0.9, 96, 0.96, 0.02, 1.05, 1, 0.35, True, True], (480, 390)))
    trim = g.add(clean_node("StringTrim", 8, "Trim exact scene prompt", (1640, 200), ["", "Both"]))
    g.connect(clip4["id"], 0, switch["id"], 0, "CLIP")
    g.connect(clip8["id"], 0, switch["id"], 1, "CLIP")
    g.connect(-10, 3, switch["id"], 2, "BOOLEAN")
    g.connect(-10, 0, concat["id"], "string_a", "STRING")
    g.connect(-10, 1, concat["id"], "string_b", "STRING")
    g.connect(concat["id"], 0, concat2["id"], "string_a", "STRING")
    g.connect(guidance["id"], 0, concat2["id"], "string_b", "STRING")
    g.connect(switch["id"], 0, generate["id"], "clip", "CLIP")
    g.connect(concat2["id"], 0, generate["id"], "prompt", "STRING")
    g.connect(-10, 2, generate["id"], 12, "INT")
    g.connect(generate["id"], 0, trim["id"], "string", "STRING")
    g.connect(trim["id"], 0, -20, 0, "STRING")
    return make_subgraph(graph_id, "Scene director — inventory to exact prompt", [("actor_inventory", "STRING"), ("director_instruction", "STRING"), ("seed", "INT"), ("use_8b", "BOOLEAN")], [("scene_prompt", "STRING")], g)


def renderer_common(g, prefix, actor_inputs, prompt_slot, seed_slot, size, steps, start_id=1, preview_input=None, denoise_input=None):
    nid = start_id
    model = g.add(clean_node("UNETLoader", nid, f"{prefix} — MiracleIn 309B FP8", (-420, 540), ["miraclein309bFp8.aUKt.safetensors", "default"])); nid += 1
    clip = g.add(clean_node("CLIPLoader", nid, f"{prefix} — Qwen 3 8B encoder", (-420, 700), ["qwen_3_8b.safetensors", "flux2", "default"])); nid += 1
    vae = g.add(clean_node("VAELoader", nid, f"{prefix} — Flux.2 VAE", (-420, 860), ["flux2-vae.safetensors"])); nid += 1
    encode_prompt = g.add(clean_node("CLIPTextEncode", nid, "Encode exact scene prompt", (0, 520), [""])); nid += 1
    zero = g.add(clean_node("ConditioningZeroOut", nid, "Zero negative", (320, 650))); nid += 1
    # Flux uses a zeroed copy of the prompt conditioning as its negative input.
    # ConditioningZeroOut still requires the source conditioning to be wired.
    g.connect(encode_prompt["id"], 0, zero["id"], "conditioning", "CONDITIONING")
    pos = encode_prompt
    neg = zero
    x = -260
    for index, input_slot in enumerate(actor_inputs):
        scale = g.add(clean_node("ImageScaleToMaxDimension", nid, f"Actor {'ABC'[index]} reference max 768", (-430, -100 + index * 170), ["lanczos", 768])); nid += 1
        vae_encode = g.add(clean_node("VAEEncode", nid, f"Encode Actor {'ABC'[index]}", (x, -100 + index * 170))); nid += 1
        pos_ref = g.add(clean_node("ReferenceLatent", nid, f"Positive Actor {'ABC'[index]}", (x + 300, -70 + index * 170))); nid += 1
        neg_ref = g.add(clean_node("ReferenceLatent", nid, f"Negative Actor {'ABC'[index]}", (x + 300, 10 + index * 170))); nid += 1
        g.connect(-10, input_slot, scale["id"], "image", "IMAGE")
        g.connect(scale["id"], 0, vae_encode["id"], "pixels", "IMAGE")
        g.connect(vae["id"], 0, vae_encode["id"], "vae", "VAE")
        g.connect(pos["id"], 0, pos_ref["id"], "conditioning", "CONDITIONING")
        g.connect(neg["id"], 0, neg_ref["id"], "conditioning", "CONDITIONING")
        g.connect(vae_encode["id"], 0, pos_ref["id"], "latent", "LATENT")
        g.connect(vae_encode["id"], 0, neg_ref["id"], "latent", "LATENT")
        pos, neg = pos_ref, neg_ref
        x += 360
    scheduler = g.add(clean_node("Flux2Scheduler", nid, f"{prefix} scheduler — {steps} steps", (850, 600), [steps, size, size])); nid += 1
    noise = g.add(clean_node("RandomNoise", nid, f"{prefix} noise", (850, 760), [0, "fixed"])); nid += 1
    sampler = g.add(clean_node("KSamplerSelect", nid, "Euler", (850, 900), ["euler"])); nid += 1
    guider = g.add(clean_node("CFGGuider", nid, f"{prefix} guidance", (1200, 500), [1.1])); nid += 1
    g.connect(model["id"], 0, guider["id"], "model", "MODEL")
    g.connect(pos["id"], 0, guider["id"], "positive", "CONDITIONING")
    g.connect(neg["id"], 0, guider["id"], "negative", "CONDITIONING")
    g.connect(clip["id"], 0, encode_prompt["id"], "clip", "CLIP")
    g.connect(-10, prompt_slot, encode_prompt["id"], "text", "STRING")
    g.connect(-10, seed_slot, noise["id"], "noise_seed", "INT")
    sigmas = scheduler
    if preview_input is None:
        latent = g.add(clean_node("EmptyFlux2LatentImage", nid, f"Locked {size} x {size} latent", (1200, 820), [size, size, 1])); nid += 1
    else:
        preview_scale = g.add(clean_node("ImageScaleToMaxDimension", nid, f"Selected preview to {size}", (830, 40), ["lanczos", size])); nid += 1
        latent = g.add(clean_node("VAEEncode", nid, "Encode selected preview as starting latent", (1180, 60))); nid += 1
        split = custom_node("SplitSigmasDenoise", nid, "Partial-denoise schedule", (1180, 250), [("sigmas", "SIGMAS", False), ("denoise", "FLOAT", True)], [("high_sigmas", "SIGMAS"), ("low_sigmas", "SIGMAS")], [0.4], (330, 110)); nid += 1
        g.add(split)
        g.connect(-10, preview_input, preview_scale["id"], "image", "IMAGE")
        g.connect(preview_scale["id"], 0, latent["id"], "pixels", "IMAGE")
        g.connect(vae["id"], 0, latent["id"], "vae", "VAE")
        g.connect(scheduler["id"], 0, split["id"], "sigmas", "SIGMAS")
        g.connect(-10, denoise_input, split["id"], "denoise", "FLOAT")
        sigmas = split
    sample = g.add(clean_node("SamplerCustomAdvanced", nid, f"{prefix} sample", (1530, 570))); nid += 1
    decode = g.add(clean_node("VAEDecode", nid, f"Decode {prefix.lower()}", (1810, 570))); nid += 1
    g.connect(noise["id"], 0, sample["id"], "noise", "NOISE")
    g.connect(guider["id"], 0, sample["id"], "guider", "GUIDER")
    g.connect(sampler["id"], 0, sample["id"], "sampler", "SAMPLER")
    g.connect(sigmas["id"], 1 if preview_input is not None else 0, sample["id"], "sigmas", "SIGMAS")
    g.connect(latent["id"], 0, sample["id"], "latent_image", "LATENT")
    g.connect(sample["id"], 0, decode["id"], "samples", "LATENT")
    g.connect(vae["id"], 0, decode["id"], "vae", "VAE")
    g.connect(decode["id"], 0, -20, 0, "IMAGE")


def build_preview_renderer(graph_id):
    g = Graph(object_links=True)
    renderer_common(g, "Preview", (0, 1, 2), 3, 4, 640, 12)
    return make_subgraph(graph_id, "MiracleIn preview — 640 square, 12 steps", [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("scene_prompt", "STRING"), ("seed", "INT")], [("preview", "IMAGE")], g, 2200)


def build_final_renderer(graph_id):
    g = Graph(object_links=True)
    renderer_common(g, "Final refinement", (0, 1, 2), 4, 5, 1024, 26, preview_input=3, denoise_input=6)
    return make_subgraph(graph_id, "MiracleIn refinement — 1024 square, 26 steps", [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("selected_preview", "IMAGE"), ("scene_prompt", "STRING"), ("seed", "INT"), ("denoise", "FLOAT")], [("final_render", "IMAGE")], g, 2250)


def build_workflow():
    ids = {name: str(uuid.uuid4()) for name in ("analyzer", "director", "preview", "final")}
    subgraphs = [
        build_actor_analyzer(ids["analyzer"]),
        build_scene_director(ids["director"]),
        build_preview_renderer(ids["preview"]),
        build_final_renderer(ids["final"]),
    ]
    g = Graph(object_links=False)
    note_text = (
        "# ACTOR CANDIDATE QUALITY GATE\n\n"
        "**MODE OFF — PREVIEW BATCH**: Actor A/B/C are analyzed independently, the director invents a scene, and MiracleIn makes a 640px / 12-step candidate. Portable candidates are saved to `output/candidates/inbox`.\n\n"
        "Move approved candidate PNGs to `output/candidates/selected`. Do not move or delete their matching files in `output/candidates/actors`.\n\n"
        "**MODE ON — FINALIZE SELECTED**: Every selected candidate is loaded with its exact prompt, inventory, seeds, preview, and three actor references. MiracleIn refines the approved preview at 1024px / 26 steps using 0.40 denoise. Results go to `output/candidates/final`.\n\n"
        "The inactive branch is lazy, so only the models needed by the chosen mode execute. Actor analysis is cached for unchanged source images."
    )
    note = g.add(clean_node("MarkdownNote", 1, "How to use", (-1200, -420), [note_text], (700, 430)))
    mode = g.add(clean_node("PrimitiveBoolean", 2, "MODE — OFF=PREVIEW BATCH / ON=FINALIZE SELECTED", (-1180, 80), [False]))
    dreamer = g.add(clean_node("PrimitiveBoolean", 3, "VLM — OFF=4B / ON=8B", (-1180, 220), [False]))
    actor_a = g.add(clean_node("LoadImage", 4, "PREVIEW — Actor A", (-1180, 400), ["imagen2_00062_.png", "image"]))
    actor_b = g.add(clean_node("LoadImage", 5, "PREVIEW — Actor B", (-1180, 700), ["imagen2_00062_.png", "image"]))
    actor_c = g.add(clean_node("LoadImage", 6, "PREVIEW — Actor C", (-1180, 1000), ["imagen2_00062_.png", "image"]))
    instruction_text = (
        "Invent one bold, visually specific adult scene using ACTOR A, ACTOR B, and ACTOR C as distinct people. "
        "Preserve each actor's recognizable identity. You may substantially change pose, action, environment, wardrobe, "
        "camera framing, lighting, and physical arrangement. Make the result achievable in a single photograph and avoid "
        "fused bodies, unclear limb ownership, tangled poses, impossible joints, extreme foreshortening, background people, "
        "reflections, animals, fantasy, science fiction, and cartoon elements."
    )
    instruction = g.add(clean_node("TextBox1", 7, "PREVIEW — scene direction", (-760, 80), [instruction_text], (600, 330)))
    count = g.add(clean_node("PrimitiveInt", 8, "PREVIEW — number of candidates", (-760, 460), [20, "fixed"]))
    director_base = g.add(clean_node("SeedNode", 9, "PREVIEW — randomized director base seed", (-760, 590), [123456789, "randomize"]))
    render_base = g.add(clean_node("SeedNode", 10, "Image seed base — randomized each queue", (-760, 720), [987654321, "randomize"]))
    snapshot = g.add(custom_node("CandidateDirectorySnapshot", 11, "FINAL — selected candidate directory", (-760, 910), [("directory", "STRING", True)], [("snapshot", "STRING"), ("count", "INT"), ("summary", "STRING")], [str(Path(r"C:\Users\mrtom\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\candidates\selected"))], (520, 130)))
    count_switch = g.add(clean_node("LazySwitchKJ", 12, "Mode-aware loop count (lazy)", (-80, 50), [False]))
    loop = g.add(clean_node("easy forLoopStart", 13, "Candidate/final batch loop", (300, 40), [20]))
    director_seed = g.add(clean_node("easy mathInt", 14, "Director seed = base + index", (300, 230), [0, 0, "add"]))
    image_seed = g.add(clean_node("easy mathInt", 15, "Image seed = base + index", (300, 370), [0, 0, "add"]))
    analyzer = g.add(subgraph_node(16, ids["analyzer"], "SUBGRAPH — Actor analyzer", (720, 40), [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("use_8b", "BOOLEAN")], [("actor_inventory", "STRING")], (440, 190)))
    director = g.add(subgraph_node(17, ids["director"], "SUBGRAPH — Scene director", (1210, 40), [("actor_inventory", "STRING"), ("director_instruction", "STRING"), ("seed", "INT"), ("use_8b", "BOOLEAN")], [("scene_prompt", "STRING")], (440, 210)))
    preview = g.add(subgraph_node(18, ids["preview"], "SUBGRAPH — Cheap MiracleIn preview", (1710, 40), [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("scene_prompt", "STRING"), ("seed", "INT")], [("preview", "IMAGE")], (440, 230)))
    settings = g.add(clean_node("TextBox1", 19, "Portable candidate settings", (1710, 330), [json.dumps({"preview": {"width": 640, "height": 640, "steps": 12, "sampler": "euler", "cfg": 1.1}, "final": {"width": 1024, "height": 1024, "steps": 26, "sampler": "euler", "cfg": 1.1, "denoise": 0.4}}, indent=2)], (440, 280)))
    create = g.add(custom_node("CreateActorCandidateJob", 20, "Create portable actor candidate", (2210, 40), [("actor_a", "IMAGE", False), ("actor_b", "IMAGE", False), ("actor_c", "IMAGE", False), ("preview_image", "IMAGE", False), ("actor_inventory", "STRING", False), ("scene_prompt", "STRING", False), ("director_seed", "INT", False), ("preview_seed", "INT", False), ("settings_json", "STRING", True)], [("job", "ACTOR_CANDIDATE_JOB")], ["{}"], (470, 300)))
    load = g.add(custom_node("LoadActorCandidateJob", 21, "FINAL — load selected candidate", (720, 520), [("snapshot", "STRING", False), ("index", "INT", False)], [("job", "ACTOR_CANDIDATE_JOB")], [], (410, 110)))
    job_switch = g.add(clean_node("LazySwitchKJ", 22, "MODE ROUTER — new / selected job", (2740, 160), [False]))
    unpack = g.add(custom_node("UnpackActorCandidateJob", 23, "Portable candidate context", (3110, 60), [("job", "ACTOR_CANDIDATE_JOB", False)], [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("preview_image", "IMAGE"), ("actor_inventory", "STRING"), ("scene_prompt", "STRING"), ("candidate_id", "STRING"), ("director_seed", "INT"), ("preview_seed", "INT"), ("settings_json", "STRING")], [], (430, 310)))
    denoise = g.add(clean_node("PrimitiveFloat", 24, "FINAL — refinement denoise", (3120, 430), [0.4])) if "PrimitiveFloat" in templates else g.add(custom_node("PrimitiveFloat", 24, "FINAL — refinement denoise", (3120, 430), [("value", "FLOAT", True)], [("FLOAT", "FLOAT")], [0.4], (300, 100)))
    final = g.add(subgraph_node(25, ids["final"], "SUBGRAPH — Final partial-denoise refinement", (3600, 40), [("actor_a", "IMAGE"), ("actor_b", "IMAGE"), ("actor_c", "IMAGE"), ("selected_preview", "IMAGE"), ("scene_prompt", "STRING"), ("seed", "INT"), ("denoise", "FLOAT")], [("final_render", "IMAGE")], (470, 270)))
    image_switch = g.add(clean_node("LazySwitchKJ", 26, "MODE ROUTER — preview / final image", (4140, 170), [False]))
    save = g.add(custom_node("SaveActorCandidateOutput", 27, "Save candidate / final render", (4510, 80), [("job", "ACTOR_CANDIDATE_JOB", False), ("image", "IMAGE", False), ("render_seed", "INT", False), ("finalize_mode", "BOOLEAN", True), ("candidate_root", "STRING", True)], [("image", "IMAGE"), ("saved_path", "STRING")], [False, "candidates"], (440, 220)))
    path_preview = g.add(clean_node("PreviewAny", 28, "Last saved path", (5000, 80), []))
    loop_end = g.add(clean_node("easy forLoopEnd", 29, "Finish batch", (5010, 300), []))
    image_preview = g.add(clean_node("PreviewImage", 30, "Batch complete — last image", (5330, 260), []))

    g.connect(mode["id"], 0, count_switch["id"], 2, "BOOLEAN")
    g.connect(count["id"], 0, count_switch["id"], 0, "INT")
    g.connect(snapshot["id"], 1, count_switch["id"], 1, "INT")
    g.connect(count_switch["id"], 0, loop["id"], 1, "INT")
    g.connect(director_base["id"], 0, director_seed["id"], 0, "INT")
    g.connect(loop["id"], 1, director_seed["id"], 1, "INT")
    g.connect(render_base["id"], 0, image_seed["id"], 0, "INT")
    g.connect(loop["id"], 1, image_seed["id"], 1, "INT")
    for actor, slot in zip((actor_a, actor_b, actor_c), range(3)):
        g.connect(actor["id"], 0, analyzer["id"], slot, "IMAGE")
        g.connect(actor["id"], 0, preview["id"], slot, "IMAGE")
        g.connect(actor["id"], 0, create["id"], slot, "IMAGE")
    g.connect(dreamer["id"], 0, analyzer["id"], 3, "BOOLEAN")
    g.connect(analyzer["id"], 0, director["id"], 0, "STRING")
    g.connect(instruction["id"], 0, director["id"], 1, "STRING")
    g.connect(director_seed["id"], 0, director["id"], 2, "INT")
    g.connect(dreamer["id"], 0, director["id"], 3, "BOOLEAN")
    g.connect(director["id"], 0, preview["id"], 3, "STRING")
    g.connect(image_seed["id"], 0, preview["id"], 4, "INT")
    g.connect(preview["id"], 0, create["id"], 3, "IMAGE")
    g.connect(analyzer["id"], 0, create["id"], 4, "STRING")
    g.connect(director["id"], 0, create["id"], 5, "STRING")
    g.connect(director_seed["id"], 0, create["id"], 6, "INT")
    g.connect(image_seed["id"], 0, create["id"], 7, "INT")
    g.connect(settings["id"], 0, create["id"], 8, "STRING")
    g.connect(snapshot["id"], 0, load["id"], 0, "STRING")
    g.connect(loop["id"], 1, load["id"], 1, "INT")
    g.connect(create["id"], 0, job_switch["id"], 0, "ACTOR_CANDIDATE_JOB")
    g.connect(load["id"], 0, job_switch["id"], 1, "ACTOR_CANDIDATE_JOB")
    g.connect(mode["id"], 0, job_switch["id"], 2, "BOOLEAN")
    g.connect(job_switch["id"], 0, unpack["id"], 0, "ACTOR_CANDIDATE_JOB")
    for source_slot, target_slot in zip((0, 1, 2, 3), (0, 1, 2, 3)):
        g.connect(unpack["id"], source_slot, final["id"], target_slot, "IMAGE")
    g.connect(unpack["id"], 5, final["id"], 4, "STRING")
    g.connect(image_seed["id"], 0, final["id"], 5, "INT")
    g.connect(denoise["id"], 0, final["id"], 6, "FLOAT")
    g.connect(unpack["id"], 3, image_switch["id"], 0, "IMAGE")
    g.connect(final["id"], 0, image_switch["id"], 1, "IMAGE")
    g.connect(mode["id"], 0, image_switch["id"], 2, "BOOLEAN")
    g.connect(job_switch["id"], 0, save["id"], 0, "ACTOR_CANDIDATE_JOB")
    g.connect(image_switch["id"], 0, save["id"], 1, "IMAGE")
    g.connect(image_seed["id"], 0, save["id"], 2, "INT")
    g.connect(mode["id"], 0, save["id"], 3, "BOOLEAN")
    g.connect(save["id"], 1, path_preview["id"], 0, "*")
    g.connect(loop["id"], 0, loop_end["id"], 0, "FLOW_CONTROL")
    g.connect(save["id"], 0, loop_end["id"], 1, "IMAGE")
    g.connect(loop_end["id"], 0, image_preview["id"], 0, "IMAGE")

    base = copy.deepcopy(source_docs[0])
    base["id"] = str(uuid.uuid4())
    base["revision"] = 0
    base["last_node_id"] = max(node["id"] for node in g.nodes)
    base["last_link_id"] = g.next_link - 1
    base["nodes"] = g.nodes
    base["links"] = g.links
    base["groups"] = []
    base["definitions"] = {"subgraphs": subgraphs}
    base.setdefault("extra", {})["workflowRendererVersion"] = "LG"
    base["extra"]["ds"] = {"scale": 0.2, "offset": [280, 300]}
    return base


if __name__ == "__main__":
    workflow = build_workflow()
    DESTINATION.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(DESTINATION)
    print(f"nodes={len(workflow['nodes'])} links={len(workflow['links'])} subgraphs={len(workflow['definitions']['subgraphs'])}")
