import json
from pathlib import Path


WF_DIR = Path(r"C:\Users\mrtom\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows")
NAMES = (
    "actor-idea-sketch-batch.json",
    "actor-idea-sketch-ultrafast.json",
    "actor-finalize-direct-uplift.json",
    "actor-finalize-fresh-render.json",
    "actor-finalize-hybrid.json",
    "actor-progression-escalation-loop.json",
    "actor-progression-escalation-loop-realistic.json",
)


def audit_graph(nodes, links, object_links):
    node_map = {node["id"]: node for node in nodes}
    assert len(node_map) == len(nodes), "duplicate node IDs"
    seen = set()
    for raw in links:
        if object_links:
            link_id = raw["id"]
            origin_id, origin_slot = raw["origin_id"], raw["origin_slot"]
            target_id, target_slot = raw["target_id"], raw["target_slot"]
        else:
            link_id, origin_id, origin_slot, target_id, target_slot, _ = raw
        assert link_id not in seen, f"duplicate link {link_id}"
        seen.add(link_id)
        if origin_id >= 0:
            output = node_map[origin_id]["outputs"][origin_slot]
            assert link_id in (output.get("links") or []), f"origin does not record link {link_id}"
        if target_id >= 0:
            input_value = node_map[target_id]["inputs"][target_slot]
            assert input_value.get("link") == link_id, f"target does not record link {link_id}"
    for node in nodes:
        for value in node.get("inputs", []):
            if value.get("link") is not None:
                assert value["link"] in seen, f"orphan input link {value['link']}"
        for value in node.get("outputs", []):
            for link_id in value.get("links") or []:
                assert link_id in seen, f"orphan output link {link_id}"


for name in NAMES:
    path = WF_DIR / "loops" / name if name.startswith("actor-progression-") else WF_DIR / name
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document.get("nodes"), list)
    assert isinstance(document.get("links"), list)
    audit_graph(document["nodes"], document["links"], False)
    local_ids = {node["id"] for node in document["nodes"]}
    all_node_ids = set(local_ids)
    for subgraph in document.get("definitions", {}).get("subgraphs", []):
        assert isinstance(subgraph.get("inputs"), list)
        assert isinstance(subgraph.get("outputs"), list)
        audit_graph(subgraph["nodes"], subgraph["links"], True)
        internal_ids = {node["id"] for node in subgraph["nodes"]}
        assert not (local_ids & internal_ids), "top-level/subgraph node ID collision"
        assert not (all_node_ids & internal_ids), "cross-subgraph node ID collision"
        all_node_ids.update(internal_ids)
    print(f"PASS {name}: nodes={len(document['nodes'])}, links={len(document['links'])}, subgraphs={len(document['definitions']['subgraphs'])}")
