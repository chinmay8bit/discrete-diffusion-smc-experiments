import os
import json

def get_metadata(local_vars, ignore_internal=False):
    metadata = {}
    primitive_types = (int, float, bool, str)
    for var, val in local_vars.items():
        if ignore_internal and var.startswith("_"):
            continue
        if isinstance(val, primitive_types):
            metadata[var] = val
    return metadata

def save_metadata_json(metadata: dict, output_dir: str, filename: str = "metadata.json"):
    """
    Save a Python dict to a JSON file in the given output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to {out_path}")
