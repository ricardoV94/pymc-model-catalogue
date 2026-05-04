"""Write experiment metadata JSON for the comparison dashboard."""

import json
import sys

import yaml

yaml_path, out_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(yaml_path))
meta = {
    "name": cfg["name"],
    "description": cfg.get("description", ""),
    "yaml_path": yaml_path,
}
if cfg.get("base"):
    meta["base"] = cfg["base"]
json.dump(meta, open(out_path, "w"))
