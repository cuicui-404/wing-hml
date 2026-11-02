from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import numpy as np

from .tasks import Task
from .target import read_targets,validate_splits
from .synthetic import generate_task
from .utils import file_hash,read_json


def audit(prepared,synthetic,target,splits,quality_only=False):
    result={}
    if (prepared/"manifest.json").exists():
        info=read_json(prepared/"manifest.json")
        seen,ids={},set()
        for record in info["tasks"]:
            if record["task_id"] in ids: raise ValueError("Duplicate source task ID")
            ids.add(record["task_id"])
            if "data_sha256" in record and file_hash(prepared/record["path"])!=record["data_sha256"]:
                raise ValueError(f"Source task checksum changed: {record['task_id']}")
            task=Task.load(prepared/record["path"])
            if not np.isfinite(task.y).all(): raise ValueError("Nonfinite source label")
            for group in np.unique(task.row_groups):
                previous=seen.setdefault(group,record["partition"])
                if previous!=record["partition"]: raise ValueError(f"Leaky source group {group}")
        result["real"]={"tasks":len(ids),"partitions":dict(Counter(r["partition"] for r in info["tasks"])),
                        "unique_raw_groups":len(seen),"group_leakage":False}
    if (synthetic/"manifest.json").exists():
        info=read_json(synthetic/"manifest.json")
        if file_hash(synthetic/"tasks.jsonl")!=info["spec_hash"]: raise ValueError("Synthetic spec file changed")
        seen,seeds,counts=set(),set(),Counter()
        with (synthetic/"tasks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row=json.loads(line)
                if row["task_id"] in seen: raise ValueError("Duplicate synthetic task ID")
                if row["spec"]["seed"] in seeds: raise ValueError("Synthetic task seed collision")
                seeds.add(row["spec"]["seed"])
                seen.add(row["task_id"])
                key=row["partition"]
                counts[key]+=1
                if counts[key]<=10:
                    task=generate_task(row["spec"])
                    assert len(task.x)==row["spec"]["n"]
                    assert task.x.shape[1]==row["spec"]["d"]
        result["synthetic"]={"tasks":len(seen),"partitions":dict(counts),"spec_hash_verified":True}
    if target.exists() and splits.exists():
        tables=read_targets(target,quality_only)
        manifest=read_json(splits)
        for table in tables.values(): validate_splits(manifest,table)
        result["target"]={"conditions":manifest["n_conditions"],"filters":list(tables),
                          "splits":len(manifest["splits"]),"outer_inner_leakage":False}
    if not result: raise ValueError("Nothing to audit: prepare/synthesize/splits first")
    return result
