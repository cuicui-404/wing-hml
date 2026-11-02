from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from .utils import digest, save_json

FEATURES = ["HWI","wind_speed_m_s","flapping_frequency_Hz"]
TARGETS = {"T":"corrected_mean_thrust_N","L":"corrected_mean_lift_N"}


@dataclass
class TargetTable:
    x: np.ndarray
    targets: dict
    ids: list[str]
    metadata: list[dict]
    filter_method: str

    @property
    def signature(self):
        return digest({"features":FEATURES,"x":self.x.tolist(),"ids":self.ids})

    @property
    def label_signature(self):
        return digest(self.targets)


def read_targets(path: Path,quality_only=False):
    with path.open(encoding="utf-8-sig",newline="") as f:
        reader = csv.DictReader(f)
        required = FEATURES+list(TARGETS.values())
        missing = set(required)-set(reader.fieldnames or [])
        if missing: raise ValueError(f"Missing target CSV columns: {missing}")
        rows = list(reader)
    groups = {}
    for row in rows:
        if quality_only and row.get("selection_status")!="ok": continue
        groups.setdefault(row.get("filter_method") or "default",[]).append(row)
    if not groups: raise ValueError("Target CSV has no rows after selection")
    tables = {}
    for method,rr in sorted(groups.items()):
        rr.sort(key=lambda r:tuple(float(r[k]) for k in FEATURES))
        x = np.asarray([[float(r[k]) for k in FEATURES] for r in rr])
        y = {k:np.asarray([float(r[v]) for r in rr]) for k,v in TARGETS.items()}
        ids = ["|".join(f"{value:.12g}" for value in row) for row in x]
        if len(set(ids))!=len(ids): raise ValueError(f"Duplicate target conditions in filter {method}")
        if not np.isfinite(x).all() or any(not np.isfinite(v).all() for v in y.values()):
            raise ValueError("Nonfinite target feature or label")
        angles = {float(r["angle_deg"]) for r in rr if r.get("angle_deg")}
        if len(angles)>1: raise ValueError("Angle varies but is not an input; define a new task explicitly")
        tables[method] = TargetTable(x,y,ids,rr,method)
    if len({t.signature for t in tables.values()})!=1:
        raise ValueError("Filters do not have identical conditions/quality selections; use separate manifests")
    return tables


def inner_split(train,seed):
    train = np.asarray(train)
    if len(train)<8: return []
    return [{"train":train[a].tolist(),"test":train[b].tolist()}
            for a,b in KFold(3,shuffle=True,random_state=seed).split(train)]


def make_splits(table,cfg):
    n = len(table.x)
    if n<4: raise ValueError("At least four target conditions are required")
    seed = int(cfg["split_seed"])
    splits=[]
    def add(protocol,label,train,test):
        train,test = [int(i) for i in train],[int(i) for i in test]
        if not train or not test: return
        index = len(splits)
        splits.append({"id":f"{protocol}/{label}","protocol":protocol,"train":train,"test":test,
                       "inner":inner_split(train,seed+index),"n_support":len(train)})
    for k in range(n):
        add("loo",str(k),[j for j in range(n) if j!=k],[k])
    for col,protocol in enumerate(["leave_hwi","leave_wind","leave_frequency"]):
        for v in np.unique(table.x[:,col]):
            add(protocol,f"{v:g}",np.flatnonzero(table.x[:,col]!=v),np.flatnonzero(table.x[:,col]==v))
    rng = np.random.default_rng(seed)
    for k in cfg["fewshot_sizes"]:
        if k>=n: continue
        for r in range(cfg["fewshot_repeats"]):
            ids = rng.permutation(n)
            add(f"fewshot_{k}",str(r),ids[:k],ids[k:])
    result = {"format":"target-splits-v1","features":FEATURES,"condition_signature":table.signature,
              "condition_ids":table.ids,"n_conditions":n,"split_seed":seed,"splits":splits}
    validate_splits(result,table)
    return result


def validate_splits(manifest,table):
    if manifest["condition_signature"]!=table.signature or manifest["condition_ids"]!=table.ids:
        raise ValueError("Target condition/order differs from split manifest")
    if len({s["id"] for s in manifest["splits"]})!=len(manifest["splits"]): raise ValueError("Duplicate split ids")
    universe=set(range(len(table.x)))
    loo_test=[]
    for s in manifest["splits"]:
        a,b=set(s["train"]),set(s["test"])
        if a&b or a|b!=universe or len(a)!=len(s["train"]) or len(b)!=len(s["test"]):
            raise ValueError(f"Invalid outer split: {s['id']}")
        inner_test=[]
        for inner in s["inner"]:
            i,j=set(inner["train"]),set(inner["test"])
            if i&j or i|j!=a: raise ValueError(f"Leaky inner split: {s['id']}")
            inner_test.extend(inner["test"])
        if s["inner"] and sorted(inner_test)!=sorted(a): raise ValueError("Invalid inner validation coverage")
        if s["protocol"]=="loo": loo_test.extend(s["test"])
    if sorted(loo_test)!=list(range(len(table.x))): raise ValueError("LOO does not cover each condition exactly once")
