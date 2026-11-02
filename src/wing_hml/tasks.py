from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import json
import warnings

import numpy as np

from .utils import digest,file_hash,read_json


@dataclass
class Task:
    task_id: str
    x: np.ndarray
    y: np.ndarray
    meta: dict = field(default_factory=dict)
    row_groups: np.ndarray | None = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        if self.x.ndim != 2 or self.y.ndim != 2 or len(self.x) != len(self.y):
            raise ValueError(f"Invalid shapes in task {self.task_id}")
        if not np.isfinite(self.y).all():
            raise ValueError(f"Nonfinite response in {self.task_id}")
        if self.row_groups is None:
            self.row_groups = np.asarray([f"{self.task_id}:{i}" for i in range(len(self.x))])
        self.row_groups = np.asarray(self.row_groups, dtype=str)
        if len(self.row_groups) != len(self.x):
            raise ValueError("row_groups length mismatch")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=self.x, y=self.y, row_groups=self.row_groups,
                            meta=json.dumps(self.meta, ensure_ascii=False), task_id=self.task_id)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as z:
            return cls(str(z["task_id"]), z["x"], z["y"], json.loads(str(z["meta"])), z["row_groups"])


class Corpus:
    """Immutable manifest plus bounded cache: 100k synthetic tables stay lazy."""
    def __init__(self, root: Path, partition: str):
        self.root = Path(root)
        self.info = read_json(self.root / "manifest.json")
        if self.info["format"] == "synthetic-v1":
            if file_hash(self.root/"tasks.jsonl")!=self.info["spec_hash"]:
                raise ValueError("Synthetic manifest checksum mismatch")
            self.records = []
            with (self.root / "tasks.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row["partition"] == partition:
                        self.records.append(row)
        else:
            if digest(self.info["tasks"])!=self.info["task_hash"]:
                raise ValueError("Real task manifest checksum mismatch")
            self.records = [r for r in self.info["tasks"] if r["partition"] == partition]
        self.partition = partition
        self.families = {}
        for i, record in enumerate(self.records):
            self.families.setdefault(record["family"], []).append(i)

    def __len__(self):
        return len(self.records)

    @lru_cache(maxsize=64)
    def get(self, index: int) -> Task:
        row = self.records[int(index)]
        if "spec" in row:
            from .synthetic import generate_task
            return generate_task(row["spec"])
        path=self.root/row["path"]
        if "data_sha256" in row and file_hash(path)!=row["data_sha256"]:
            raise ValueError(f"Task file checksum mismatch: {path}")
        return Task.load(path)

    def pick_index(self, rng: np.random.Generator, balanced: bool = False, excluded=None):
        if not self.records:
            raise ValueError(f"No tasks in {self.root} partition={self.partition}")
        excluded=excluded or set()
        if len(excluded)>=len(self.records): raise ValueError("No unused task remains in this batch")
        if balanced:
            used={}
            for i in excluded:
                family=self.records[i]["family"]
                used[family]=used.get(family,0)+1
            eligible=[f for f in sorted(self.families) if used.get(f,0)<len(self.families[f])]
            family = rng.choice(eligible)
            pool=self.families[family]
            if used.get(family,0)>len(pool)/2:
                pool=[i for i in pool if i not in excluded]
            index = int(rng.choice(pool))
            while index in excluded: index=int(rng.choice(pool))
        else:
            index = int(rng.integers(len(self)))
            while index in excluded: index=int(rng.integers(len(self)))
        return index

    def pick(self, rng: np.random.Generator, balanced: bool = False):
        return self.get(self.pick_index(rng,balanced))


@dataclass
class EpisodeScaler:
    """Fit on support only. An identical instance is used for neural and sklearn models."""
    eps: float = 1e-8

    def fit(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float).reshape(len(x), -1)
        self.keep = np.any(np.isfinite(x), axis=0)
        if not self.keep.any():
            raise ValueError("All support attributes are missing")
        x = np.where(np.isfinite(x[:, self.keep]), x[:, self.keep], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(x, axis=0)
        x = np.where(np.isfinite(x), x, self.median)
        self.xmean, self.xstd = x.mean(0), x.std(0)
        self.xstd = np.where(self.xstd < self.eps, 1.0, self.xstd)
        self.ymean, self.ystd = y.mean(0), y.std(0)
        self.ystd = np.where(self.ystd < self.eps, 1.0, self.ystd)
        return self

    def transform_x(self, x):
        x = np.asarray(x, float)[:, self.keep]
        return (np.where(np.isfinite(x), x, self.median) - self.xmean) / self.xstd

    def transform_y(self, y):
        return (np.asarray(y).reshape(-1, len(self.ymean)) - self.ymean) / self.ystd

    def inverse_y(self, y):
        return np.asarray(y).reshape(-1, len(self.ymean)) * self.ystd + self.ymean


@dataclass
class Episode:
    sx: np.ndarray
    sy: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    support_indices: np.ndarray
    query_indices: np.ndarray
    task_id: str


def sample_episode(task: Task, rng, support_sizes, query_sizes, gap: int = 1) -> Episode:
    n = len(task.x)
    temporal = bool(task.meta.get("temporal", False))
    gap = max(int(gap), int(task.meta.get("history_rows", 0))) if temporal else 0
    pairs = [(int(s), int(q)) for s in support_sizes for q in query_sizes if s + q + gap <= n]
    if pairs:
        ns, nq = pairs[int(rng.integers(len(pairs)))]
    else:
        available = n - gap
        if available < 3:
            raise ValueError(f"Task {task.task_id} is too small for disjoint support/query")
        ns = min(max(1, available // 2), min(support_sizes))
        nq = min(max(query_sizes), available - ns)
    if temporal:
        start = int(rng.integers(n - ns - nq - gap + 1))
        si = np.arange(start, start + ns)
        qi = np.arange(start + ns + gap, start + ns + gap + nq)
    else:
        order = rng.permutation(n)
        si, qi = order[:ns], order[ns:ns + nq]
    assert not np.intersect1d(si, qi).size
    scaler = EpisodeScaler().fit(task.x[si], task.y[si])
    return Episode(scaler.transform_x(task.x[si]), scaler.transform_y(task.y[si]),
                   scaler.transform_x(task.x[qi]), scaler.transform_y(task.y[qi]), si, qi, task.task_id)


def pack_episodes(episodes, device):
    import torch
    b = len(episodes)
    ns = max(len(e.sx) for e in episodes)
    nq = max(len(e.qx) for e in episodes)
    nf = max(e.sx.shape[1] for e in episodes)
    nr = max(e.sy.shape[1] for e in episodes)
    arrays = {"sx": np.zeros((b, ns, nf), np.float32), "sy": np.zeros((b, ns, nr), np.float32),
              "qx": np.zeros((b, nq, nf), np.float32), "qy": np.zeros((b, nq, nr), np.float32),
              "support_mask": np.zeros((b, ns), bool), "query_mask": np.zeros((b, nq), bool),
              "feature_mask": np.zeros((b, nf), bool), "response_mask": np.zeros((b, nr), bool)}
    for k,e in enumerate(episodes):
        s,q,i,j = len(e.sx),len(e.qx),e.sx.shape[1],e.sy.shape[1]
        arrays["sx"][k,:s,:i], arrays["sy"][k,:s,:j] = e.sx,e.sy
        arrays["qx"][k,:q,:i], arrays["qy"][k,:q,:j] = e.qx,e.qy
        arrays["support_mask"][k,:s] = True
        arrays["query_mask"][k,:q] = True
        arrays["feature_mask"][k,:i] = True
        arrays["response_mask"][k,:j] = True
    return {k: torch.as_tensor(v, device=device) for k,v in arrays.items()}
