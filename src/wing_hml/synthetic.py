"""Auditable statistical task priors. These are not CFD or measured forces."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .tasks import Task
from .utils import digest, file_hash, jsonable, save_json, stable_seed


def make_spec(kind: str, seed: int, partition: str, index: int, cfg: dict, ood: bool = False):
    rng = np.random.default_rng(seed)
    low, high = int(cfg["min_rows"]), int(cfg["max_rows"])
    if rng.random() < 0.7 and max(low, 48) <= min(high, 96):
        n = int(rng.integers(max(low, 48), min(high, 96) + 1))
    else:
        n = int(rng.integers(low, high + 1))
    lo = max(3 if kind == "flapping" else 2, int(cfg["min_features"]))
    d = int(rng.integers(lo, int(cfg["max_features"]) + 1))
    spec = {"version": 1, "task_id": f"synthetic/{partition}/{kind}/{index:07d}",
            "family": kind, "seed": int(seed), "partition": partition, "n": n, "d": d,
            "ood": bool(ood), "row_seed": int(rng.integers(2**31)),
            "noise": float(rng.uniform(0.01, 0.10)), "permutation": rng.permutation(d).tolist()}
    if kind == "generic":
        # The entire DAG and each deterministic mechanism are saved per task.
        roots = int(rng.integers(2, 5))
        nodes = []
        for i in range(int(rng.integers(d + 3, d + 12))):
            parents = rng.choice(roots + i, size=min(int(rng.integers(1, 4)), roots + i), replace=False)
            ops = ["linear", "sin", "tanh", "square", "hinge", "product"]
            op = "step" if ood and i % 3 == 0 else str(rng.choice(ops))
            nodes.append({"parents": parents.tolist(), "weights": rng.normal(0, 0.7, len(parents)).tolist(),
                          "bias": float(rng.normal()), "op": op})
        # No target value is copied into an input column.
        target = roots + len(nodes) - 1
        feature_nodes = rng.choice(target, size=d, replace=False).tolist()
        spec.update(roots=roots, nodes=nodes, feature_nodes=feature_nodes, target_node=target,
                    root_distribution=str(rng.choice(["normal", "uniform", "mixed"])),
                    prototype_fraction=float(rng.choice([0, 0, 0.2, 0.5])),
                    warp=bool(rng.integers(2)), bins=int(rng.choice([0, 0, 4, 8])),
                    missing=float(rng.choice([0, 0, 0.02, 0.05])))
    elif kind == "airfoil":
        spec.update(target=str(rng.choice(["Cl", "Cdp", "Cm"])),
                    slope=float(rng.uniform(3, 7)), stall=float(rng.uniform(0.18, 0.4)),
                    camber=float(rng.uniform(-0.15, 0.4)), lag=float(rng.uniform(-3, 3)),
                    flap_gain=float(rng.uniform(0.2, 1.8)), drag=float(rng.uniform(0.01, 0.1)),
                    moment=float(rng.uniform(0.02, 0.3)))
    elif kind == "flapping":
        spec.update(target=str(rng.choice(["mean_T", "mean_L", "mean_M", "instant_force"])),
                    coefficients=rng.normal(0, 0.5, 12).tolist(),
                    radius=float(rng.uniform(0.08, 0.4)), rho=float(rng.uniform(0.8, 1.4)),
                    scale=float(rng.lognormal(-1, 0.8)), compliance_gain=float(rng.uniform(0.1, 1)))
    else:
        raise ValueError(f"Unknown synthetic family: {kind}")
    return spec


def generate_task(spec: dict) -> Task:
    rng = np.random.default_rng(spec["row_seed"])
    n,d = spec["n"],spec["d"]
    family = spec["family"]
    names = [f"x{i}" for i in range(d)]
    if family == "generic":
        roots = rng.normal(size=(n, spec["roots"]))
        if spec["root_distribution"] == "uniform":
            roots = rng.uniform(-2, 2, roots.shape)
        elif spec["root_distribution"] == "mixed":
            roots[:, ::2] = rng.uniform(-2, 2, roots[:, ::2].shape)
        if spec["prototype_fraction"]:
            m = max(2, int(n * spec["prototype_fraction"]))
            prototypes = roots[:m].copy()
            weights = rng.dirichlet(np.full(m, 0.15), size=n)
            roots = weights @ prototypes
        columns = [roots[:, i] for i in range(roots.shape[1])]
        for node in spec["nodes"]:
            values = np.column_stack([columns[i] for i in node["parents"]])
            z = values @ np.asarray(node["weights"]) + node["bias"]
            z = np.clip(z, -8, 8)
            op = node["op"]
            if op == "sin": z = np.sin(z)
            elif op == "tanh": z = np.tanh(z)
            elif op == "square": z = z**2 / 4
            elif op == "hinge": z = np.maximum(z, 0)
            elif op == "product": z = np.prod(np.tanh(values), axis=1)
            elif op == "step": z = (z > 0).astype(float) * 2 - 1
            z = z + rng.normal(0, 0.02, n)
            columns.append(z)
        x = np.column_stack([columns[i] for i in spec["feature_nodes"]])
        y = columns[spec["target_node"]].copy()
        if spec["warp"]:
            # Kumaraswamy CDF applied only to attributes.
            u = (np.tanh(x) + 1) / 2
            x = 1 - (1 - u**0.7)**1.8
        if spec["bins"]:
            edges = np.quantile(x[:, 0], np.linspace(0, 1, spec["bins"] + 1))
            x[:, 0] = np.digitize(x[:, 0], edges[1:-1])
        x[rng.random(x.shape) < spec["missing"]] = np.nan
    elif family == "airfoil":
        phase = rng.uniform(-np.pi, np.pi, n)
        amp = rng.uniform(0.02, 0.3, n)
        reduced_f = rng.uniform(0, 0.3 if not spec["ood"] else 0.6, n)
        alpha = rng.uniform(-0.25, 0.45, n) + amp * np.sin(phase)
        flap = rng.uniform(-0.3, 0.3, n)
        rate = reduced_f * amp * np.cos(phase)
        base = [alpha, phase, rate, flap, reduced_f, amp, rng.uniform(4, 6.5, n)]
        names = ["alpha_rad", "phase_rad", "normalized_alpha_rate", "flap_rad", "reduced_frequency",
                 "amplitude_rad", "log10_Re"]
        while len(base) < d:
            base.append(rng.normal(size=n)); names.append(f"irrelevant_{len(base)}")
        x = np.column_stack(base[:d]); names = names[:d]
        # Only observed attributes influence y. Unobserved latent coordinates
        # are fixed, rather than turning every omitted predictor into label noise.
        a = x[:, 0]
        p = x[:, 1]
        r = x[:, 2] if d > 2 else np.zeros(n)
        fl = x[:, 3] if d > 3 else np.zeros(n)
        lift = spec["camber"] + spec["slope"] * a / np.sqrt(1 + (a / spec["stall"])**4)
        lift += spec["lag"] * r + spec["flap_gain"] * fl
        if spec["ood"]:
            lift += 0.3 * np.sign(a) * (np.abs(a) > spec["stall"])
        else:
            lift += 0.05 * np.sin(p)
        if spec["target"] == "Cl": y = lift
        elif spec["target"] == "Cdp": y = spec["drag"] + 0.1 * lift**2 + 0.4 * np.maximum(abs(a) - spec["stall"], 0)
        else: y = -spec["moment"] * a + 0.1 * r - 0.1 * fl
    else:
        h = rng.uniform(10, 80, n)
        u = rng.uniform(0, 8 if spec["ood"] else 6, n)
        u[rng.random(n) < 0.2] = 0
        f = rng.uniform(0.5, 6 if spec["ood"] else 4, n)
        amp = rng.uniform(0.3, 1.2, n)
        compliance = rng.uniform(0, 1, n)
        phase = rng.uniform(-np.pi, np.pi, n)
        base = [h, u, f, amp, compliance, phase]
        names = ["geometry_ratio_100", "U", "frequency", "amplitude_rad", "compliance", "phase_rad"]
        while len(base) < d:
            base.append(rng.normal(size=n)); names.append(f"irrelevant_{len(base)}")
        x = np.column_stack(base[:d]); names = names[:d]
        amp = x[:, 3] if d > 3 else np.full(n, 0.7)
        comp = x[:, 4] if d > 4 else np.full(n, 0.5)
        phase = x[:, 5] if d > 5 else np.zeros(n)
        c = np.asarray(spec["coefficients"])
        g = (h - 40) / 30
        vwing = 2 * np.pi * f * spec["radius"] * amp
        q = 0.5 * spec["rho"] * (u**2 + vwing**2)
        profile = c[0] + c[1]*g + c[2]*g**2 + c[3]*u*f/12 + c[4]*comp
        profile += spec["compliance_gain"] * np.tanh(f/2) + c[5]*amp
        if spec["target"] == "mean_T": y = spec["scale"] * (q * profile - (0.05 + abs(c[6])) * u**2)
        elif spec["target"] == "mean_L": y = spec["scale"] * q * (profile + c[7]*f + c[8]*u)
        elif spec["target"] == "mean_M": y = spec["scale"] * q * spec["radius"] * (profile + c[9]*g*f)
        else: y = spec["scale"] * q * (profile * np.sin(phase) + c[10]*np.cos(2*phase))
    y = y + rng.normal(0, max(float(np.std(y)), 0.1) * spec["noise"], n)
    perm = spec["permutation"]
    return Task(spec["task_id"], x[:, perm], y[:, None],
                {"family": family, "partition": spec["partition"], "temporal": False,
                 "features": [names[i] for i in perm], "target": spec.get("target", "random_node"),
                 "units": "statistical_prior_not_measured_or_CFD", "ood": spec["ood"], "spec": spec})


def build_synthetic(out: Path, cfg: dict, seed: int, materialize: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Synthetic corpus already exists: {out}; choose a new directory")
    counts = cfg["counts"]
    if any(int(v) < 1 for v in counts.values()):
        raise ValueError("Every synthetic partition must contain at least one task")
    kinds = list(cfg["weights"])
    weights = np.asarray([cfg["weights"][k] for k in kinds], float)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid family weights")
    if cfg["min_rows"] < 4 or cfg["min_rows"] > cfg["max_rows"] or cfg["max_features"] < 3:
        raise ValueError("Invalid task size bounds")
    totals = {}
    with (out / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for partition, count in counts.items():
            count = int(count)
            exact = count * weights / weights.sum()
            allocation = np.floor(exact).astype(int)
            for i in np.argsort(-(exact - allocation))[:count - allocation.sum()]:
                allocation[i] += 1
            schedule = np.repeat(kinds, allocation)
            rng = np.random.default_rng(stable_seed(partition, seed))
            rng.shuffle(schedule)
            ood_count = int(round(count * cfg.get("ood_fraction", 0.5))) if partition == "test" else 0
            ood_ids = set(rng.choice(count, ood_count, replace=False).tolist())
            for i, kind in enumerate(schedule):
                spec = make_spec(str(kind), stable_seed(f"{partition}/{i}", seed), partition, i, cfg, i in ood_ids)
                row = {"task_id": spec["task_id"], "partition": partition, "family": str(kind), "spec": spec}
                if materialize:
                    rel = Path("tables") / (digest(spec)[:24] + ".npz")
                    generate_task(spec).save(out / rel)
                    row["materialized_path"] = str(rel)
                stream.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
            totals[partition] = {k: int(v) for k,v in zip(kinds, allocation)}
    info = {"format": "synthetic-v1", "seed": seed, "config": cfg, "counts": totals,
            "spec_hash": file_hash(out / "tasks.jsonl"), "units": "statistical_prior", "materialized": materialize}
    save_json(out / "manifest.json", info)
    return info
