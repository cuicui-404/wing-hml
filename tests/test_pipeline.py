import csv
from copy import deepcopy
import numpy as np
import pytest
import torch

from wing_hml.evaluation import evaluate_target,predict_independent
from wing_hml.synthetic import build_synthetic
from wing_hml.target import read_targets,make_splits
from wing_hml.training import train,load_checkpoint
from wing_hml.utils import PROJECT_ROOT,load_config,save_json
from test_data import write_target


@pytest.mark.parametrize("method",["hml","ds_maml","np_maml"])
def test_short_training_and_exact_resume(tmp_path,method):
    cfg=load_config(PROJECT_ROOT/"configs/smoke.yaml")
    cfg["device"]="cpu"
    cfg["training"].update(synthetic_steps=4,batch_size=2,validation_every=2,validation_episodes=2)
    synthetic=tmp_path/"synthetic"
    build_synthetic(synthetic,cfg["synthetic"],42)
    train(cfg,method,42,synthetic,None,tmp_path/"full",stage="synthetic")
    short=deepcopy(cfg); short["training"]["synthetic_steps"]=2
    train(short,method,42,synthetic,None,tmp_path/"resumed",stage="synthetic")
    train(cfg,method,42,synthetic,None,tmp_path/"resumed",stage="synthetic",resume=True)
    a=load_checkpoint(tmp_path/"full"/"last.pt")
    b=load_checkpoint(tmp_path/"resumed"/"last.pt")
    for k in a["model_state"]: torch.testing.assert_close(a["model_state"][k],b["model_state"][k],rtol=0,atol=0)


def test_target_complete_loo_and_resume_skip(tmp_path):
    cfg=load_config(PROJECT_ROOT/"configs/smoke.yaml")
    cfg["device"]="cpu"
    target=tmp_path/"target.csv"; write_target(target)
    table=next(iter(read_targets(target).values()))
    splits=tmp_path/"splits.json"; save_json(splits,make_splits(table,cfg["evaluation"]))
    result=evaluate_target(cfg,target,splits,["mean","rsm"],[42],tmp_path/"no_checkpoints",tmp_path/"results",["loo"],fold_stop=3)
    out=tmp_path/"results"/"default"
    assert not (out/"summary.csv").exists()  # incomplete predictions must not be ranked
    evaluate_target(cfg,target,splits,["mean","rsm"],[42],tmp_path/"no_checkpoints",tmp_path/"results",["loo"],resume=True)
    with (out/"metrics.csv").open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    assert len(rows)==4 and all(r["successful_predictions"]=="12" and r["status"]=="complete" for r in rows)
    before=(out/"predictions.jsonl").read_text()
    evaluate_target(cfg,target,splits,["mean","rsm"],[42],tmp_path/"no_checkpoints",tmp_path/"results",["loo"],resume=True)
    assert (out/"predictions.jsonl").read_text()==before


def test_independent_labels_never_change_fitting_or_selection(tmp_path):
    cfg=load_config(PROJECT_ROOT/"configs/smoke.yaml"); cfg["device"]="cpu"
    support=tmp_path/"support.csv"; write_target(support)
    query=tmp_path/"query.csv"
    query.write_text("HWI,wind_speed_m_s,flapping_frequency_Hz,angle_deg\n27.5,1,1.5,5\n42.5,3,1.5,5\n")
    first=predict_independent(cfg,support,query,["gpr","mean"],[42],tmp_path/"unused",tmp_path/"a")
    query.write_text("HWI,wind_speed_m_s,flapping_frequency_Hz,angle_deg,corrected_mean_thrust_N,corrected_mean_lift_N\n27.5,1,1.5,5,1000,-1000\n42.5,3,1.5,5,2000,-2000\n")
    second=predict_independent(cfg,support,query,["gpr","mean"],[42],tmp_path/"unused",tmp_path/"b")
    np.testing.assert_array_equal([r["predicted_N"] for r in first],[r["predicted_N"] for r in second])
    assert [r["best_params"] for r in first]==[r["best_params"] for r in second]
