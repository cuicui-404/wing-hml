from pathlib import Path
import csv
import numpy as np
import pytest

from wing_hml.synthetic import build_synthetic,generate_task,make_spec
from wing_hml.tasks import Corpus,EpisodeScaler,Task,sample_episode
from wing_hml.target import make_splits,read_targets,validate_splits
from wing_hml.prepare import prepare_sources
from wing_hml.utils import load_config,PROJECT_ROOT


def test_scaler_support_only_and_temporal_gap():
    sx=np.array([[1.,np.nan,5],[3.,np.nan,5],[5.,np.nan,5]])
    scaler=EpisodeScaler().fit(sx,np.array([1.,2.,3.]))
    assert scaler.keep.tolist()==[True,False,True]
    mean=scaler.xmean.copy()
    scaler.transform_x([[1e9,0,9]])
    np.testing.assert_array_equal(scaler.xmean,mean)
    np.testing.assert_allclose(scaler.inverse_y(scaler.transform_y([1,2,3])).ravel(),[1,2,3])
    task=Task("time",np.arange(100).reshape(-1,1),np.arange(100),{"temporal":True,"history_rows":4})
    e=sample_episode(task,np.random.default_rng(42),[16],[8],gap=1)
    assert e.query_indices.min()-e.support_indices.max()>4


def test_synthetic_determinism_partitioning_and_ood(tmp_path):
    cfg=load_config(PROJECT_ROOT/"configs/smoke.yaml")["synthetic"]
    build_synthetic(tmp_path,cfg,42)
    for split in ["train","val","test"]:
        corpus=Corpus(tmp_path,split)
        assert len(corpus)==cfg["counts"][split]
        for i in range(len(corpus)):
            task=corpus.get(i)
            same=generate_task(corpus.records[i]["spec"])
            np.testing.assert_allclose(task.x,same.x,equal_nan=True)
            np.testing.assert_allclose(task.y,same.y)
            assert task.x.shape[1]>=2
    assert sum(r["spec"]["ood"] for r in Corpus(tmp_path,"test").records)==6
    c=Corpus(tmp_path,"train"); rng=np.random.default_rng(9); seen=set()
    for _ in range(len(c)):
        i=c.pick_index(rng,True,seen)
        assert i not in seen
        seen.add(i)
    with pytest.raises(FileExistsError): build_synthetic(tmp_path,cfg,42)


def write_target(path,n=12):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["HWI","wind_speed_m_s","flapping_frequency_Hz","angle_deg","corrected_mean_thrust_N","corrected_mean_lift_N","selection_status"])
        for k in range(n):
            x=[20+15*(k//4),2*((k//2)%2),1+k%2]
            writer.writerow([*x,5,0.1*x[2]+0.01*x[0],x[2]+0.1*x[1],"ok"])


def test_nested_splits_and_filter_identity(tmp_path):
    path=tmp_path/"target.csv"; write_target(path)
    table=next(iter(read_targets(path).values()))
    cfg=load_config(PROJECT_ROOT/"configs/smoke.yaml")["evaluation"]
    manifest=make_splits(table,cfg)
    validate_splits(manifest,table)
    loo=[s for s in manifest["splits"] if s["protocol"]=="loo"]
    assert len(loo)==12 and all(len(s["train"])==11 for s in loo)
    manifest["splits"][0]["inner"][0]["train"].append(manifest["splits"][0]["test"][0])
    with pytest.raises(ValueError): validate_splits(manifest,table)


def test_static_nan_time_is_kept_and_geometry_is_held_out(tmp_path):
    raw=tmp_path/"raw"
    for geometry in ["S809","NACA4415"]:
        d=raw/"data_ASCI"/"NREL"/geometry; d.mkdir(parents=True)
        alpha=np.arange(12.)
        z=np.column_stack([np.full(12,np.nan),alpha,np.full(12,np.nan),alpha/10,np.ones(12)*0.1,np.zeros(12)])
        np.savetxt(d/"static.txt",z,header="Time AoA Flap Cl Cdp Cm",comments="")
    cfg=load_config(PROJECT_ROOT/"configs/default.yaml")["prepare"]
    info=prepare_sources(raw,tmp_path/"prepared",cfg,42)
    assert info["counts"]["val"]==3 and info["counts"]["train"]==3
    task=Corpus(tmp_path/"prepared","val").get(0)
    assert len(task.x)==12 and task.meta["static_time_nan_preserved"]
