from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from .models import adapt_parameters,base_method,batch_loss,build_model,forward_batch
from .tasks import Corpus,pack_episodes,sample_episode
from .utils import PROJECT_ROOT,digest,file_hash,get_device,read_json,save_json,set_seed

TRAIN_METHODS = ["hml","ds","np","ds_maml","np_maml"]


def environment_info():
    packages={}
    for name in ["torch","numpy","scipy","scikit-learn","PyYAML","npTDMS","openpyxl"]:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    return {"packages":packages,"cuda_available":torch.cuda.is_available(),"torch_cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python":os.sys.version,"code_hash":digest({str(p.relative_to(PROJECT_ROOT)):file_hash(p)
                      for p in sorted((PROJECT_ROOT/"src").rglob("*.py"))})}


def save_checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix=".pt")
    os.close(fd)
    try:
        torch.save(value,name)
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def load_checkpoint(path,device="cpu"):
    # Checkpoints contain optimizer/RNG state. Only load files you created or trust.
    return torch.load(path,map_location=device,weights_only=False)


def rng_state(rng):
    return {"numpy_sampler":rng.bit_generator.state,"numpy_global":np.random.get_state(),
            "python":random.getstate(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state,rng):
    rng.bit_generator.state=state["numpy_sampler"]
    np.random.set_state(state["numpy_global"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def episode_from(corpus,rng,cfg,balanced=False):
    task=corpus.pick(rng,balanced)
    return sample_episode(task,rng,cfg["support_sizes"],cfg["query_sizes"],cfg.get("temporal_gap",1))


def validation_plan(corpus,episode_cfg,count,seed,balanced=False):
    rng=np.random.default_rng(seed)
    return [episode_from(corpus,rng,episode_cfg,balanced) for _ in range(count)]


def predict_batch(model,batch,method,steps,lr):
    model.eval()
    if method.endswith("_ft") or method.endswith("_maml"):
        with torch.enable_grad():
            params=adapt_parameters(model,batch,steps,lr,detach=True)
        with torch.no_grad(): return forward_batch(model,batch,params)
    with torch.no_grad(): return forward_batch(model,batch)


def score_plan(model,plan,method,device,steps,lr):
    values=[]
    for episode in plan:
        batch=pack_episodes([episode],device)
        pred=predict_batch(model,batch,method,steps,lr)
        values.append(float(batch_loss(pred,batch)))
    return float(np.mean(values))


def train(cfg,method,seed,synthetic_root,real_root,out,stage="all",resume=False,init_checkpoint=None):
    if method not in TRAIN_METHODS:
        raise ValueError(f"Train one of {TRAIN_METHODS}; *_ft reuses its base checkpoint at evaluation")
    set_seed(seed,int(cfg.get("threads",1)))
    device=get_device(cfg.get("device","auto"))
    synth_train,synth_val=Corpus(synthetic_root,"train"),Corpus(synthetic_root,"val")
    if not len(synth_train) or not len(synth_val): raise ValueError("Synthetic train/val tasks are required")
    real_train=real_val=None
    if real_root and (real_root/"manifest.json").exists():
        real_train,real_val=Corpus(real_root,"train"),Corpus(real_root,"val")
    tc=cfg["training"]
    phases=["synthetic","mixed"] if stage=="all" else [stage]
    phases=[s for s in phases if int(tc[s+"_steps"])>0]
    if not phases: raise ValueError("No training steps requested")
    if "mixed" in phases and (real_train is None or not len(real_train) or not len(real_val)):
        raise ValueError("Mixed stage requires nonempty prepared real train AND val partitions; run prepare or choose --stage synthetic")
    out.mkdir(parents=True,exist_ok=True)
    if not 0<=tc["real_probability"]<=1 or min(tc["batch_size"],tc["validation_every"],tc["validation_episodes"],tc["patience"])<1:
        raise ValueError("Invalid training probabilities or batch/validation budget")
    for phase in phases:
        available=len(synth_train)+(len(real_train) if phase=="mixed" else 0)
        if tc["batch_size"]>available: raise ValueError("Batch size exceeds distinct source task count")
    env=environment_info()
    setting={k:v for k,v in tc.items() if k not in {"synthetic_steps","mixed_steps"}}
    signature=digest({"method":method,"seed":seed,"model":cfg["model"],"episode":cfg["episode"],
                      "settings":setting,"synthetic":synth_train.info["spec_hash"],
                      "real":real_train.info["task_hash"] if real_train else None,
                      "code_hash":env["code_hash"],"packages":env["packages"]})
    last=out/"last.pt"
    previous=load_checkpoint(last) if resume else None
    if resume and previous["signature"]!=signature: raise ValueError("Resume config/data signature differs from saved run")
    if previous and previous["phase"] not in phases:
        raise ValueError("The saved phase is not in the requested training stages")
    if previous and previous["phase"]=="mixed" and tc["synthetic_steps"]!=previous["config"]["training"]["synthetic_steps"]:
        raise ValueError("Cannot change the completed synthetic-stage budget after mixed training has begun")
    if not resume and (out/"run.json").exists(): raise FileExistsError(f"Run exists: {out}; use --resume or a new output")
    model=build_model(method,cfg["model"]).to(device)
    if previous:
        model.load_state_dict(previous["model_state"])
    elif init_checkpoint:
        initial=load_checkpoint(init_checkpoint)
        if base_method(initial["method"])!=base_method(method) or initial["model_config"]!=cfg["model"]:
            raise ValueError("Initial checkpoint architecture/method differs")
        model.load_state_dict(initial["model_state"])
    elif phases[0]=="mixed":
        raise ValueError("Standalone mixed training requires --init-checkpoint or --resume")
    rng=np.random.default_rng(seed)
    plans={"synthetic":validation_plan(synth_val,cfg["episode"],tc["validation_episodes"],seed+2020)}
    if real_val and len(real_val):
        plans["real"]=validation_plan(real_val,cfg["episode"],tc["validation_episodes"],seed+3030,True)
    save_json(out/"validation_episodes.json",{k:[{"task_id":e.task_id,"support":e.support_indices.tolist(),
                       "query":e.query_indices.tolist()} for e in v] for k,v in plans.items()})
    save_json(out/"run.json",{"signature":signature,"config":cfg,"method":method,"seed":seed,
                 "environment":env,"parameter_count":sum(p.numel() for p in model.parameters()),
                 "init_checkpoint":str(init_checkpoint) if init_checkpoint else None,
                 "init_checkpoint_sha256":file_hash(init_checkpoint) if init_checkpoint else None})
    start_time=time.monotonic()
    for phase in phases:
        if previous and phase=="synthetic" and previous["phase"]=="mixed": continue
        optimizer=torch.optim.Adam(model.parameters(),lr=float(tc[phase+"_lr"]))
        start,best,bad=0,float("inf"),0
        if previous and previous["phase"]==phase:
            optimizer.load_state_dict(previous["optimizer_state"])
            start,best,bad=previous["step"],previous["best_score"],previous["bad_validations"]
            restore_rng(previous["rng_state"],rng)
        steps=int(tc[phase+"_steps"])
        if start>steps: raise ValueError("Requested step budget is less than the resume step")
        for step in range(start+1,steps+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            episodes=[]
            seen_synthetic,seen_real=set(),set()
            for _ in range(tc["batch_size"]):
                use_real=phase=="mixed" and rng.random()<tc["real_probability"]
                if use_real and len(seen_real)>=len(real_train): use_real=False
                if not use_real and len(seen_synthetic)>=len(synth_train): use_real=True
                corpus=real_train if use_real else synth_train
                seen=seen_real if use_real else seen_synthetic
                index=corpus.pick_index(rng,bool(tc["balanced_real"]) if use_real else False,seen)
                seen.add(index)
                task=corpus.get(index)
                ec=cfg["episode"]
                episodes.append(sample_episode(task,rng,ec["support_sizes"],ec["query_sizes"],ec.get("temporal_gap",1)))
            if method.endswith("_maml"):
                loss_value=0.
                for episode in episodes:
                    batch=pack_episodes([episode],device)
                    params=adapt_parameters(model,batch,tc["inner_steps"],tc["inner_lr"],create_graph=True)
                    loss=batch_loss(forward_batch(model,batch,params),batch)/len(episodes)
                    if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite MAML loss: {episode.task_id}")
                    loss.backward()
                    loss_value+=float(loss.detach())
            else:
                batch=pack_episodes(episodes,device)
                loss=batch_loss(forward_batch(model,batch),batch)
                if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss in {[e.task_id for e in episodes]}")
                loss.backward()
                loss_value=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc["grad_clip"]),error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%int(tc["validation_every"])==0 or step==steps:
                scores={k:score_plan(model,plan,method,device,tc["inner_steps"],tc["inner_lr"])
                        for k,plan in plans.items() if phase=="mixed" or k=="synthetic"}
                score=float(np.mean(list(scores.values())))
                if not np.isfinite(score): raise FloatingPointError("Nonfinite validation score")
                improved=score<best
                if improved: best,bad=score,0
                else: bad+=1
                record={"phase":phase,"step":step,"loss":loss_value,"validation":scores,
                        "selection_score":score,"best_score":best,"elapsed_s":time.monotonic()-start_time,
                        "task_exposures_this_phase":step*int(tc["batch_size"]),
                        "real_tasks_in_batch":len(seen_real),"synthetic_tasks_in_batch":len(seen_synthetic)}
                with (out/"history.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(record)+"\n")
                checkpoint={"format":"wing-hml-checkpoint-v1","method":method,"seed":seed,"signature":signature,
                            "model_config":cfg["model"],"config":cfg,"model_state":model.state_dict(),
                            "optimizer_state":optimizer.state_dict(),"phase":phase,"step":step,
                            "best_score":best,"bad_validations":bad,"rng_state":rng_state(rng),
                            "environment":env,"source_hashes":{"synthetic":synth_train.info["spec_hash"],
                                      "real":real_train.info["task_hash"] if real_train else None}}
                save_checkpoint(last,checkpoint)
                if improved: save_checkpoint(out/f"{phase}_best.pt",checkpoint)
                print(f"{method} seed={seed} {phase} step={step}/{steps} loss={loss_value:.5g} val={score:.5g}",flush=True)
                if bad>=int(tc["patience"]): break
        best_checkpoint=load_checkpoint(out/f"{phase}_best.pt")
        model.load_state_dict(best_checkpoint["model_state"])
        save_checkpoint(out/"best.pt",best_checkpoint)
        previous=None
    save_json(out/"completed.json",{"best_checkpoint":"best.pt","elapsed_s":time.monotonic()-start_time,
                                    "method":method,"seed":seed,"signature":signature})
    return out/"best.pt"


def evaluate_source(checkpoint,corpus_root,partition,cfg,out,episodes=128):
    saved=load_checkpoint(checkpoint)
    set_seed(saved["seed"],int(cfg.get("threads",1)))
    device=get_device(cfg.get("device","auto"))
    model=build_model(saved["method"],saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state"])
    corpus=Corpus(corpus_root,partition)
    if not len(corpus): raise ValueError(f"Empty source partition {partition}")
    rng=np.random.default_rng(int(cfg["seed"])+9010)
    rows=[]
    for i in range(episodes):
        task=corpus.pick(rng,balanced=True)
        episode=sample_episode(task,rng,cfg["episode"]["support_sizes"],cfg["episode"]["query_sizes"],cfg["episode"].get("temporal_gap",1))
        batch=pack_episodes([episode],device)
        pred=predict_batch(model,batch,saved["method"],saved["config"]["training"]["inner_steps"],saved["config"]["training"]["inner_lr"])
        rows.append({"episode":i,"task_id":task.task_id,"family":task.meta.get("family"),"ood":task.meta.get("ood",False),
                     "mse_standardized":float(batch_loss(pred,batch)),"support":episode.support_indices.tolist(),
                     "query":episode.query_indices.tolist()})
    save_json(out,{"checkpoint_sha256":file_hash(checkpoint),"partition":partition,"episodes":rows,
                   "mse_standardized":float(np.mean([r["mse_standardized"] for r in rows])),
                   "by_ood":{str(v):float(np.mean([r["mse_standardized"] for r in rows if r["ood"]==v]))
                             for v in {r["ood"] for r in rows}}})
    return rows
