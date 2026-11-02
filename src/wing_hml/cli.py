from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import PROJECT_ROOT,load_config,project_path,read_json,save_json


def parser():
    main=argparse.ArgumentParser(description="Heterogeneous-attribute meta-learning for aerodynamic tables")
    sub=main.add_subparsers(dest="command",required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--config",default=str(PROJECT_ROOT/"configs/default.yaml"))
    common.add_argument("--set",action="append",default=[],metavar="key=value",help="YAML override, repeatable")
    p=sub.add_parser("doctor",parents=[common],help="Print dependency and GPU information")
    p.add_argument("--out")
    p=sub.add_parser("prepare",parents=[common],help="Convert local source data; raw files remain read-only")
    p.add_argument("--data-root"); p.add_argument("--out")
    p=sub.add_parser("synthesize",parents=[common],help="Create deterministic synthetic task specifications")
    p.add_argument("--out"); p.add_argument("--seed",type=int); p.add_argument("--materialize",action="store_true")
    p=sub.add_parser("splits",parents=[common],help="Write one shared manifest for all target methods")
    p.add_argument("--target"); p.add_argument("--out"); p.add_argument("--quality-only",action="store_true")
    p.add_argument("--expected-rows",type=int,default=64,help="0 disables the target row-count check")
    p=sub.add_parser("train",parents=[common],help="Train shared source weights; never reads the target CSV")
    p.add_argument("--method",choices=["hml","ds","np","ds_maml","np_maml"],default="hml")
    p.add_argument("--seed",type=int); p.add_argument("--synthetic"); p.add_argument("--real"); p.add_argument("--out")
    p.add_argument("--stage",choices=["all","synthetic","mixed"],default="all")
    p.add_argument("--resume",action="store_true"); p.add_argument("--auto-resume",action="store_true")
    p.add_argument("--init-checkpoint")
    p=sub.add_parser("meta-evaluate",parents=[common],help="Evaluate held-out synthetic/real source tasks")
    p.add_argument("--checkpoint",required=True); p.add_argument("--corpus",required=True)
    p.add_argument("--partition",choices=["val","test"],default="test"); p.add_argument("--episodes",type=int,default=128)
    p.add_argument("--out",required=True)
    p=sub.add_parser("evaluate",parents=[common],help="Common nested target evaluation for neural and classical models")
    p.add_argument("--target"); p.add_argument("--splits"); p.add_argument("--checkpoint-root"); p.add_argument("--out")
    p.add_argument("--models",help="Comma-separated model names or historical,neural,simple")
    p.add_argument("--seeds",help="Comma-separated model seeds")
    p.add_argument("--protocols",default="loo",help="loo,leave_hwi,leave_wind,leave_frequency,fewshot_3,...")
    p.add_argument("--filter",default="all"); p.add_argument("--quality-only",action="store_true")
    p.add_argument("--resume",action="store_true"); p.add_argument("--allow-failures",action="store_true")
    p.add_argument("--fold-start",type=int,default=0); p.add_argument("--fold-stop",type=int)
    p.add_argument("--checkpoint",action="append",default=[],metavar="model=path")
    p=sub.add_parser("predict",parents=[common],help="Support on all old conditions and predict independent new conditions")
    p.add_argument("--train-target"); p.add_argument("--query",required=True); p.add_argument("--out",required=True)
    p.add_argument("--checkpoint-root"); p.add_argument("--models",default="hml"); p.add_argument("--seeds",default="42")
    p.add_argument("--filter")
    p=sub.add_parser("audit",parents=[common],help="Check corpus hashes, source isolation and target split leakage")
    p.add_argument("--prepared"); p.add_argument("--synthetic"); p.add_argument("--target"); p.add_argument("--splits")
    p.add_argument("--quality-only",action="store_true"); p.add_argument("--out")
    return main


def main(argv=None):
    args=parser().parse_args(argv)
    cfg=load_config(args.config,args.set)
    path=lambda name,override=None:project_path(override or cfg["paths"][name])
    seed=getattr(args,"seed",None)
    seed=int(cfg["seed"] if seed is None else seed)
    if args.command=="doctor":
        from .training import environment_info
        info=environment_info()
        print(json.dumps(info,ensure_ascii=False,indent=2))
        if args.out: save_json(project_path(args.out),info)
    elif args.command=="prepare":
        from .prepare import prepare_sources
        info=prepare_sources(path("raw_data",args.data_root),path("prepared",args.out),cfg["prepare"],seed)
        print(json.dumps({"counts":info["counts"],"skipped":len(info["skipped"]),"output":str(path("prepared",args.out))},ensure_ascii=False))
    elif args.command=="synthesize":
        from .synthetic import build_synthetic
        info=build_synthetic(path("synthetic",args.out),cfg["synthetic"],seed,args.materialize)
        print(json.dumps({"counts":info["counts"],"output":str(path("synthetic",args.out))},ensure_ascii=False))
    elif args.command=="splits":
        from .target import make_splits,read_targets
        tables=read_targets(path("target",args.target),args.quality_only)
        table=next(iter(tables.values()))
        expected=0 if args.quality_only else args.expected_rows
        if expected and len(table.x)!=expected: raise ValueError(f"Expected {expected} target rows, got {len(table.x)}")
        out=path("splits",args.out)
        if out.exists(): raise FileExistsError(f"Split manifest exists: {out}; choose a new filename")
        result=make_splits(table,cfg["evaluation"])
        save_json(out,result)
        print(json.dumps({"conditions":len(table.x),"splits":len(result["splits"]),"output":str(out)}))
    elif args.command=="train":
        from .training import train
        out=project_path(args.out) if args.out else path("checkpoints")/args.method/f"seed_{seed}"
        result=train(cfg,args.method,seed,path("synthetic",args.synthetic),path("prepared",args.real),out,
                     args.stage,args.resume or (args.auto_resume and (out/"last.pt").exists()),
                     project_path(args.init_checkpoint) if args.init_checkpoint else None)
        print(f"Checkpoint: {result}")
    elif args.command=="meta-evaluate":
        from .training import evaluate_source
        evaluate_source(project_path(args.checkpoint),project_path(args.corpus),args.partition,cfg,project_path(args.out),args.episodes)
        print(f"Saved source evaluation: {project_path(args.out)}")
    elif args.command=="evaluate":
        from .baselines import resolve_models
        from .evaluation import evaluate_target
        names=resolve_models(args.models.split(",") if args.models else cfg["evaluation"]["models"])
        seeds=[int(s) for s in args.seeds.split(",")] if args.seeds else cfg["evaluation"]["seeds"]
        overrides={}
        for item in args.checkpoint:
            k,sep,v=item.partition("=")
            if not sep: raise ValueError("--checkpoint must be model=path")
            overrides[k]=str(project_path(v))
        result=evaluate_target(cfg,path("target",args.target),path("splits",args.splits),names,seeds,
                               path("checkpoints",args.checkpoint_root),path("results",args.out),
                               None if args.protocols=="all" else args.protocols.split(","),args.filter,args.quality_only,
                               args.resume,args.fold_start,args.fold_stop,overrides,args.allow_failures)
        print(json.dumps({k:v["directory"] for k,v in result.items()},ensure_ascii=False))
    elif args.command=="predict":
        from .baselines import resolve_models
        from .evaluation import predict_independent
        rows=predict_independent(cfg,path("target",args.train_target),project_path(args.query),
                                  resolve_models(args.models.split(",")),[int(s) for s in args.seeds.split(",")],
                                  path("checkpoints",args.checkpoint_root),project_path(args.out),args.filter)
        print(f"Saved {len(rows)} independent predictions: {project_path(args.out)}")
    elif args.command=="audit":
        from .audit import audit
        result=audit(path("prepared",args.prepared),path("synthetic",args.synthetic),path("target",args.target),
                     path("splits",args.splits),args.quality_only)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if args.out: save_json(project_path(args.out),result)
