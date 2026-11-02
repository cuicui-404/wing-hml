"""Prepare once, audit, then train with automatic per-run resume."""
import argparse
import subprocess
import sys

from wing_hml.utils import load_config,project_path

p=argparse.ArgumentParser()
p.add_argument("--config",default="configs/default.yaml")
p.add_argument("--data-root",required=True); p.add_argument("--target",required=True)
p.add_argument("--methods",nargs="+",default=["hml","ds","np","ds_maml","np_maml"])
p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46])
args=p.parse_args()
cfg=load_config(args.config)

def run(command,*options):
    subprocess.run([sys.executable,"-m","wing_hml",command,"--config",args.config,*map(str,options)],check=True)

root=project_path(cfg["paths"]["checkpoints"]).parent
run("doctor","--out",root/"environment.json")
if not (project_path(cfg["paths"]["prepared"])/"manifest.json").exists():
    run("prepare","--data-root",args.data_root)
if not (project_path(cfg["paths"]["synthetic"])/"manifest.json").exists(): run("synthesize")
if not project_path(cfg["paths"]["splits"]).exists(): run("splits","--target",args.target)
run("audit","--target",args.target,"--out",root/"audit.json")
for method in args.methods:
    for seed in args.seeds: run("train","--method",method,"--seed",seed,"--auto-resume")
print("Training complete. Choose target models/protocols with the evaluate command in README.md.")
