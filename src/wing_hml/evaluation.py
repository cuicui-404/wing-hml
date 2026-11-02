from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from .baselines import ALL_MODELS,NEURAL,build_estimator,candidates
from .models import base_method,build_model
from .target import FEATURES,read_targets,validate_splits
from .tasks import Episode,EpisodeScaler,pack_episodes
from .training import environment_info,load_checkpoint,predict_batch
from .utils import digest,file_hash,get_device,read_json,save_json,set_seed

RANDOM_MODELS = set(NEURAL+["random_forest","xgboost_lite","target_mlp","tabpfn"])


class Predictor:
    def __init__(self,cfg,checkpoint_root,checkpoint_overrides=None):
        self.cfg=cfg
        self.device=get_device(cfg.get("device","auto"))
        self.root=checkpoint_root
        self.overrides=checkpoint_overrides or {}
        self.cache={}

    def checkpoint_path(self,name,seed):
        method=base_method(name)
        return Path(self.overrides.get(name,self.overrides.get(method,
                    self.root/method/f"seed_{seed}"/"best.pt")))

    def get_neural(self,name,seed):
        key=(base_method(name),seed)
        if key not in self.cache:
            path=self.checkpoint_path(name,seed)
            saved=load_checkpoint(path)
            if saved["method"]!=base_method(name) or saved["seed"]!=seed:
                raise ValueError(f"Checkpoint method/seed mismatch for {name}/{seed}: {path}")
            model=build_model(name,saved["model_config"]).to(self.device)
            model.load_state_dict(saved["model_state"])
            model.eval()
            self.cache[key]=(model,saved)
        return self.cache[key]

    def predict(self,name,params,x,y,qx,seed):
        scaler=EpisodeScaler().fit(x,y)
        sx,sy,query=scaler.transform_x(x),scaler.transform_y(y),scaler.transform_x(qx)
        if name in NEURAL:
            model,saved=self.get_neural(name,seed)
            episode=Episode(sx,sy,query,np.zeros((len(query),sy.shape[1])),
                            np.arange(len(sx)),np.arange(len(query)),"target_support_only")
            batch=pack_episodes([episode],self.device)
            tc=saved["config"]["training"]
            pred=predict_batch(model,batch,name,tc["inner_steps"],params.get("inner_lr",tc["inner_lr"]))
            output=pred[0].detach().cpu().numpy()
        else:
            estimator=build_estimator(name,params,sx.shape[1],seed)
            with warnings.catch_warnings():
                # Convergence messages remain available once per location;
                # no estimator failures are hidden or converted to successes.
                warnings.simplefilter("once")
                estimator.fit(sx,sy[:,0])
            output=np.asarray(estimator.predict(query)).reshape(-1,1)
        result=scaler.inverse_y(output)[:,0]
        if result.shape!=(len(qx),) or not np.isfinite(result).all():
            raise FloatingPointError(f"Nonfinite or wrong-shape prediction: {name}")
        return result


def select_params(predictor,name,split,x,y,seed):
    grid=candidates(name,small=len(split["train"])<8)
    if len(grid)==1:
        return grid[0],None
    if not split["inner"]:
        raise ValueError("Multiple candidates requested without inner validation")
    scores=[]
    for params in grid:
        fold_scores=[]
        for inner in split["inner"]:
            a,b=inner["train"],inner["test"]
            pred=predictor.predict(name,params,x[a],y[a],x[b],seed)
            fold_scores.append(float(np.sqrt(mean_squared_error(y[b],pred))))
        scores.append(float(np.mean(fold_scores)))
    best=int(np.argmin(scores))
    return grid[best],scores[best]


def compute_metrics(actual,predicted):
    actual,predicted=np.asarray(actual),np.asarray(predicted)
    result={"rmse_N":float(np.sqrt(mean_squared_error(actual,predicted))),
            "mae_N":float(mean_absolute_error(actual,predicted)),"r2":None}
    if len(actual)>1 and np.var(actual)>0:
        result["r2"]=float(r2_score(actual,predicted))
    return result


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def summarize(rows,splits,out):
    expected=defaultdict(int)
    expected_splits=defaultdict(set)
    for split in splits:
        expected[split["protocol"]]+=len(split["test"])
        expected_splits[split["protocol"]].add(split["id"])
    groups=defaultdict(list)
    for row in rows:
        groups[(row["model"],row["target"],row["seed"],row["protocol"])].append(row)
    metrics,fold_metrics=[],[]
    for (model,target,seed,protocol),rr in groups.items():
        successes=[r for r in rr if r["status"]=="ok"]
        record={"model":model,"target":target,"seed":seed,"protocol":protocol,
                "successful_predictions":len(successes),"failed_predictions":len(rr)-len(successes),
                "expected_predictions":expected[protocol],"status":"complete" if len(successes)==expected[protocol] else "partial_or_failed",
                "rmse_N":None,"mae_N":None,"r2":None}
        by_fold=defaultdict(list)
        for r in successes: by_fold[r["split_id"]].append(r)
        local=[]
        for split_id,values in by_fold.items():
            m=compute_metrics([r["actual_N"] for r in values],[r["predicted_N"] for r in values])
            fold_metrics.append({"model":model,"target":target,"seed":seed,"protocol":protocol,
                                 "split_id":split_id,"n_test":len(values),**m})
            local.append(m)
        if record["status"]=="complete":
            if protocol.startswith("fewshot_"):
                for key in ["rmse_N","mae_N","r2"]:
                    vals=[m[key] for m in local if m[key] is not None]
                    record[key]=float(np.mean(vals)) if vals else None
                    record[key+"_split_std"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
            else:
                record.update(compute_metrics([r["actual_N"] for r in successes],[r["predicted_N"] for r in successes]))
        metrics.append(record)
    combined=defaultdict(list)
    for row in metrics:
        if row["status"]=="complete": combined[(row["model"],row["target"],row["protocol"])].append(row)
    summary=[]
    for (model,target,protocol),rr in combined.items():
        row={"model":model,"target":target,"protocol":protocol,"n_model_seeds":len(rr)}
        for key in ["rmse_N","mae_N","r2"]:
            vals=[r[key] for r in rr if r[key] is not None]
            row[key+"_mean"]=float(np.mean(vals)) if vals else None
            row[key+"_std"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
        summary.append(row)
    for target,protocol in {(r["target"],r["protocol"]) for r in summary}:
        ranked=sorted([r for r in summary if r["target"]==target and r["protocol"]==protocol],key=lambda r:r["rmse_N_mean"])
        for rank,row in enumerate(ranked,1): row["rank_by_rmse"]=rank
    write_csv(out/"metrics.csv",metrics)
    write_csv(out/"summary.csv",summary)
    write_csv(out/"fold_metrics.csv",fold_metrics)
    write_csv(out/"predictions.csv",rows)
    return summary


def evaluate_target(cfg,target_path,manifest_path,models,seeds,checkpoint_root,out,
                    protocols=None,filter_method=None,quality_only=False,resume=False,
                    fold_start=0,fold_stop=None,checkpoint_overrides=None,allow_failures=False):
    tables=read_targets(target_path,quality_only)
    if filter_method and filter_method!="all":
        if filter_method not in tables: raise ValueError(f"Unknown filter {filter_method}; choices={list(tables)}")
        tables={filter_method:tables[filter_method]}
    manifest=read_json(manifest_path)
    chosen=[s for s in manifest["splits"] if not protocols or s["protocol"] in protocols]
    if protocols and set(protocols)-{s["protocol"] for s in chosen}: raise ValueError("Protocol absent from manifest")
    if not chosen: raise ValueError("No requested target splits")
    chosen_window=chosen[fold_start:fold_stop]
    if not chosen_window: raise ValueError("Empty evaluation split range")
    for name in models:
        if name not in ALL_MODELS: raise ValueError(f"Unknown method {name}")
    predictor=Predictor(cfg,checkpoint_root,checkpoint_overrides)
    used_seeds={name:list(seeds) if name in RANDOM_MODELS else [seeds[0]] for name in models}
    checkpoint_hashes={}
    # Missing checkpoints fail before any target evaluation starts.
    for name in models:
        if name in NEURAL:
            for seed in used_seeds[name]:
                path=predictor.checkpoint_path(name,seed)
                if not path.exists(): raise FileNotFoundError(f"Missing checkpoint: {path}; train the base model first")
                predictor.get_neural(name,seed)
                checkpoint_hashes[f"{name}/{seed}"]=file_hash(path)
    results={}
    for filter_name,table in tables.items():
        validate_splits(manifest,table)
        sub=out/("default" if filter_name=="default" else digest(filter_name)[:12])
        sub.mkdir(parents=True,exist_ok=True)
        signature=digest({"target_labels":table.label_signature,"conditions":table.signature,
                          "manifest":file_hash(manifest_path),"models":models,"seeds":used_seeds,
                          "protocols":protocols,"checkpoints":checkpoint_hashes,
                          "code":environment_info()["code_hash"],"inner_steps":cfg["training"]["inner_steps"]})
        run_path=sub/"run.json"
        old_rows=[]
        if run_path.exists():
            if not resume: raise FileExistsError(f"Evaluation exists: {sub}; use --resume or a new output")
            if read_json(run_path)["signature"]!=signature: raise ValueError("Evaluation resume signature differs")
            log=sub/"predictions.jsonl"
            if log.exists():
                with log.open(encoding="utf-8") as f:
                    for line in f:
                        old_rows.append(json.loads(line))
        save_json(run_path,{"signature":signature,"config":cfg,"environment":environment_info(),
                           "filter_method":filter_name,"source_target":str(target_path),"source_target_sha256":file_hash(target_path),
                           "checkpoints":checkpoint_hashes,"manifest":manifest,"models":models,"seeds":used_seeds})
        def key(r): return (r["model"],r["target"],r["seed"],r["split_id"],r["condition_id"])
        old_successes=[r for r in old_rows if r["status"]=="ok"]
        if len({key(r) for r in old_successes})!=len(old_successes): raise ValueError("Duplicate successful saved prediction keys")
        done={key(r) for r in old_rows if r["status"]=="ok"}
        rows={key(r):r for r in old_rows}
        failed=0
        for model in models:
            for seed in used_seeds[model]:
                set_seed(seed,int(cfg.get("threads",1)))
                for target,y in table.targets.items():
                    for split in chosen_window:
                        pending=[k for k in split["test"] if (model,target,seed,split["id"],table.ids[k]) not in done]
                        if not pending: continue
                        start_time=time.monotonic()
                        try:
                            params,inner_score=select_params(predictor,model,split,table.x,y,seed)
                            train_ids=split["train"]
                            prediction=predictor.predict(model,params,table.x[train_ids],y[train_ids],table.x[pending],seed)
                            error=None
                        except Exception as exc:
                            if not allow_failures: raise
                            params,inner_score,prediction={},None,[None]*len(pending)
                            error=f"{type(exc).__name__}: {exc}"
                            failed+=len(pending)
                        with (sub/"predictions.jsonl").open("a",encoding="utf-8") as stream:
                            for index,pred in zip(pending,prediction):
                                row={"filter_method":filter_name,"model":model,"target":target,"seed":seed,
                                     "protocol":split["protocol"],"split_id":split["id"],"condition_id":table.ids[index],
                                     **{f:float(table.x[index,j]) for j,f in enumerate(FEATURES)},
                                     "selection_status":table.metadata[index].get("selection_status",""),
                                     "actual_N":float(y[index]),"predicted_N":float(pred) if pred is not None else None,
                                     "residual_N":float(pred-y[index]) if pred is not None else None,
                                     "n_train":len(split["train"]),"n_test":len(split["test"]),"best_params":params,
                                     "inner_cv_rmse_N":inner_score,"elapsed_s":time.monotonic()-start_time,
                                     "status":"ok" if error is None else "failed","error":error}
                                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                                rows[key(row)]=row
                        print(f"{filter_name}: {model}/{target}/seed={seed}/{split['id']}",flush=True)
        summary=summarize(list(rows.values()),chosen,sub)
        save_json(sub/"completion.json",{"selected_splits":len(chosen),"requested_window":len(chosen_window),
                  "successful_predictions":sum(r["status"]=="ok" for r in rows.values()),"failures_this_run":failed,
                  "summary_rows":len(summary),"partial_results_are_not_ranked":True})
        results[filter_name]={"directory":str(sub),"summary":summary}
    return results


def predict_independent(cfg,train_csv,query_csv,models,seeds,checkpoint_root,out,filter_method=None):
    """Select on old support only; optional new labels are used only for scoring."""
    from .target import TARGETS,inner_split
    tables=read_targets(train_csv)
    if filter_method is None:
        if len(tables)!=1: raise ValueError("Choose --filter for a multi-filter training table")
        table=next(iter(tables.values()))
    else:
        table=tables[filter_method]
    with query_csv.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if not set(FEATURES)<=set(reader.fieldnames or []): raise ValueError(f"Query CSV needs {FEATURES}")
        queries=list(reader)
    if not queries: raise ValueError("Empty query CSV")
    qx=np.asarray([[float(row[k]) for k in FEATURES] for row in queries])
    if not np.isfinite(qx).all(): raise ValueError("Nonfinite query attributes")
    ids=["|".join(f"{v:.12g}" for v in row) for row in qx]
    if len(ids)!=len(set(ids)) or set(ids)&set(table.ids):
        raise ValueError("Independent query conditions must be distinct and absent from support conditions")
    angles={float(r["angle_deg"]) for r in queries if r.get("angle_deg")}
    original_angles={float(r["angle_deg"]) for r in table.metadata if r.get("angle_deg")}
    if len(angles)>1 or (angles and original_angles and angles!=original_angles):
        raise ValueError("New angle differs from the fixed-angle training task")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"predictions.csv").exists(): raise FileExistsError(f"Independent predictions already exist: {out}")
    predictor=Predictor(cfg,checkpoint_root)
    split={"train":list(range(len(table.x))),"inner":inner_split(list(range(len(table.x))),cfg["evaluation"]["split_seed"])}
    rows,metrics=[],[]
    for name in models:
        for seed in (seeds if name in RANDOM_MODELS else [seeds[0]]):
            set_seed(seed,int(cfg.get("threads",1)))
            for target,y in table.targets.items():
                params,inner_score=select_params(predictor,name,split,table.x,y,seed)
                pred=predictor.predict(name,params,table.x,y,qx,seed)
                # Labels are never passed to fitting/selection above.
                actual=[float(r[TARGETS[target]]) if r.get(TARGETS[target],"") else None for r in queries]
                for i,prediction in enumerate(pred):
                    rows.append({"model":name,"seed":seed,"target":target,"condition_id":ids[i],
                                 **{k:float(qx[i,j]) for j,k in enumerate(FEATURES)},
                                 "predicted_N":float(prediction),"actual_N":actual[i],"n_support":len(table.x),
                                 "best_params":params,"inner_cv_rmse_N":inner_score})
                if all(v is not None and np.isfinite(v) for v in actual):
                    metrics.append({"model":name,"seed":seed,"target":target,"n_test":len(qx),**compute_metrics(actual,pred)})
    write_csv(out/"predictions.csv",rows)
    write_csv(out/"metrics.csv",metrics)
    save_json(out/"run.json",{"support_sha256":file_hash(train_csv),"query_sha256":file_hash(query_csv),
                              "models":models,"seeds":seeds,"config":cfg,"environment":environment_info(),
                              "new_labels_used_for_selection":False})
    return rows
