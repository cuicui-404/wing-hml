"""Read-only import of the three local source datasets into portable task files."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

from .tasks import Task
from .utils import digest, file_hash, save_json, stable_seed


def partition_group(group: str, cfg: dict):
    if group in cfg["validation_groups"]:
        return "val"
    if group in cfg["test_groups"]:
        return "test"
    return "train"


def dominant_frequency(t, signal):
    t,signal = np.asarray(t),np.asarray(signal)
    valid = np.isfinite(t) & np.isfinite(signal)
    t,signal = t[valid],signal[valid]
    if len(t) < 8 or np.ptp(signal) < 1e-8:
        return 0.0
    dt = np.median(np.diff(t))
    if dt <= 0:
        return 0.0
    uniform = np.linspace(t[0],t[-1],len(t))
    z = np.interp(uniform,t,signal)
    power = abs(np.fft.rfft((z-z.mean())*np.hanning(len(z))))**2
    frequencies = np.fft.rfftfreq(len(z),d=(uniform[-1]-uniform[0])/(len(z)-1))
    power[0] = 0
    return float(frequencies[np.argmax(power)])


def backward_rate(t, values):
    if not np.isfinite(values).any():
        return np.full(len(values),np.nan)
    rates = np.zeros(len(values))
    rates[1:] = np.diff(values)/np.diff(t)
    return rates


class TaskWriter:
    def __init__(self,out,raw,cfg):
        self.out,self.raw,self.cfg = out,raw,cfg
        self.records,self.skipped = [],[]
        self.group_splits = {}

    def add(self,task,partition,family,parent):
        if len(task.x) < self.cfg.get("min_rows",8):
            self.skipped.append({"task":task.task_id,"reason":"too_few_rows","n":len(task.x)})
            return
        if not np.isfinite(task.x).any():
            self.skipped.append({"task":task.task_id,"reason":"all_features_missing"})
            return
        for group in np.unique(task.row_groups):
            previous = self.group_splits.setdefault(str(group),partition)
            if previous != partition:
                raise ValueError(f"Source group leaks across partitions: {group}")
        task.meta.update(partition=partition,family=family,parent=parent)
        path = Path("tasks")/(digest(task.task_id)[:24]+".npz")
        task.save(self.out/path)
        self.records.append({"task_id":task.task_id,"path":str(path),"data_sha256":file_hash(self.out/path),"partition":partition,
                             "family":family,"parent":parent,"rows":len(task.x),
                             "features":task.x.shape[1],"responses":task.y.shape[1],"meta":task.meta})


def prepare_asci(raw,writer,cfg):
    root = raw/"data_ASCI"
    if not root.is_dir():
        return
    summaries = defaultdict(list)
    cener_conditions = {}
    workbook = root/"CENER"/"Selected_runs.xlsx"
    if workbook.exists():
        import openpyxl
        book = openpyxl.load_workbook(workbook,read_only=True,data_only=True)
        for row in book.worksheets[0].iter_rows(min_row=3,values_only=True):
            if not isinstance(row[0],str): continue
            condition = {}
            for name,index in [("mean_alpha_deg",3),("alpha_amplitude_deg",4),("pitch_k",5),
                               ("pitch_frequency_Hz",6),("mean_flap_deg",8),("flap_amplitude_deg",9),
                               ("flap_k",10),("flap_frequency_Hz",11)]:
                value = row[index] if index<len(row) else None
                if isinstance(value,(int,float)) and np.isfinite(value): condition[name]=float(value)
            cener_conditions[row[0]]=condition
        book.close()
    for site in ["GLASGOW","NREL","CENER","Oldenburg"]:
        base = root/site
        files = [p for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")
                 and p.suffix.lower() not in {".xlsx",".xls"}]
        for path in sorted(files):
            rel = path.relative_to(root).as_posix()
            header = path.read_text(errors="replace").splitlines()[0].split()
            if not header or header[0].lower() != "time":
                raise ValueError(f"Unexpected ASCI header: {path}")
            data = np.loadtxt(path,skiprows=1,ndmin=2)
            if data.shape[1] != len(header):
                raise ValueError(f"ASCI header/column mismatch: {path}")
            geometry = "DU00W212" if site == "Oldenburg" else path.parent.name
            group = f"ASCI/{site}/{geometry}"
            partition = partition_group(group,cfg)
            t,alpha,flap = data[:,0],data[:,1],data[:,2]
            dynamic = bool(np.isfinite(t).any())
            if dynamic:
                finite=np.isfinite(t)
                data,t,alpha,flap=data[finite],t[finite],alpha[finite],flap[finite]
                if (np.diff(t)<0).any(): raise ValueError(f"Nonmonotonic dynamic time axis: {path}")
                keep = np.r_[True,np.diff(t)>0]
                data,t,alpha,flap = data[keep],t[keep],alpha[keep],flap[keep]
                f = dominant_frequency(t,alpha)
                phase = 2*np.pi*f*(t-t[0])
                x = np.column_stack([alpha,backward_rate(t,alpha),flap,backward_rate(t,flap),
                                     np.sin(phase),np.cos(phase)])
                names = ["alpha_deg","alpha_rate_deg_s","flap_deg","flap_rate_deg_s","sin_phase","cos_phase"]
            else:
                x = np.column_stack([alpha,flap])
                names = ["alpha_deg","flap_deg"]
                f = 0.0
            extra = dict(cener_conditions.get(path.name,{})) if site == "CENER" else {}
            if site == "Oldenburg":
                match = re.match(r"(\d+)_(.+)_(cl|tr)_(\d+)k-",path.name)
                if match:
                    extra = {"Re":int(match[4])*1000,"tripped":int(match[3]=="tr")}
            for key,value in extra.items():
                x = np.column_stack([x,np.full(len(x),value)])
                names.append(key)
            parent = "ASCI/"+rel
            if site == "Oldenburg":
                # Force and pressure measurements of the same test stay linked.
                parent = f"ASCI/Oldenburg/test_{path.name.split('_')[0]}"
            meta = {"source_file":str(path.relative_to(raw)),"source_sha256":file_hash(path),
                    "physical_group":group,"temporal":dynamic,"history_rows":1 if dynamic else 0,
                    "features":names,"units":"coefficient","frequency_estimate_Hz":f,
                    "static_time_nan_preserved":not dynamic,"matched_conditions":extra}
            for j,name in enumerate(header[3:],3):
                valid = np.isfinite(data[:,j]) & np.isfinite(alpha)
                if not valid.any():
                    continue
                groups = np.asarray([parent]*int(valid.sum()))
                task = Task(f"ASCI/{rel}/{name}",x[valid],data[valid,j],{**meta,"target":name},groups)
                writer.add(task,partition,f"ASCI/{site}",parent)
            if dynamic and len(data)>4:
                stats = [float(np.nanmean(alpha)),float(np.ptp(alpha)/2),f,
                         float(np.nanmean(flap)) if np.isfinite(flap).any() else np.nan,
                         float(np.nanmax(flap)-np.nanmin(flap))/2 if np.isfinite(flap).any() else np.nan]
                labels = {name:float(np.mean(data[np.isfinite(data[:,j]),j]))
                          for j,name in enumerate(header[3:],3) if np.isfinite(data[:,j]).any()}
                summaries[(group,partition,site)].append((stats,labels,parent,rel))
    for (group,partition,site),rows in summaries.items():
        for target in sorted({k for _,labels,_,_ in rows for k in labels}):
            selected = [r for r in rows if target in r[1]]
            task = Task(f"{group}/condition_means/{target}",np.asarray([r[0] for r in selected]),
                        np.asarray([r[1][target] for r in selected]),
                        {"features":["mean_alpha_deg","amplitude_alpha_deg","frequency_Hz",
                                     "mean_flap_deg","amplitude_flap_deg"],
                         "target":"record_mean_"+target,"units":"coefficient","temporal":False,
                         "physical_group":group,"averaging":"entire_supplied_record",
                         "source_files":[r[3] for r in selected]},
                        np.asarray([r[2] for r in selected]))
            writer.add(task,partition,f"ASCI/{site}",group+"/condition_means")


def prepare_dryad(raw,writer,cfg,seed):
    paths = list(raw.glob("doi_10_5061_dryad*/flapping_wing_aerodynamics.mat"))
    if not paths:
        return
    path = paths[0]
    data = loadmat(path,variable_names=["ds_pos","ds_u_raw","ds_y_raw"])
    pos,u,y = data["ds_pos"],data["ds_u_raw"],data["ds_y_raw"]
    if pos.shape[0] != 548 or u.shape[-1] != 7 or y.shape[-1] != 5:
        raise ValueError("Unexpected Dryad schema; review source documentation")
    inputs = ["speed","AoA","ax","ay","az","pitch_rate","pitch_acceleration"]
    outputs = ["Fx_normal","Fy_chordwise","Mx","My","Mz"]
    rng = np.random.default_rng(stable_seed("dryad-trajectory-partition",seed))
    order = rng.permutation(len(pos))
    partitions = {}
    a,b = int(0.8*len(pos)),int(0.9*len(pos))
    for split,ids in [("train",order[:a]),("val",order[a:b]),("test",order[b:])]:
        for i in ids: partitions[int(i)] = split
    rows = defaultdict(list)
    source_meta = {"source_file":str(path.relative_to(raw)),"source_sha256":file_hash(path),
                   "units":"original_dataset_units","frame":"wing_normal_chordwise_not_global_lift_thrust"}
    for i in range(len(pos)):
        t = np.arange(len(pos[i]))/25.0
        freq = dominant_frequency(t,pos[i,:,0])
        distance = max(2,int(0.6*25/freq)) if freq>0 else len(t)
        peaks,_ = find_peaks(pos[i,:,0],distance=distance)
        start,end = (int(peaks[0]),int(peaks[-1])+1) if len(peaks)>=2 else (0,len(t))
        a_u,a_p,a_y = u[i,start:end],pos[i,start:end],y[i,start:end]
        summary = [freq,*((np.max(a_p,0)-np.min(a_p,0))/2).tolist(),float(np.mean(a_u[:,0])),
                   float(np.sqrt(np.mean(a_u[:,0]**2))),float(np.mean(a_u[:,1])),float(np.std(a_u[:,1])),
                   float(np.sqrt(np.mean(a_u[:,2]**2))),float(np.sqrt(np.mean(a_u[:,5]**2)))]
        label = np.mean(a_y,axis=0)
        parent = f"Dryad/trajectory/{i}"
        rows[partitions[i]].append((summary,label,parent,start,end,len(peaks)-1))
        if cfg.get("include_dryad_instantaneous"):
            # Use causal input history only; never insert lagged force labels.
            x = np.column_stack([u[i,1:],u[i,:-1]])
            for j,target in enumerate(outputs):
                task = Task(f"{parent}/instant/{target}",x,y[i,1:,j],
                            {**source_meta,"target":target,"features":inputs+["lag_"+n for n in inputs],
                             "temporal":True,"history_rows":1},np.asarray([parent]*len(x)))
                writer.add(task,partitions[i],"Dryad/instantaneous",parent)
    for split,rr in rows.items():
        for j,target in enumerate(outputs):
            task = Task(f"Dryad/trajectory_means/{split}/{target}",np.asarray([r[0] for r in rr]),
                        np.asarray([r[1][j] for r in rr]),
                        {**source_meta,"target":"mean_"+target,"temporal":False,
                         "features":["frequency_Hz","stroke_amplitude","deviation_amplitude","pitch_amplitude",
                                     "mean_speed","rms_speed","mean_AoA","std_AoA","rms_ax","rms_pitch_rate"],
                         "averaging":"first_to_last_detected_stroke_peak_else_whole_record",
                         "intervals":[[r[2],r[3],r[4],max(0,r[5])] for r in rr]},
                        np.asarray([r[2] for r in rr]))
            writer.add(task,split,"Dryad/means",f"Dryad/trajectory_means/{split}")


def prepare_delfly(raw,writer,cfg):
    dirs = list(raw.glob("Data supporting*Delfly*"))
    if not dirs: return
    root = dirs[0]
    if cfg.get("include_delfly_summary"):
        for path in sorted(root.glob("fig_9b_avg*.txt")):
            z = np.loadtxt(path)
            if z.shape!=(15,6): raise ValueError(f"Unexpected DelFly summary schema: {path}")
            aircraft = "flex" if "_flex_" in path.name else "nimble"
            # Categories are one-hot; no fictitious ordering of maneuvers.
            x = np.eye(5)
            averaged = z.reshape(5,3,6).mean(1)
            group = f"DelFly/{aircraft}/summary"
            for j,target in enumerate(["Fx","Fy","Fz","Mx","My","Mz"]):
                task = Task(group+"/"+target,x,averaged[:,j],
                            {"source_file":str(path.relative_to(raw)),"features":[f"maneuver_{k}" for k in range(5)],
                             "target":target,"temporal":False,"auxiliary":True,"repeats_per_condition":3,
                             "units":"N" if j<3 else "Nmm",
                             "repeat_std":z.reshape(5,3,6).std(1)[:,j].tolist()},
                            np.asarray([group+f"/condition/{k}" for k in range(5)]))
                # Intentionally honor the global min_rows: set min_rows<=5 to use.
                writer.add(task,partition_group(group,cfg),"DelFly/summary",group)
    if cfg.get("include_delfly_tdms"):
        from nptdms import TdmsFile
        for path in sorted(root.glob("*.tdms")):
            tdms = TdmsFile.read(path)
            group = tdms["Time Domain"]
            control_names = [k for k in ["PWM1_ms","PWM2_ms","PWM3_ms","PWM4_ms","ESC1_ms","ESC2_ms"]
                             if k in group]
            force_names = [k for k in ["Fx_N","Fy_N","Fz_N","Mx_Nmm","My_Nmm","Mz_Nmm"] if k in group]
            if not control_names or not force_names: raise ValueError(f"No calibrated channels in {path}")
            reference = group[force_names[0]]
            t = reference.time_track()
            if len(t)<2: continue
            dt = float(np.median(np.diff(t)))
            w = max(1,int(round(cfg.get("tdms_window_s",0.5)/dt)))
            names = control_names+force_names
            series = []
            for name in names:
                c = group[name]
                series.append(np.interp(t,c.time_track(),c[:],left=np.nan,right=np.nan))
            z = np.column_stack(series)
            z = z[:len(z)//w*w].reshape(-1,w,len(names)).mean(1)
            parent = "DelFly/TDMS/"+path.stem
            for j,target in enumerate(force_names):
                valid = np.isfinite(z[:,len(control_names)+j])
                task = Task(parent+"/"+target,z[valid,:len(control_names)],z[valid,len(control_names)+j],
                            {"source_file":str(path.relative_to(raw)),"features":control_names,"target":target,
                             "units":"N" if target.endswith("_N") else "Nmm","temporal":True,
                             "history_rows":1,"window_s":w*dt,"auxiliary":True},
                            np.asarray([parent]*int(valid.sum())))
                writer.add(task,partition_group(parent,cfg),"DelFly/TDMS",parent)
    if cfg.get("include_flight"):
        prepare_flight(root,raw,writer,cfg)


def prepare_flight(root,raw,writer,cfg):
    path = root/"fig_10_processed_flight_data.mat"
    keys = ["time_log","time_rate_log","time_acc_log","cmd_thrust","cmd_pitch","cmd_roll","cmd_yaw",
            "roll_log","pitch_log","yaw_log","roll_rate_log","pitch_rate_log","yaw_rate_log",
            "accx_log","accy_log","accz_log"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = loadmat(path,variable_names=keys)
    t = data["time_log"].ravel()
    features = ["cmd_thrust","cmd_pitch","cmd_roll","cmd_yaw","roll_log","pitch_log","yaw_log"]
    x = np.column_stack([data[k].ravel() for k in features])
    horizon = float(cfg.get("flight_horizon_s",0.1))
    parent = "DelFly/flight/"+path.stem
    for target in ["roll_log","pitch_log","yaw_log","roll_rate_log","pitch_rate_log","yaw_rate_log",
                   "accx_log","accy_log","accz_log"]:
        time_key = "time_acc_log" if target.startswith("acc") else "time_rate_log" if "rate" in target else "time_log"
        tt,values = data[time_key].ravel(),data[target].ravel()
        unique,ii = np.unique(tt,return_index=True)
        yy = np.interp(t+horizon,unique,values[ii],left=np.nan,right=np.nan)
        valid = np.isfinite(yy)&np.isfinite(x).all(1)
        task = Task(parent+"/"+target,x[valid],yy[valid],
                    {"source_file":str(path.relative_to(raw)),"features":features,"target":"future_"+target,
                     "units":"original_flight_log_units_not_force","temporal":True,"auxiliary":True,
                     "history_rows":max(1,int(np.ceil(horizon/np.median(np.diff(t))))),"horizon_s":horizon},
                    np.asarray([parent]*int(valid.sum())))
        writer.add(task,partition_group(parent,cfg),"DelFly/flight",parent)


def prepare_sources(raw: Path,out: Path,cfg: dict,seed: int):
    if not raw.is_dir(): raise FileNotFoundError(f"Raw source directory not found: {raw}")
    if (out/"manifest.json").exists(): raise FileExistsError(f"Prepared corpus exists: {out}; choose a new directory")
    overlap = set(cfg["validation_groups"])&set(cfg["test_groups"])
    if overlap: raise ValueError(f"Source validation/test group overlap: {overlap}")
    out.mkdir(parents=True,exist_ok=True)
    writer = TaskWriter(out,raw,cfg)
    prepare_asci(raw,writer,cfg)
    prepare_dryad(raw,writer,cfg,seed)
    prepare_delfly(raw,writer,cfg)
    if not writer.records: raise ValueError("No usable source tasks found; check raw-data layout")
    ids = [r["task_id"] for r in writer.records]
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate source task ids")
    info = {"format":"real-v1","config":cfg,"seed":seed,"tasks":writer.records,
            "counts":dict(Counter(r["partition"] for r in writer.records)),"skipped":writer.skipped,
            "task_hash":digest(writer.records),"group_audit":"passed_no_shared_raw_group_across_partitions"}
    save_json(out/"manifest.json",info)
    save_json(out/"summary.json",{k:v for k,v in info.items() if k!="tasks"})
    return info
