#!/usr/bin/env python3
"""Report on the T3-stable motion set: how the other arms fare on the motions where T3 never falls.
Inputs: kin/<arm>_s<seed>.npz + <arm>_s<seed>.json (held-out), kin_train/<arm>_s<seed>.{npz,json} (train), t3_stable_motions.json
Output: outputs/stage1_endpoint_compare/trajectories/stable_set_report.{md,json}
"""
import glob, json, collections
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]; OUT=REPO/"outputs/stage1_endpoint_compare"; DST=OUT/"trajectories"
ARMS=["T0","T1","T2","T3","T4"]
stable=json.load(open(DST/"t3_stable_motions.json"))["stable"]; keys=[(r["split"],r["traj"]) for r in stable]; name={(r["split"],r["traj"]):(r["motion"],r["frames"]) for r in stable}
early=collections.defaultdict(dict); met=collections.defaultdict(dict)
for split,d in (("val","kin"),("train","kin_train")):
    for f in sorted(glob.glob(str(OUT/d/"T*_s*.npz"))):
        z=np.load(f); arm,seed=Path(f).stem.split("_s"); seed=int(seed)
        for i,e in enumerate(z["early_terminated"]): early[(split,i)][(arm,seed)]=bool(e)
    jdir = OUT if split=="val" else OUT/"kin_train"
    for f in sorted(glob.glob(str(jdir/"T*_s*.json"))):
        dd=json.load(open(f)); arm,seed=dd["arm"],int(dd["seed"])
        for r in dd["per_traj"]: met[(split,r["traj"])][(arm,seed)]=r
def nofall(k,arm): return all(not v for (a,s),v in early[k].items() if a==arm)
def m(k,arm,key): 
    vals=[r[key] for (a,s),r in met[k].items() if a==arm]; return float(np.mean(vals)) if vals else np.nan
rows=[]
for k in keys:
    row={"split":k[0],"traj":k[1],"motion":name[k][0],"frames":name[k][1]}
    for a in ARMS:
        row[f"{a}.nofall"]=nofall(k,a); row[f"{a}.syn"]=m(k,a,"synergy_loss"); row[f"{a}.cos"]=m(k,a,"synergy_shape_cosine"); row[f"{a}.anchor"]=m(k,a,"anchor_loss"); row[f"{a}.cov"]=m(k,a,"coverage")
    rows.append(row)
summary={}
for a in ARMS:
    summary[a]={"nofall_all_seeds":int(sum(r[f"{a}.nofall"] for r in rows)),"nofall_heldout":int(sum(r[f"{a}.nofall"] for r in rows if r["split"]=="val")),
                "syn_mean":float(np.nanmean([r[f"{a}.syn"] for r in rows])),"cos_mean":float(np.nanmean([r[f"{a}.cos"] for r in rows])),
                "anchor_mean":float(np.nanmean([r[f"{a}.anchor"] for r in rows])),"cov_mean":float(np.nanmean([r[f"{a}.cov"] for r in rows]))}
wins={"T3<T4":int(sum(r["T3.syn"]<r["T4.syn"] for r in rows)),"T3<T0":int(sum(r["T3.syn"]<r["T0.syn"] for r in rows)),"T3<T2 syn":int(sum(r["T3.syn"]<r["T2.syn"] for r in rows)),"T3<T2 anchor":int(sum(r["T3.anchor"]<r["T2.anchor"] for r in rows))}
json.dump({"n":len(rows),"summary":summary,"wins":wins,"rows":rows},open(DST/"stable_set_report.json","w"),indent=1,ensure_ascii=False)
L=[f"# 稳定集（T3 三个 seed 都不倒的 {len(rows)} 条）上各臂的表现","","| arm | 在这 {n} 条上也全不倒 | 其中 held-out（共 {v}） | 协同 loss 均值 | 协同余弦 | anchor loss | 覆盖率 |".format(n=len(rows),v=sum(r['split']=='val' for r in rows)),"|---|---:|---:|---:|---:|---:|---:|"]
for a in ARMS:
    s=summary[a]; L.append(f"| {a} | {s['nofall_all_seeds']} | {s['nofall_heldout']} | {s['syn_mean']:.3f} | {s['cos_mean']:.3f} | {s['anchor_mean']:.3f} | {s['cov_mean']:.3f} |")
L+=["",f"逐条胜负（协同 loss）：T3 优于 T4 {wins['T3<T4']}/{len(rows)}，优于 T0 {wins['T3<T0']}/{len(rows)}，优于 T2 {wins['T3<T2 syn']}/{len(rows)}；anchor loss 上 T3 优于 T2 {wins['T3<T2 anchor']}/{len(rows)}。","",
    "## 逐条：各臂是否全 seed 不倒（✓/×）与协同 loss","","| split | traj | 动作 | 帧 | T0 | T1 | T2 | T3 | T4 | T0 syn | T1 syn | T2 syn | T3 syn | T4 syn |","|---|---:|---|---:|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|"]
for r in rows:
    L.append(f"| {r['split']} | {r['traj']} | {r['motion']} | {r['frames']} | "+" | ".join("✓" if r[f"{a}.nofall"] else "×" for a in ARMS)+" | "+" | ".join(f"{r[f'{a}.syn']:.2f}" for a in ARMS)+" |")
(DST/"stable_set_report.md").write_text("\n".join(L)+"\n",encoding="utf-8"); print("\n".join(L[:12]))
