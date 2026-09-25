"""Evaluate official imputation models on the existing ctx1440 mask protocol.

context_len in output denotes total model window (512h), not the mask-placement
context. mask_context_len_h=1440 is fixed to reproduce BiTFI's exact gap locations.
Natural missing values remain missing, no ground-truth-filled base is consumed.
"""
from __future__ import annotations
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from preprocessing import preprocess_file, TARGET_COLS
from masking_v2 import create_gap_masks, SCENARIO_CONFIGS, RANDOM_SEED
from run_comparison import make_group_labels, _stable_seed, _eval_one, _load_completed
from models.imputation_models import ROOT, SAITSImputation, MOMENTImputation, MOMENTFineTunedImputation, ClassicalImputation

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=["SAITS","MOMENT","MOMENT-FT","LI","SeasonalNaive","AG-LightGBM"],required=True)
    p.add_argument("--quick",action="store_true")
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--n-shards",type=int,default=1)
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42);np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    out=ROOT/"03_result"/("smoke_"+args.model.lower() if args.quick else "comparison_"+args.model.lower())
    if args.n_shards>1: out=out/"shards"/str(args.shard)
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/"03_result/comparison/split.json"
    split=json.loads(source.read_text())
    (out/"split.json").write_text(source.read_text())
    if args.model=="SAITS":
        checkpoint_dir=ROOT/"03_result/comparison_saits/models/SAITS"
        if not (checkpoint_dir/"complete.json").exists(): raise RuntimeError("SAITS training is not complete")
        m=SAITSImputation(checkpoint_dir/"best.pt")
    elif args.model in ("LI","SeasonalNaive","AG-LightGBM"):
        m=ClassicalImputation(args.model)
    else:
        from huggingface_hub import snapshot_download
        snapshot=snapshot_download("AutonLab/MOMENT-1-large",
            revision="ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc",
            cache_dir=str(ROOT/"03_result/model_cache/huggingface"),
            allow_patterns=["config.json","model.safetensors"])
        if args.model=="MOMENT-FT":
            m=MOMENTFineTunedImputation(Path(snapshot),ROOT/"03_result/comparison_moment-ft/models/MOMENT/best.pt")
        else:
            m=MOMENTImputation(Path(snapshot))
    params={"model":args.model,"window_len_h":m.window,"mask_context_len_h":1440,
            "seed":42,"scenarios":["A","B","C"],"gap_lengths_h":[6,12,24,72,168],
            "repeats":10,"split_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "input":"observed masked data only; original missing values explicitly masked; no base cache",
            "window":"gap-centered 512h, shifted to remain inside series",
            "context_note":"512h is TOTAL input including gap, not 512h per side",
            "torch":str(torch.__version__),"device":torch.cuda.get_device_name(0)}
    if args.model.startswith("MOMENT"):
        params.update(checkpoint=str(snapshot),mode="head-tuned reconstruction; independent channels" if args.model=="MOMENT-FT" else "zero-shot reconstruction; independent channels")
    if args.model=="MOMENT-FT":
        params["head_checkpoint"]="03_result/comparison_moment-ft/models/MOMENT/best.pt"
        params["head_checkpoint_sha256"]=hashlib.sha256((ROOT/params["head_checkpoint"]).read_bytes()).hexdigest()
    if args.model in ("LI","SeasonalNaive","AG-LightGBM"):
        params.update(window="existing baseline implementation", context_note="720h inference context; 1440h mask placement",input="existing baseline adapter; trained LightGBM checkpoint reused")
    (out/"protocol.json").write_text(json.dumps(params,indent=2))
    csv=out/"results.csv"; done=_load_completed(csv)
    manifest=[]; t0=time.time(); n_calls=0
    paths=split["test"][:1] if args.quick else split["test"]
    paths=paths[args.shard::args.n_shards]
    for fp in paths:
        res=preprocess_file(fp,include_covariates=True)
        if res is None:raise ValueError(f"Missing test site {fp}")
        name=res["name"];cols=list(TARGET_COLS.values())
        full=res["data"];target=full[cols]
        full_valid=full.notna().astype("float32")
        groups=make_group_labels(full.index)
        cov=[c for c in full.columns if c not in cols]
        ranges={c:float(res["data_raw"][c].max()-res["data_raw"][c].min()) for c in cols}
        base=m.compute_base(target,full_valid[cols])
        site_calls=0; site_start=time.time()
        for scenario,sets in SCENARIO_CONFIGS.items():
            for vs in sets:
                for gap in ([72] if args.quick else [6,12,24,72,168]):
                    masks=create_gap_masks(full,full_valid,scenario,gap,vs,context_len_h=1440,
                        n_repeats=1 if args.quick else 10,
                        random_seed=RANDOM_SEED+_stable_seed(f"{name}{scenario}{gap}"))
                    for mr in masks:
                        key=(name,args.model,scenario,",".join(vs),gap,mr.repeat)
                        manifest.append(dict(greenhouse=name,scenario=scenario,masked_vars=",".join(vs),
                            gap_length_h=gap,repeat=mr.repeat,start_idx=mr.gap_start_idx,
                            end_idx=mr.gap_end_idx,start_time=str(full.index[mr.gap_start_idx])))
                        if key in done:continue
                        rows=_eval_one(args.model,m,mr,res["scaler"],groups,name,cov,cols,m.window,
                                       base=base,norm_ranges=ranges)
                        for row in rows:
                            row["window_len_h"]=m.window;row["mask_context_len_h"]=1440
                        frame=pd.DataFrame(rows)
                        if not np.isfinite(frame.NMAE).all():raise FloatingPointError("Invalid NMAE")
                        frame.to_csv(csv,mode="a",header=not csv.exists(),index=False,encoding="utf-8-sig")
                        done.add(key);n_calls+=1;site_calls+=1
                        if mr.repeat==0 and gap==72:
                            artificial=mr.artificial_mask[cols].eq(0)
                            pred=m.impute_artificial(mr.masked_data[cols],mr.effective_mask[cols],artificial,base)
                            if not np.allclose(pred.to_numpy()[mr.effective_mask[cols].eq(1).to_numpy()],
                                               mr.masked_data[cols].to_numpy()[mr.effective_mask[cols].eq(1).to_numpy()]):
                                raise AssertionError("Observed values changed")
                            if args.quick and args.model in ("SAITS","MOMENT","MOMENT-FT"):
                                # Poison the unused base: predictions must remain identical.
                                poison=base.copy();poison[:]=1e6
                                again=m.impute_artificial(mr.masked_data[cols],mr.effective_mask[cols],artificial,poison)
                                np.testing.assert_allclose(pred.to_numpy(),again.to_numpy(),equal_nan=True)
                            lo=max(0,mr.gap_start_idx-72);hi=min(len(full),mr.gap_end_idx+73)
                            ex=[]
                            for c in vs:
                                z=pd.DataFrame({"datetime":full.index[lo:hi],"variable":c,
                                    "truth":mr.ground_truth[c].iloc[lo:hi].to_numpy(),
                                    "prediction":pred[c].iloc[lo:hi].to_numpy(),
                                    "artificial":artificial[c].iloc[lo:hi].to_numpy()})
                                ex.append(z)
                            ed=out/"examples";ed.mkdir(exist_ok=True)
                            pd.concat(ex).to_csv(ed/f"{name}_{scenario}_{'-'.join(vs)}_72h.csv",index=False)
                        if site_calls%50==0:print(f"{name} {site_calls} calls ({time.time()-site_start:.1f}s)",flush=True)
        print(f"DONE {name} calls={site_calls} time={time.time()-site_start:.1f}s",flush=True)
    pd.DataFrame(manifest).to_csv(out/"mask_manifest.csv",index=False)
    d=pd.read_csv(csv);a=d[d.group_type=="all"]
    scores=a.groupby("greenhouse").apply(lambda z:np.average(z.NMAE,weights=z.n_eval),include_groups=False)
    summary={"model":args.model,"n_greenhouses":len(scores),"rows":len(d),"all_cells":len(a),
             "NMAE":scores.mean(),"SD":scores.std(),"new_calls":n_calls,"elapsed_s":time.time()-t0}
    (out/"complete.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
