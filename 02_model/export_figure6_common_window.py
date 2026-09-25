"""Figure 6 common-window examples. Separate illustration; never appended to benchmarks."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from preprocessing import preprocess_file,TARGET_COLS
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True,choices=["SAITS","MOMENT","MOMENT-FT","Spatial-Ridge","BiTFI"]);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    split=json.loads((ROOT/"03_result/comparison/split.json").read_text())
    sel=json.loads((ROOT/"03_result/figure6_common_window/selection.json").read_text())
    fp=next(fp for fp in split["test"] if Path(fp).stem==sel["greenhouse"])
    res=preprocess_file(fp,include_covariates=True);cols=list(TARGET_COLS.values())
    full=res["data"][cols];valid=full.notna().astype("float32")
    gs=full.index.get_loc(pd.Timestamp(sel["start"]));ge=full.index.get_loc(pd.Timestamp(sel["end"]))
    assert ge-gs+1==72 and full.iloc[gs:ge+1].notna().all().all()
    if args.model=="BiTFI":
        from spatial import NeighborBank
        from dafi import DAFITimesFM3
        model=DAFITimesFM3(bank=NeighborBank([Path(x) for x in split["train"]],cols),context_len=1440,name="DAFI-TimesFM3")
        assert model._ready()
    elif args.model=="Spatial-Ridge":
        from spatial import NeighborBank,SpatialRidgeImputation
        model=SpatialRidgeImputation(NeighborBank([Path(x) for x in split["train"]],cols))
    else:
        from models.imputation_models import SAITSImputation,MOMENTImputation,MOMENTFineTunedImputation
        model=SAITSImputation(ROOT/"03_result/comparison_saits/models/SAITS/best.pt") if args.model=="SAITS" else None if args.model=="MOMENT-FT" else MOMENTImputation(ROOT/"03_result/model_cache/huggingface/models--AutonLab--MOMENT-1-large/snapshots/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc")
        if args.model=="MOMENT-FT":
            model=MOMENTFineTunedImputation(ROOT/"03_result/model_cache/huggingface/models--AutonLab--MOMENT-1-large/snapshots/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc",ROOT/"03_result/comparison_moment-ft/models/MOMENT/best.pt")
    base=None if args.model=="Spatial-Ridge" else model.compute_base(full,valid)
    rows=[]
    jobs=[("A",[c]) for c in cols]+[("B",["Tin","RH","CO2"]),("C",cols)]
    lo=max(0,gs-72);hi=min(len(full),ge+25)
    for scenario,vs in jobs:
        if args.model in ["BiTFI","Spatial-Ridge"]:model.set_greenhouse(res["name"])
        art=pd.DataFrame(False,index=full.index,columns=cols);art.loc[full.index[gs:ge+1],vs]=True
        masked=full.mask(art);effective=valid.mask(art,0)
        pred=model.impute(masked,effective) if args.model=="Spatial-Ridge" else model.impute_artificial(masked,effective,art,base)
        assert np.isfinite(pred.to_numpy()[art.to_numpy()]).all()
        np.testing.assert_allclose(pred.to_numpy()[effective.eq(1).to_numpy()],full.to_numpy()[effective.eq(1).to_numpy()],rtol=1e-5,atol=1e-6)
        shown=vs if scenario=="A" else cols
        for c in shown:
            truth=res["scaler"][c].inverse_transform(full[c].iloc[lo:hi].to_numpy().reshape(-1,1)).ravel()
            y=res["scaler"][c].inverse_transform(pred[c].iloc[lo:hi].to_numpy().reshape(-1,1)).ravel()
            rows.append(pd.DataFrame(dict(model="DAFI-TimesFM3" if args.model=="BiTFI" else args.model,scenario=scenario,variable=c,datetime=full.index[lo:hi],hours=np.arange(lo,hi)-gs,truth=truth,prediction=y,artificial=art[c].iloc[lo:hi].to_numpy())))
        print(args.model,scenario,vs,"done",flush=True)
    out=ROOT/"03_result/figure6_common_window";out.mkdir(exist_ok=True)
    pd.concat(rows).to_csv(out/f"{args.model}.csv",index=False)
    (out/f"{args.model}_protocol.json").write_text(json.dumps({**sel,"scenarios":["A","B","C"],"selection_note":"Post hoc selected C/72h strength example; A and B rerun on the same window. Not a representative or aggregate evaluation.","display_context_hours":[72,24],"seed":42,"donor_cache":"Reset for each mask job; target-variable masking is identical across scenarios","observed_values_preserved":True},indent=2))
if __name__=="__main__":main()
