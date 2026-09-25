"""Controlled univariate backbone comparison: identical past-only inputs and masks."""
import argparse,json,time,hashlib,contextlib,io,os
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"03_result/univariate_backbone_comparison"
CLEAN=os.environ.get("BITFI_CLEAN")=="1" or (os.environ.get("BITFI_CLEAN")!="0" and (ROOT/"03_result/active_evaluation.json").exists())
if CLEAN:
    OUT=ROOT/"03_result/reevaluation_clean_20260911/univariate_backbone_comparison"

def prepare():
    from preprocessing import preprocess_file,TARGET_COLS
    OUT.mkdir(exist_ok=True)
    clean=CLEAN
    manifest=pd.read_csv(ROOT/("03_result/reevaluation_clean_20260911/mask_manifest.csv" if clean else "03_result/comparison_saits/mask_manifest.csv"))
    split=json.loads((ROOT/"03_result/comparison/split.json").read_text())
    contexts=[];targets=[];rows=[]
    for fp in split["test"]:
        if clean:
            from clean_protocol import site as load_site
            res=load_site(Path(fp).stem)
        else:
            with contextlib.redirect_stdout(io.StringIO()):res=preprocess_file(fp,include_covariates=True)
        full=res["data"];site=res["name"]
        for q in manifest[manifest.greenhouse==site].itertuples():
            for v in q.masked_vars.split(","):
                gs,ge=int(q.start_idx),int(q.end_idx);h=ge-gs+1
                past=full[v].iloc[gs-1440:gs].copy()
                assert len(past)==1440 and past.notna().any()
                # Natural missingness is filled within the past slice only.
                past=past.interpolate(limit_direction="both").ffill().bfill().to_numpy(np.float32)
                truth=full[v].iloc[gs:ge+1].to_numpy(np.float32)
                assert np.isfinite(past).all() and np.isfinite(truth).all()
                targets.append(np.pad(truth,(0,168-h),constant_values=np.nan));contexts.append(past)
                scale=float(res["scaler"][v].data_range_[0]);span=float(res["data_raw"][v].max()-res["data_raw"][v].min())
                rows.append(dict(greenhouse=site,scenario=q.scenario,masked_vars=q.masked_vars,gap_length_h=h,repeat=q.repeat,variable=v,n_eval=h,context_len=1440,group_type="all",group_value="all",start_time=q.start_time,physical_scale=scale,nmae_factor=scale/span))
        print("Prepared",site,len(rows),flush=True)
    meta=pd.DataFrame(rows);meta.to_csv(OUT/"cases.csv",index=False)
    np.savez_compressed(OUT/"inputs.npz",context=np.stack(contexts),truth=np.stack(targets))
    p=dict(protocol="clean-20260911" if clean else "original",n_cases=len(meta),context_hours=1440,information="One target variable; pre-gap observations only; no covariates or post-gap observations",natural_missingness="Linear interpolation and endpoint fill inside past context only, identical for all backbones",mask_source="03_result/comparison_saits/mask_manifest.csv",horizons=[6,12,24,72,168],forecast="Full gap horizon in one call; no chunking",seed=42,inputs_sha256=hashlib.sha256((OUT/"inputs.npz").read_bytes()).hexdigest())
    if clean:p["mask_source"]="03_result/reevaluation_clean_20260911/mask_manifest.csv"
    (OUT/"protocol.json").write_text(json.dumps(p,indent=2));print(p,flush=True)
def run(name):
    import torch
    torch.set_num_threads(4);torch.manual_seed(42);torch.set_float32_matmul_precision("high")
    data=np.load(OUT/"inputs.npz");x=data["context"];y=data["truth"];meta=pd.read_csv(OUT/"cases.csv")
    start=time.time();pred=np.full_like(y,np.nan)
    if name=="Chronos2":
        from chronos import Chronos2Pipeline
        model=Chronos2Pipeline.from_pretrained("amazon/chronos-2",device_map="cuda",dtype=torch.float32)
        def predict(inputs,h):
            _,means=model.predict_quantiles(inputs=[a for a in inputs],prediction_length=h,quantile_levels=[.1,.5,.9],context_length=1440,cross_learning=False,batch_size=32)
            return np.stack([a.detach().float().cpu().numpy().reshape(-1)[:h] for a in means])
    elif name=="TimesFM2.5":
        import timesfm
        model=timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        model.compile(timesfm.ForecastConfig(max_context=1440,max_horizon=256,per_core_batch_size=32,normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=False,fix_quantile_crossing=True))
        def predict(inputs,h):return np.asarray(model.forecast(horizon=h,inputs=list(inputs))[0])[:,:h]
    else:
        import timesfm
        model=timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch",device="cuda")
        def predict(inputs,h):return np.stack([np.asarray(a.forecast).reshape(-1)[:h] for a in model.predict_batch(contexts=list(inputs),horizon=h)])
    # Check that batching does not leak information between unrelated series.
    ix=np.flatnonzero(meta.gap_length_h.eq(24).to_numpy())[:2]
    one=predict(x[ix[:1]],24);two=predict(x[ix],24)[:1]
    np.testing.assert_allclose(one,two,atol=2e-4,rtol=2e-3)
    for h in [6,12,24,72,168]:
        inds=np.flatnonzero(meta.gap_length_h.eq(h).to_numpy())
        for j in range(0,len(inds),32):
            ids=inds[j:j+32];p=predict(x[ids],h)
            assert p.shape==(len(ids),h) and np.isfinite(p).all()
            pred[ids,:h]=p
        print(name,"horizon",h,"cases",len(inds),"seconds",round(time.time()-start),flush=True)
    err=np.abs(pred-y);mean=np.nanmean(err,axis=1);mse=np.nanmean((pred-y)**2,axis=1)
    result=meta.copy();result["model"]=name+"-UNI";result["MAE"]=mean*meta.physical_scale;result["MSE"]=mse*meta.physical_scale**2;result["NMAE"]=mean*meta.nmae_factor
    result.to_csv(OUT/f"{name}_results.csv",index=False);np.savez_compressed(OUT/f"{name}_predictions.npz",prediction=pred)
    (OUT/f"{name}_complete.json").write_text(json.dumps(dict(model=name,n_cases=len(meta),elapsed_s=time.time()-start,batch_independence_check=True,torch=str(torch.__version__)),indent=2))
    print(name,"DONE",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--model",choices=["Chronos2","TimesFM2.5","TimesFM3.0"]);a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.model)
