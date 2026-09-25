"""Train official SAITS using only the 24 training greenhouses.

Per-site chronological 80/20 train/validation split, with no windows crossing
the boundary. ORT + MIT losses are unchanged from the official implementation.
MIT mixes MCAR (25%) and A/B/C contiguous gaps (25% each); gap lengths match
the benchmark. Validation corruption is fixed; test results never select weights.
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocess_file, TARGET_COLS
from models.imputation_models import saits_network, WINDOW, ROOT

GAPS = [6, 12, 24, 72, 168]

def corrupt(raw, rng):
    observed = np.isfinite(raw)
    indicating = np.zeros(raw.shape, dtype=bool)
    kind = rng.randint(4)
    if kind == 0:
        indicating = (rng.rand(*raw.shape) < 0.2) & observed
    else:
        length = int(rng.choice(GAPS))
        start = rng.randint(0, len(raw) - length + 1)
        cols = [int(rng.randint(5))] if kind == 1 else ([0, 2, 3] if kind == 2 else list(range(5)))
        indicating[start:start+length, cols] = True
        indicating &= observed
    mask = observed & ~indicating
    return {"X": np.where(mask, raw, 0).astype("float32"),
            "missing_mask": mask.astype("float32"),
            "X_holdout": np.nan_to_num(raw, nan=0).astype("float32"),
            "indicating_mask": indicating.astype("float32")}

class Windows(Dataset):
    def __init__(self, values, seed, fixed=False):
        self.values, self.seed, self.fixed = values, seed, fixed
        self.rng = np.random.RandomState(seed)
    def __len__(self):
        return len(self.values)
    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed+i) if self.fixed else self.rng
        return corrupt(self.values[i], rng)

def prepare(out, window):
    cache = out / "windows.npz"
    metadata = out / "training_protocol.json"
    split = json.loads((ROOT/"03_result/comparison/split.json").read_text())
    if cache.exists() and metadata.exists():
        arrays = np.load(cache)
        return arrays["train"], arrays["val"], json.loads(metadata.read_text())
    train, val, sites = [], [], []
    for fp in split["train"]:
        r = preprocess_file(fp)
        if r is None:
            continue
        a = r["data"][list(TARGET_COLS.values())].to_numpy(dtype=np.float32)
        cut = int(len(a)*0.8)
        counts = []
        for bucket, left, right, stride in [(train,0,cut,128),(val,cut,len(a),256)]:
            before = len(bucket)
            for start in range(left, right-window+1, stride):
                block = a[start:start+window]
                if np.isfinite(block).mean() >= 0.8 and np.isfinite(block).sum(axis=0).min() >= 48:
                    bucket.append(block)
            counts.append(len(bucket)-before)
        sites.append({"name":r["name"],"split_index":cut,"n_train":counts[0],"n_val":counts[1]})
    if set(x["name"] for x in sites) & set(Path(p).stem for p in split["test"]):
        raise AssertionError("Train/test overlap")
    if not train or not val:
        raise ValueError("Empty training or validation windows")
    train, val = np.stack(train), np.stack(val)
    np.savez_compressed(cache, train=train, val=val)
    info = {"sites":sites,"window":window,"train_windows":len(train),"validation_windows":len(val),
            "split":"chronological 80/20 within each training greenhouse; non-crossing windows",
            "scaling":"existing per-greenhouse MinMax preprocessing",
            "masking":"uniform mixture: MCAR 20%, scenario A, B, C; block lengths 6/12/24/72/168",
            "SAITS_commit":subprocess.check_output(["git","-C",str(ROOT/"third_party/SAITS"),"rev-parse","HEAD"],text=True).strip()}
    metadata.write_text(json.dumps(info, indent=2))
    return train, val, info

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clean",action="store_true")
    p.add_argument("--out", type=Path, default=ROOT/"03_result/comparison_saits/models/SAITS")
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--patience",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision("high")
    if args.clean:
        from clean_protocol import windows
        tr,va,info=windows(args.out,WINDOW)
    else:
        tr,va,info=prepare(args.out,WINDOW)
    info.update(seed=args.seed,max_epochs=args.epochs,patience=args.patience,
                batch_size=args.batch_size,learning_rate=0.001,
                architecture={"n_groups":2,"d_model":256,"d_inner":128,"n_heads":4,"dropout":0.1},
                objective="ORT + MIT, equal weights",checkpoint_selection="fixed validation masked MAE",
                torch=str(torch.__version__),device=torch.cuda.get_device_name(0))
    (args.out/"training_protocol.json").write_text(json.dumps(info,indent=2))
    print(f"Train {tr.shape}; validation {va.shape}",flush=True)
    dl=DataLoader(Windows(tr,args.seed),batch_size=args.batch_size,shuffle=True,num_workers=0)
    vl=DataLoader(Windows(va,142,fixed=True),batch_size=args.batch_size,num_workers=0)
    model=saits_network()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    best=float("inf"); stale=0; history=[]; t0=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for b in dl:
            b={k:v.cuda() for k,v in b.items()}
            optimizer.zero_grad(set_to_none=True)
            result=model(b,stage="train")
            loss=result["reconstruction_loss"]+result["imputation_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("SAITS training loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); losses.append(loss.item())
        model.eval(); total=0.; count=0.
        with torch.inference_mode():
            for b in vl:
                b={k:v.cuda() for k,v in b.items()}
                pred=model.impute(b)[0]
                total+=((pred-b["X_holdout"]).abs()*b["indicating_mask"]).sum().item()
                count+=b["indicating_mask"].sum().item()
        score=total/count
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_mae=score,
                            elapsed_s=time.time()-t0))
        pd.DataFrame(history).to_csv(args.out/"history.csv",index=False)
        improved=score<best
        if improved:
            best=score; stale=0
            torch.save({"state_dict":model.state_dict(),"window":WINDOW,"epoch":epoch,
                        "validation_mae":score,"seed":args.seed},args.out/"best.pt")
        else: stale+=1
        print(f"Epoch {epoch:03d}: train={np.mean(losses):.6f} val={score:.6f} best={best:.6f} stale={stale} time={time.time()-t0:.0f}s",flush=True)
        if stale>=args.patience:break
    (args.out/"complete.json").write_text(json.dumps({"epochs":epoch,"best_validation_mae":best,
              "elapsed_s":time.time()-t0,"stopped_early":stale>=args.patience},indent=2))

if __name__=="__main__":
    main()
