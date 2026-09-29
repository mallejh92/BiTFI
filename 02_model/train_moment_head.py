"""MOMENT reconstruction-head tuning; training greenhouses only, fixed validation."""
import argparse,json,time,copy,hashlib,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from models.imputation_models import ROOT,MOMENTImputation
OUT=ROOT/"03_result/comparison_moment-ft/models/MOMENT";GAPS=[6,12,24,72,168]
SNAP=ROOT/"03_result/model_cache/huggingface/models--AutonLab--MOMENT-1-large/snapshots/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"
def features(model,windows,validation=False):
    rng=np.random.RandomState(142 if validation else 42);items=[]
    for wi,w in enumerate(windows):
        for repeat in range(5 if validation else 2):
            h=GAPS[repeat] if validation else int(rng.choice(GAPS))
            # Match gap-centered inference; vary patch phase by a few hours.
            start=(512-h)//2+(0 if validation else int(rng.randint(-4,5)))
            for c in range(5):
                truth=w[:,c].copy();obs=np.isfinite(truth);art=np.zeros(512,bool);art[start:start+h]=obs[start:start+h]
                mask=obs&~art
                if art.any() and mask.sum()>=48:items.append((truth,mask,art))
    chunks={k:[] for k in ["features","target","mask","mean","std"]}
    capture={}
    hook=model.head.register_forward_pre_hook(lambda module,args:capture.update(x=args[0].detach()))
    for i in range(0,len(items),32):
        batch=items[i:i+32]
        truth=torch.tensor(np.stack([np.nan_to_num(a[0],nan=0) for a in batch]),device="cuda")
        mask=torch.tensor(np.stack([a[1] for a in batch]),device="cuda")
        art=torch.tensor(np.stack([a[2] for a in batch]),device="cuda")
        x=truth.masked_fill(~mask,0).unsqueeze(1)
        with torch.no_grad():
            result=model(x_enc=x,mask=mask.float(),input_mask=torch.ones_like(mask,dtype=torch.float32)).reconstruction
            feat=capture["x"].squeeze(1);mp=art.reshape(-1,64,8);sel=mp.any(-1)
            means=model.normalizer.mean.reshape(-1,1,1).expand(-1,64,1)
            std=model.normalizer.stdev.reshape(-1,1,1).expand(-1,64,1)
            cached=model.head.linear(feat)*std+means
            torch.testing.assert_close(cached.reshape(-1,512),result.squeeze(1),atol=1e-5,rtol=1e-4)
        for key,value in [("features",feat[sel]),("target",truth.reshape(-1,64,8)[sel]),("mask",mp[sel]),("mean",means[sel]),("std",std[sel])]:
            chunks[key].append(value.cpu())
    hook.remove();return {k:torch.cat(v) for k,v in chunks.items()}
def evaluate(head,d):
    head.eval();total=0.;n=0.
    with torch.no_grad():
        for i in range(0,len(d["features"]),2048):
            sl=slice(i,i+2048);p=head(d["features"][sl])*d["std"][sl]+d["mean"][sl]
            total+=((p-d["target"][sl]).abs()*d["mask"][sl]).sum().item();n+=d["mask"][sl].sum().item()
    return total/n
def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--learning-rates',type=float,nargs='+',default=[1e-4,3e-4,1e-3])
    args=parser.parse_args()
    if args.epochs<1 or args.patience<1 or any(lr<=0 for lr in args.learning_rates):
        parser.error('Epochs, patience and learning rates must be positive')
    clean=os.environ.get("BITFI_CLEAN")=="1"
    if clean:
        from clean_protocol import OUT as CLEAN
        OUT=CLEAN/"models/MOMENT"
    if args.out is not None:OUT=args.out
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    source=ROOT/"03_result/comparison_saits/models/SAITS"
    if clean:source=CLEAN/"models/SAITS"
    arrays=np.load(source/"windows.npz");protocol=json.loads((source/"training_protocol.json").read_text())
    split=json.loads((ROOT/"03_result/comparison/split.json").read_text())
    assert not ({s["name"] for s in protocol["sites"]}&{Path(p).stem for p in split["test"]})
    adapter=MOMENTImputation(SNAP);model=adapter.model
    assert not model.normalizer.affine
    initial=copy.deepcopy(model.head.linear.state_dict())
    cache=OUT/"frozen_features.pt"
    if cache.exists():stored=torch.load(cache,weights_only=True);train,val=stored["train"],stored["val"]
    else:
        print("Extract train",flush=True);train=features(model,arrays["train"])
        print("Extract validation",flush=True);val=features(model,arrays["val"],True)
        torch.save(dict(train=train,val=val),cache)
    del adapter,model;torch.cuda.empty_cache()
    train={k:v.cuda() for k,v in train.items()};val={k:v.cuda() for k,v in val.items()}
    protocol={k:protocol[k] for k in ["sites","window","train_windows","validation_windows","split","scaling"]}
    protocol.update(backbone="AutonLab/MOMENT-1-large",revision=SNAP.name,trainable="reconstruction head only (linear 1024→8)",encoder="frozen, eval mode; exact float32 hidden states cached",masking="contiguous 6/12/24/72/168 h; two fixed training corruptions/window, five fixed validation corruptions/window; independent channels",learning_rates=args.learning_rates,weight_decay=1e-4,epochs=args.epochs,patience=args.patience,batch_size=512,batch_unit="masked patches from cached independent-channel encoder outputs",dropout=0.1,seed=42,objective="masked MAE in existing normalized physical-variable scale",selection="lowest validation masked MAE across learning rates and epochs; test never used",features_reconstruction_equivalence_checked=True,device=torch.cuda.get_device_name(0),torch=str(torch.__version__))
    (OUT/"training_protocol.json").write_text(json.dumps(protocol,indent=2))
    bestglobal=float("inf");allhistory=[];runs=[];t0=time.time()
    for lr in protocol["learning_rates"]:
        torch.manual_seed(42)
        linear=torch.nn.Linear(1024,8).cuda();linear.load_state_dict(initial)
        head=torch.nn.Sequential(torch.nn.Dropout(.1),linear)
        baseline=evaluate(head,val);best=baseline;stale=0;state=copy.deepcopy(linear.state_dict());best_epoch=0
        optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-4)
        for epoch in range(1,args.epochs+1):
            head.train();order=torch.randperm(len(train["features"]),device="cuda");losses=[]
            for ids in order.split(512):
                pred=head(train["features"][ids])*train["std"][ids]+train["mean"][ids]
                loss=((pred-train["target"][ids]).abs()*train["mask"][ids]).sum()/train["mask"][ids].sum()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(loss.item())
            score=evaluate(head,val)
            allhistory.append(dict(learning_rate=lr,epoch=epoch,train_loss=np.mean(losses),validation_mae=score,zero_shot_validation_mae=baseline,elapsed_s=time.time()-t0))
            if score<best:best=score;stale=0;state=copy.deepcopy(linear.state_dict());best_epoch=epoch
            else:stale+=1
            print(f"lr={lr} epoch={epoch} val={score:.6f} best={best:.6f} stale={stale}",flush=True)
            if stale>=args.patience:break
        runs.append(dict(learning_rate=lr,best_validation_mae=best,best_epoch=best_epoch,zero_shot_validation_mae=baseline,epochs=epoch,stopped_early=stale>=args.patience,best_at_epoch_limit=best_epoch==args.epochs))
        if best<bestglobal:
            bestglobal=best
            torch.save(dict(head_state_dict={"linear."+k:v.cpu() for k,v in state.items()},learning_rate=lr,epoch=best_epoch,validation_mae=best,zero_shot_validation_mae=baseline,seed=42),OUT/"best.pt")
        pd.DataFrame(allhistory).to_csv(OUT/"history.csv",index=False)
    pd.DataFrame(runs).to_csv(OUT/"search.csv",index=False)
    (OUT/"complete.json").write_text(json.dumps(dict(best_validation_mae=bestglobal,zero_shot_validation_mae=baseline,runs=runs,elapsed_s=time.time()-t0),indent=2))
    print("DONE",runs,flush=True)
if __name__=="__main__":main()
