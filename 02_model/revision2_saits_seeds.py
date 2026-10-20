"""Revision 2 (E4): training-seed repeats of the SAITS information controls.

Design: the training/validation windows, reference features and artificial-missingness corpus
(``revision/saits_information/{train,validation}.npz``) are fixed; only the training seed
(initialisation, mini-batch order, dropout) changes. Seed 42 is the original run; seeds 43-46
are repeats. Every repeat is scored on the same 3,421 masks with the unchanged evaluation
function of ``revision_saits_spatial.py`` (including its reference-feature poison check).

Greenhouses stay the statistical unit: seeds are never pooled as extra greenhouses. Each seed is
analysed on its own (n = 10 test greenhouses), and the seed-averaged score is the per-greenhouse
mean of the five per-seed greenhouse scores.

  run      seed-42 reproduction (reference only) and seeds 43-46, sequentially on one GPU
  smoke    end-to-end hidden-target (1e6 poison) invariance of every new checkpoint (GPU)
  analyze  writes analysis/revision2/saits_seeds_*.csv

Both steps require BITFI_RUN_ROOT (the revision-2 run root); ``run`` also requires an explicit
CUDA_VISIBLE_DEVICES.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if not os.environ.get('BITFI_RUN_ROOT'):raise SystemExit('Set BITFI_RUN_ROOT to the revision-2 run root')
RUN=Path(os.environ['BITFI_RUN_ROOT']).resolve()
SEEDS_DIR=RUN/'revision/saits_seeds';REPRO=SEEDS_DIR/'reproduction';ANALYSIS=RUN/'analysis/revision2'
ORIGINAL_SEED=42;REPEAT_SEEDS=[43,44,45,46];ALL_SEEDS=[ORIGINAL_SEED,*REPEAT_SEEDS]
MODELS={'spatial':'SAITS-spatial','local':'SAITS-matched-local'}
BITFI='BiTFI-TimesFM3'
# Families exactly as in analysis/main_paired.csv and analysis/component_paired.csv (Holm within family).
MAIN_REFERENCES=['LI','SeasonalNaive','AG-LightGBM','Spatial-Ridge','SAITS-spatial','MOMENT-FT','TimesFM3.0-COV-SPA']
COMPONENT_PAIRS=[('BiTFI-TimesFM3','BiTFI-TimesFM3-fwd'),('BiTFI-TimesFM3','BiTFI-Chronos2'),('SAITS-spatial','SAITS-matched-local'),('MOMENT-FT','MOMENT'),('TimesFM3.0-COV','TimesFM3.0'),('TimesFM3.0-COV-SPA','TimesFM3.0-COV')]
MAIN_MODELS=[BITFI,*MAIN_REFERENCES]
RANK_NEIGHBOURS=['TimesFM3.0-COV-SPA','AG-LightGBM']

def seed_folder(model,seed):return SEEDS_DIR/f'{model}_seed{seed}'

# ----------------------------------------------------------------------------------------- run
def jobs():
 out=[]
 for key,model in MODELS.items():
  out.append((f'train_{model}_seed42_rerun',['--train',key,'--seed','42','--out-root',str(REPRO),'--suffix','_seed42'],[REPRO/f'{model}_seed42/complete.json']))
 out.append(('evaluate_seed42_rerun',['--evaluate','--out-root',str(REPRO),'--suffix','_seed42'],[REPRO/f'{m}_seed42/evaluation_complete.json' for m in MODELS.values()]))
 for seed in REPEAT_SEEDS:
  for key,model in MODELS.items():
   out.append((f'train_{model}_seed{seed}',['--train',key,'--seed',str(seed),'--out-root',str(SEEDS_DIR),'--suffix',f'_seed{seed}'],[seed_folder(model,seed)/'complete.json']))
  out.append((f'evaluate_seed{seed}',['--evaluate','--out-root',str(SEEDS_DIR),'--suffix',f'_seed{seed}'],[seed_folder(m,seed)/'evaluation_complete.json' for m in MODELS.values()]))
 return out

def run():
 if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise SystemExit('Set CUDA_VISIBLE_DEVICES explicitly')
 env=os.environ.copy();env.update(BITFI_RUN_ROOT=str(RUN),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 for k,v in dict(BITFI_CLEAN='1',JAX_PLATFORMS='cpu',HF_HOME=str(ROOT/'03_result/model_cache/huggingface'),HF_HUB_OFFLINE='1').items():env.setdefault(k,v)
 logs=SEEDS_DIR/'logs';logs.mkdir(parents=True,exist_ok=True);status=[]
 for name,args,done in jobs():
  if all(p.exists() for p in done):print('skip',name,flush=True);status.append(dict(job=name,skipped=True));continue
  command=[sys.executable,'-u',str(ROOT/'02_model/revision_saits_spatial.py'),*args];start=time.time();print(time.strftime('%H:%M:%S'),name,flush=True)
  with (logs/(name+'.log')).open('a') as log:rc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  status.append(dict(job=name,command=command,returncode=rc,elapsed_s=time.time()-start,gpu=env['CUDA_VISIBLE_DEVICES']));(logs/'queue_status.json').write_text(json.dumps(status,indent=2))
  if rc or not all(p.exists() for p in done):raise RuntimeError(f'{name} failed (returncode {rc}); see {logs/(name+".log")}')
 print('QUEUE COMPLETE',flush=True)

# --------------------------------------------------------------------------------------- smoke
def smoke():
 """End-to-end hidden-target check (as in run_selected_refinement_controls.py): the gap values of the
 masked variables are replaced by 1e6 in data and data_raw; every SAITS output must stay identical.
 Feature construction and inference repeat revision_saits_spatial.evaluate() exactly."""
 import copy,numpy as np,pandas as pd,torch
 import clean_protocol as cp,revision_saits_spatial as rs
 from models.imputation_models import extract_window
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');rows=list(manifest.groupby(['scenario','gap_length_h'],sort=False).head(1).itertuples());bank=rs.ReferenceFeatures(False)
 def batch(objs):
  xs=[];ms=[]
  for o,r in zip(objs,rows):
   mr=cp.masked_case(o,r);masked=mr.masked_data[cp.COLS];eff=mr.effective_mask[cp.COLS];art=mr.artificial_mask[cp.COLS].eq(0);x,m,_,lo,hi=extract_window(masked,eff,art);full=masked.to_numpy(np.float32);observed=eff.ne(0).to_numpy()&np.isfinite(full)
   xx,mm,_=bank.augment(r.greenhouse,masked.index,full,observed,lo,hi,x,m);xs.append(xx);ms.append(mm)
  return np.stack(xs),np.stack(ms)
 clean=[cp.site(r.greenhouse) for r in rows];poisoned=[]
 for o,r in zip(clean,rows):
  q=copy.copy(o);q['data']=o['data'].copy();q['data_raw']=o['data_raw'].copy()
  for v in r.masked_vars.split(','):q['data'].iloc[r.start_idx:r.end_idx+1,q['data'].columns.get_loc(v)]=1e6;q['data_raw'].iloc[r.start_idx:r.end_idx+1,q['data_raw'].columns.get_loc(v)]=1e6
  # Positive control: the poison reaches the hidden truth that masked_case() exposes for scoring.
  assert (cp.masked_case(q,r).ground_truth[r.masked_vars.split(',')].iloc[r.start_idx:r.end_idx+1]==1e6).all().all()
  poisoned.append(q)
 (cx,cm),(px,pm)=batch(clean),batch(poisoned);np.testing.assert_array_equal(cx,px);np.testing.assert_array_equal(cm,pm);out=[]
 folders=[REPRO/f'{m}_seed42' for m in MODELS.values()]+[seed_folder(m,s) for s in REPEAT_SEEDS for m in MODELS.values()]
 for folder in folders:
  model=rs.network();model.load_state_dict(torch.load(folder/'best.pt',weights_only=True)['state_dict']);model.eval();res=[]
  for x,m in [(cx,cm),(px,pm)]:
   x=x.copy();m=m.copy()
   if folder.name.startswith('SAITS-matched-local'):x[:,:,9:]=0;m[:,:,9:]=0
   with torch.inference_mode():res.append(model.impute({'X':torch.from_numpy(x).cuda(),'missing_mask':torch.from_numpy(m.astype(np.float32)).cuda()})[0][:,:,:5].cpu().numpy())
  np.testing.assert_array_equal(res[0],res[1]);out.append(dict(model=folder.name,cases=len(rows),case_ids=[int(r.case_id) for r in rows],hidden_target_invariant=True,max_abs_output_difference=float(np.abs(res[0]-res[1]).max())))
 (SEEDS_DIR/'smoke_hidden_target.json').write_text(json.dumps(dict(check='gap values of masked variables set to 1e6 in data and data_raw; SAITS input features and full 512x5 outputs compared',features_identical=True,models=out,device=torch.cuda.get_device_name(0)),indent=2));print('SMOKE OK',len(out),'models',flush=True)

# ------------------------------------------------------------------------------------- analyze
def analyze():
 import numpy as np,pandas as pd,torch
 import analyze_revision_experiments as ar
 ANALYSIS.mkdir(parents=True,exist_ok=True);inputs=[];frames=[]
 def load(path,label,seed=None,role=''):
  d=pd.read_csv(path);a=d.query("group_type=='all'").copy()
  assert len(a)==6205 and a.case_id.nunique()==3421 and not a.duplicated(['case_id','variable']).any(),path
  a['model']=label;a['seed']=seed;a['role']=role
  inputs.append(dict(label=label,seed=seed,role=role,path=str(Path(path).relative_to(RUN)),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),modified=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(Path(path).stat().st_mtime)),rows_all=len(a),stored_model=';'.join(sorted(d.model.unique()))))
  return a
 label=lambda model,seed:f'{model}_seed{seed}'
 for model in MODELS.values():
  frames.append(load(RUN/'evaluation'/model/'results.csv',label(model,ORIGINAL_SEED),ORIGINAL_SEED,'original'))
  for seed in REPEAT_SEEDS:frames.append(load(seed_folder(model,seed)/'results.csv',label(model,seed),seed,'repeat'))
 seeds=pd.concat(frames,ignore_index=True)
 others=sorted({*MAIN_MODELS,*[m for p in COMPONENT_PAIRS for m in p]}-set(MODELS.values()))
 fixed=pd.concat([load(RUN/'evaluation'/m/'results.csv',m,role='comparator') for m in others],ignore_index=True)
 rerun=pd.concat([load(REPRO/f'{m}_seed42/results.csv',f'{m}_seed42rerun',ORIGINAL_SEED,'reproduction (reference only)') for m in MODELS.values()],ignore_index=True)
 # All seeds are scored on identical variable-cases with identical observation counts.
 key=['case_id','variable','greenhouse'];ref=seeds.query(f"model=='{label('SAITS-spatial',42)}'")[key+['n_eval']].sort_values(key).reset_index(drop=True)
 for name,g in pd.concat([seeds,rerun,fixed.query(f"model=='{BITFI}'")]).groupby('model'):assert g[key+['n_eval']].sort_values(key).reset_index(drop=True).equals(ref),name
 # The fixed data-side corpus every seed trained on (read from the original root only for comparison).
 original=ROOT/'03_result/reevaluation_fullgap_20260929/revision/saits_information'
 for f in ['train.npz','validation.npz','train_cases.json','validation_cases.json','protocol.json']:
  path=RUN/'revision/saits_information'/f;digest=hashlib.sha256(path.read_bytes()).hexdigest()
  inputs.append(dict(label=f,seed=None,role='fixed training corpus' if f!='protocol.json' else 'corpus protocol',path=str(path.relative_to(RUN)),sha256=digest,modified=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(path.stat().st_mtime)),rows_all=None,stored_model='',identical_to_original_root=digest==hashlib.sha256((original/f).read_bytes()).hexdigest()))
 pd.DataFrame(inputs).astype({'seed':'Int64','rows_all':'Int64'}).to_csv(ANALYSIS/'saits_seeds_inputs.csv',index=False)

 # Greenhouse scores per run, and the seed-averaged greenhouse score (mean of the five per-seed scores).
 site=ar.site_scores(pd.concat([seeds,rerun,fixed]));lookup=seeds.drop_duplicates('model').set_index('model')[['seed','role']]
 mean_rows=[]
 for model in MODELS.values():
  w=site[site.model.isin([label(model,s) for s in ALL_SEEDS])].pivot(index='greenhouse',columns='model',values='NMAE')
  assert w.shape==(10,5) and w.notna().all().all();mean_rows.append(pd.DataFrame(dict(model=f'{model}_seedmean',greenhouse=w.index,NMAE=w.mean(axis=1).to_numpy())))
 site=pd.concat([site,*mean_rows],ignore_index=True);p=site.pivot(index='greenhouse',columns='model',values='NMAE');assert p.shape[0]==10 and p.notna().all().all()
 long=site[site.model.str.startswith('SAITS-')|(site.model==BITFI)].copy();long['base_model']=long.model.str.replace(r'_seed.*$','',regex=True);long['seed']=long.model.str.extract(r'_seed(\w+)$')[0];long.to_csv(ANALYSIS/'saits_seeds_site_scores.csv',index=False)

 # 1) Per-seed overall NMAE (greenhouse-level mean, bootstrap CI over greenhouses) and training records.
 summ=ar.summary(pd.concat([seeds,rerun]));train_rows=[]
 for model in MODELS.values():
  for seed in ALL_SEEDS+['42rerun']:
   folder=RUN/'revision/saits_information'/model if seed==ORIGINAL_SEED else REPRO/f'{model}_seed42' if seed=='42rerun' else seed_folder(model,seed)
   c=json.loads((folder/'complete.json').read_text());h=pd.read_csv(folder/'history.csv');best=int(h.validation_mae.idxmin());ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
   assert int(ck['epoch'])==int(h.epoch[best]) and abs(float(ck['validation_mae'])-c['best_validation_mae'])<1e-12
   train_rows.append(dict(model=f'{model}_seed{seed}',base_model=model,seed=42 if seed=='42rerun' else seed,role='original' if seed==ORIGINAL_SEED else 'reproduction (reference only)' if seed=='42rerun' else 'repeat',epochs_run=c['epochs'],best_epoch=int(ck['epoch']),best_validation_mae=c['best_validation_mae'],stopped_early=c.get('stopped_early',len(h)<60),train_seconds=c['elapsed_s'],device=c['device'],torch=c['torch']))
 runs=pd.DataFrame(train_rows).merge(summ.rename(columns=dict(mean='NMAE',sd='greenhouse_sd',low='NMAE_low',high='NMAE_high')),on='model',how='left',validate='one_to_one')
 assert runs.NMAE.notna().all();runs.to_csv(ANALYSIS/'saits_seeds_summary.csv',index=False)

 # 2) Seed spread of the five runs (seed 42 = original result; the rerun is excluded).
 spread=[];five=runs[runs.role!='reproduction (reference only)']
 for model,g in five.groupby('base_model',sort=False):
  x=g.NMAE.to_numpy();assert len(x)==5
  w=site[site.model.isin([label(model,k) for k in ALL_SEEDS])].pivot(index='greenhouse',columns='model',values='NMAE');within=w.std(axis=1,ddof=1)
  z=site[site.model==f'{model}_seedmean'].NMAE.to_numpy();i=np.random.default_rng(42).integers(len(z),size=(10000,len(z)));lo,hi=np.quantile(z[i].mean(1),[.025,.975])
  spread.append(dict(base_model=model,n_seeds=5,seeds=' '.join(map(str,g.seed)),mean_of_seeds=x.mean(),sd_of_seeds=x.std(ddof=1),min=x.min(),max=x.max(),range=x.max()-x.min(),range_pct_of_mean=100*(x.max()-x.min())/x.mean(),cv_pct=100*x.std(ddof=1)/x.mean(),seed_min=int(g.seed.iloc[x.argmin()]),seed_max=int(g.seed.iloc[x.argmax()]),original_seed42=float(g[g.seed==ORIGINAL_SEED].NMAE.iloc[0]),seedmean_NMAE=z.mean(),seedmean_low=lo,seedmean_high=hi,best_validation_mae_min=g.best_validation_mae.min(),best_validation_mae_max=g.best_validation_mae.max(),best_epoch_min=g.best_epoch.min(),best_epoch_max=g.best_epoch.max(),greenhouse_seed_sd_mean=within.mean(),greenhouse_seed_sd_max=within.max(),greenhouse_seed_range_max=(w.max(axis=1)-w.min(axis=1)).max(),between_greenhouse_sd_seedmean=z.std(ddof=1)))
 pd.DataFrame(spread).to_csv(ANALYSIS/'saits_seeds_spread.csv',index=False)

 # 3) Paired greenhouse comparisons per seed, for the seed-averaged score, and for the 42 rerun.
 rows=[];variants=[(s,label('SAITS-spatial',s),label('SAITS-matched-local',s)) for s in ALL_SEEDS]+[('mean','SAITS-spatial_seedmean','SAITS-matched-local_seedmean'),('42rerun','SAITS-spatial_seed42rerun','SAITS-matched-local_seed42rerun')]
 for seed,sp,lo_ in variants:
  main=ar.compare(site,[(BITFI,sp if r=='SAITS-spatial' else r) for r in MAIN_REFERENCES]);comp=ar.compare(site,[(sp if a=='SAITS-spatial' else a,lo_ if b=='SAITS-matched-local' else b) for a,b in COMPONENT_PAIRS])
  extra=ar.compare(site,[(BITFI,lo_)])
  for kind,family,d,name in [('BiTFI reduction vs SAITS-spatial','main (7 BiTFI comparisons)',main,sp),('reference gain: SAITS-spatial vs SAITS-matched-local','component (6 comparisons)',comp,sp),('BiTFI reduction vs SAITS-matched-local (extra)','none (single comparison)',extra,lo_)]:
   r=d[(d.model==name)|(d.reference==name)].iloc[0].to_dict();x=r.pop('holm_p')
   rows.append(dict(seed=seed,comparison=kind,**r,holm_p=x if family!='none (single comparison)' else np.nan,holm_family=family,greenhouses_model_better=int((p[r['model']]<p[r['reference']]).sum())))
 # Greenhouse + training-seed bootstrap: each replicate resamples the 10 greenhouses and one of the five
 # training runs (seed index shared by the two SAITS models, as in the per-seed analysis).
 rng=np.random.default_rng(42);i=rng.integers(10,size=(10000,10));s=rng.integers(5,size=10000)
 A=np.stack([p[label('SAITS-spatial',k)].to_numpy() for k in ALL_SEEDS]);B=np.stack([p[label('SAITS-matched-local',k)].to_numpy() for k in ALL_SEEDS]);X=p[BITFI].to_numpy()
 for kind,model,reference,num,den,point in [('BiTFI reduction vs SAITS-spatial',BITFI,'SAITS-spatial (5 seeds)',X[i],A[s[:,None],i],100*(1-X.mean()/A.mean())),('reference gain: SAITS-spatial vs SAITS-matched-local','SAITS-spatial (5 seeds)','SAITS-matched-local (5 seeds)',A[s[:,None],i],B[s[:,None],i],100*(1-A.mean()/B.mean())),('BiTFI reduction vs SAITS-matched-local (extra)',BITFI,'SAITS-matched-local (5 seeds)',X[i],B[s[:,None],i],100*(1-X.mean()/B.mean()))]:
  lo,hi=np.quantile(100*(1-num.mean(1)/den.mean(1)),[.025,.975]);rows.append(dict(seed='greenhouse+seed bootstrap',comparison=kind,model=model,reference=reference,reduction=point,low=lo,high=hi,p=np.nan,n_sites=10,holm_p=np.nan,holm_family='none (interval only; point = reduction between five-seed means)',greenhouses_model_better=np.nan))
 paired=pd.DataFrame(rows);paired.to_csv(ANALYSIS/'saits_seeds_paired.csv',index=False)
 # Cross-seed spread of the per-seed paired statistics (five runs: original 42 and 43-46).
 g=paired[paired.seed.isin(ALL_SEEDS)];pspread=[]
 for kind,z in g.groupby('comparison',sort=False):
  assert len(z)==5;pspread.append(dict(comparison=kind,n_seeds=5,reduction_mean=z.reduction.mean(),reduction_sd=z.reduction.std(ddof=1),reduction_min=z.reduction.min(),reduction_max=z.reduction.max(),seed_min=z.seed.iloc[z.reduction.argmin()],seed_max=z.seed.iloc[z.reduction.argmax()],low_min=z.low.min(),high_max=z.high.max(),p_min=z.p.min(),p_max=z.p.max(),seeds_p_below_005=int((z.p<.05).sum()),holm_p_min=z.holm_p.min(),holm_p_max=z.holm_p.max(),seeds_holm_below_005=int((z.holm_p<.05).sum()) if z.holm_p.notna().all() else np.nan,seeds_low_above_0=int((z.low>0).sum()),greenhouses_model_better_min=int(z.greenhouses_model_better.min()),greenhouses_model_better_max=int(z.greenhouses_model_better.max())))
 pd.DataFrame(pspread).to_csv(ANALYSIS/'saits_seeds_paired_spread.csv',index=False)

 # 4) Rank of SAITS among the eight main configurations when its seed-42 run is replaced by seed k.
 ranks=[];overall=site.groupby('model').NMAE.mean()
 for seed in ALL_SEEDS+['mean']:
  for base in MODELS.values():
   name=f'{base}_seed{seed}';field=[m for m in MAIN_MODELS if m!='SAITS-spatial']+[name];order=overall[field].sort_values();rank=int(np.flatnonzero(order.index==name)[0])+1
   row=dict(seed=seed,model=name,NMAE=overall[name],rank_among_8_main=rank,field='8 main configurations, this run in the SAITS-spatial slot',order=' < '.join(order.index.str.replace(r'_seed\w+$','',regex=True)))
   for other in RANK_NEIGHBOURS:
    d=ar.compare(site,[(name,other)]).iloc[0];row|={f'{other}_NMAE':overall[other],f'better_than_{other}':bool(overall[name]<overall[other]),f'greenhouses_better_than_{other}':int((p[name]<p[other]).sum()),f'reduction_vs_{other}':d.reduction,f'p_vs_{other}':d.p}
   ranks.append(row)
 rank=pd.DataFrame(ranks);rank.to_csv(ANALYSIS/'saits_seeds_rank.csv',index=False)

 # 4b) Comparator sensitivity: comparators whose new-root file differs from the read-only original root
 # (e.g. the M4 no-history fix) are also evaluated with their original-root version.
 orig_eval=ROOT/'03_result/reevaluation_fullgap_20260929/evaluation';sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 changed=[m for m in others if sha(RUN/'evaluation'/m/'results.csv')!=sha(orig_eval/m/'results.csv')];sets=[('new root (current files)',site)]
 if changed:
  old=[]
  for m in changed:
   d=pd.read_csv(orig_eval/m/'results.csv').query("group_type=='all'").assign(model=m);assert len(d)==6205;old.append(d)
   inputs.append(dict(label=m,seed=None,role='comparator, original root (sensitivity only)',path=str(orig_eval/m/'results.csv'),sha256=sha(orig_eval/m/'results.csv'),modified=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime((orig_eval/m/'results.csv').stat().st_mtime)),rows_all=len(d),stored_model=m))
  sets.append(('original root for '+' and '.join(changed),pd.concat([site[~site.model.isin(changed)],ar.site_scores(pd.concat(old))],ignore_index=True)))
 sens=[]
 for tag,st in sets:
  pp=st.pivot(index='greenhouse',columns='model',values='NMAE');ov=st.groupby('model').NMAE.mean()
  for seed in ALL_SEEDS+['mean']:
   sp,lo_=label('SAITS-spatial',seed),label('SAITS-matched-local',seed)
   main=ar.compare(st,[(BITFI,sp if r=='SAITS-spatial' else r) for r in MAIN_REFERENCES]);comp=ar.compare(st,[(sp if a=='SAITS-spatial' else a,lo_ if b=='SAITS-matched-local' else b) for a,b in COMPONENT_PAIRS])
   field=[m for m in MAIN_MODELS if m!='SAITS-spatial']+[sp];order=ov[field].sort_values();lg=ar.compare(st,[(sp,'AG-LightGBM')]).iloc[0]
   sens.append(dict(comparators=tag,seed=seed,model=sp,NMAE=ov[sp],AG_LightGBM_NMAE=ov['AG-LightGBM'],rank_among_8_main=int(np.flatnonzero(order.index==sp)[0])+1,better_than_AG_LightGBM=bool(ov[sp]<ov['AG-LightGBM']),greenhouses_better_than_AG_LightGBM=int((pp[sp]<pp['AG-LightGBM']).sum()),reduction_vs_AG_LightGBM=lg.reduction,p_vs_AG_LightGBM=lg.p,holm_p_BiTFI_vs_SAITS_main=float(main[main.reference==sp].holm_p.iloc[0]),holm_p_reference_gain_component=float(comp[comp.model==sp].holm_p.iloc[0])))
 pd.DataFrame(sens).to_csv(ANALYSIS/'saits_seeds_comparator_sensitivity.csv',index=False)
 pd.DataFrame(inputs).astype({'seed':'Int64','rows_all':'Int64'}).to_csv(ANALYSIS/'saits_seeds_inputs.csv',index=False)

 # 5) Variable-level greenhouse means per run (descriptive).
 ar.summary(seeds,extra=['variable']).merge(seeds.drop_duplicates('model')[['model','seed']],on='model').to_csv(ANALYSIS/'saits_seeds_variables.csv',index=False)

 # 6) Reproduction: original seed-42 checkpoint versus a fresh seed-42 training run in this root.
 rep=[]
 for model in MODELS.values():
  o=RUN/'revision/saits_information'/model;n=REPRO/f'{model}_seed42';q=REPRO/f'{model}_seed42_original'
  so=torch.load(o/'best.pt',map_location='cpu',weights_only=True);sn=torch.load(n/'best.pt',map_location='cpu',weights_only=True)
  diffs=[float((so['state_dict'][k].double()-sn['state_dict'][k].double()).abs().max()) for k in so['state_dict']];ho=pd.read_csv(o/'history.csv');hn=pd.read_csv(n/'history.csv');common=min(len(ho),len(hn))
  a=seeds.query(f"model=='{label(model,42)}'").set_index(['case_id','variable']).sort_index();b=rerun.query(f"model=='{model}_seed42rerun'").set_index(['case_id','variable']).sort_index()
  orig_eval=pd.read_csv(RUN/'revision/saits_information/evaluation'/f'{model}.csv');recheck=pd.read_csv(q/'results.csv');cols=[c for c in orig_eval.columns if c!='model']
  po=np.load(RUN/'evaluation'/model/'predictions.npz')['prediction'];pn=np.load(n/'predictions.npz')['prediction'];pq=np.load(q/'predictions.npz')['prediction'];fin=np.isfinite(po)
  rep.append(dict(model=model,original_checkpoint_reevaluated_identical=bool(orig_eval[cols].equals(recheck[cols]) and np.array_equal(po,pq,equal_nan=True)),
   checkpoint_bitwise_identical=bool(all(torch.equal(so['state_dict'][k],sn['state_dict'][k]) for k in so['state_dict'])),max_abs_parameter_difference=max(diffs),
   epochs_original=len(ho),epochs_rerun=len(hn),best_epoch_original=int(so['epoch']),best_epoch_rerun=int(sn['epoch']),best_validation_mae_original=float(so['validation_mae']),best_validation_mae_rerun=float(sn['validation_mae']),
   history_max_abs_validation_mae_difference=float((ho.validation_mae[:common]-hn.validation_mae[:common]).abs().max()),history_max_abs_train_loss_difference=float((ho.train_loss[:common]-hn.train_loss[:common]).abs().max()),
   NMAE_original=float(runs.set_index('model').NMAE[f'{model}_seed42']),NMAE_rerun=float(runs.set_index('model').NMAE[f'{model}_seed42rerun']),NMAE_difference=float(runs.set_index('model').NMAE[f'{model}_seed42rerun']-runs.set_index('model').NMAE[f'{model}_seed42']),
   max_abs_case_NMAE_difference=float((a.NMAE-b.NMAE).abs().max()),cells_with_NMAE_difference=int((a.NMAE!=b.NMAE).sum()),max_abs_prediction_difference=float(np.abs(po[fin]-pn[fin]).max()),predictions_identical=bool(np.array_equal(po,pn,equal_nan=True)),
   rerun_train_seconds=json.loads((n/'complete.json').read_text())['elapsed_s'],rerun_device=json.loads((n/'complete.json').read_text())['device']))
 pd.DataFrame(rep).to_csv(ANALYSIS/'saits_seeds_reproduction.csv',index=False)
 pd.set_option('display.width',250);pd.set_option('display.max_columns',40)
 print(runs[['model','epochs_run','best_epoch','best_validation_mae','train_seconds','NMAE','NMAE_low','NMAE_high']].to_string(index=False));print(pd.DataFrame(spread).T.to_string());print(paired.to_string(index=False));print(pd.DataFrame(pspread).T.to_string());print(rank[['seed','model','NMAE','rank_among_8_main','better_than_TimesFM3.0-COV-SPA','greenhouses_better_than_TimesFM3.0-COV-SPA','better_than_AG-LightGBM','greenhouses_better_than_AG-LightGBM']].to_string(index=False));print(pd.DataFrame(rep).T.to_string())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','smoke','analyze']);a=p.parse_args()
 {'run':run,'smoke':smoke,'analyze':analyze}[a.action]()
