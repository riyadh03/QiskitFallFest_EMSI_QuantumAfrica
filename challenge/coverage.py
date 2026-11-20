"""Reproducible training-size checks; final test answers stay with organizers."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from .data import ROOT,load_data,file_sha256
from .classical import reference
from .models import QuantumClassifier,MatchedHead
from .metrics import score

LABELS=['alfalfa','maize','wheat']
MODELS=['reference','matched_head','qnn']


def training_rows(size=None,extension_dir=None):
    tables,_,manifest,_=load_data();base=tables['train'];train=base
    if extension_dir:
        folder=Path(extension_dir);e=json.loads((folder/'manifest.json').read_text())
        if e['labels']!=LABELS or e['features']!=manifest['features']:raise ValueError('Extension schema differs')
        if file_sha256(folder/'train.csv')!=e['files']['train.csv']:raise ValueError('Extension checksum mismatch')
        for split in ['validation','test']:
            if e[f'{split}_sha256']!=manifest['files'][f'{split}.csv']:raise ValueError('Evaluation split changed')
        audit=e['organizer_audit']
        if not (audit['approved'] and audit['physical_groups_disjoint'] and audit['spatial_components_disjoint'] and audit['min_distance_m']>=100 and audit['max_calendar_gap_days']<=2 and audit['evidence_reference']):
            raise ValueError('Missing organizer audit under the accepted date/spatial policy')
        train=pd.read_csv(folder/'train.csv')
        if list(train)!=list(base) or not train.sample_id.is_unique or set(train.target)!=set(LABELS):raise ValueError('Invalid extension table')
        if not np.isfinite(train[manifest['features']].to_numpy()).all():raise ValueError('Nonfinite extension')
        for split in ['validation','test']:
            if set(train.sample_id)&set(tables[split].sample_id):raise ValueError('Evaluation fields entered training')
        if not set(base.sample_id)<=set(train.sample_id):raise ValueError('Original training fields removed')
        pd.testing.assert_frame_equal(train.set_index('sample_id').loc[base.sample_id].reset_index(),base,check_exact=True)
    size=len(train) if size is None else size
    if size in [100,200,600]:
        ids=json.loads((ROOT/'data/training_600_ids.json').read_text())
        train=train.set_index('sample_id').loc[ids].reset_index()
        if size<600:
            parts=[train[train.target==label].sample(size//3+(i<size%3),random_state=0) for i,label in enumerate(LABELS)]
            train=pd.concat(parts,ignore_index=True)
    elif size!=len(train):raise ValueError('Use 100, 200, 600, or the complete audited training size')
    return train,tables['validation'],manifest


def train(size,results,extension_dir=None,layers=1,hidden_units=1,seeds=None):
    results=Path(results)
    if results.resolve()==(ROOT/'results').resolve():raise ValueError('Supplied reference results cannot be overwritten; use outputs/')
    t,v,m=training_rows(size,extension_dir);F=m['features'];results.mkdir(parents=True,exist_ok=True)
    prep=make_pipeline(StandardScaler(),PCA(4,svd_solver='full'),StandardScaler())
    start=perf_counter();A=prep.fit_transform(t[F].to_numpy());B=prep.transform(v[F].to_numpy());prep_time=perf_counter()-start
    records=[]
    training_digest=hashlib.sha256(t.to_csv(index=False).encode()).hexdigest()
    run_seeds = [0, 1, 2] if seeds is None or len(seeds) == 0 else seeds
    for seed in run_seeds:
        for name in ['dummy_majority','dummy_stratified']+MODELS:
            if name=='qnn':model=QuantumClassifier(LABELS,encoding='ry',max_evals=100,layers=layers);X,D=A,B
            elif name=='matched_head':model=MatchedHead(LABELS,hidden_units=hidden_units,max_evals=100);X,D=A,B
            elif name=='reference':model=reference(seed);X,D=t[F].to_numpy(),v[F].to_numpy()
            else:model=DummyClassifier(strategy='most_frequent' if name=='dummy_majority' else 'stratified',random_state=seed);X,D=t[F].to_numpy(),v[F].to_numpy()
            start=perf_counter()
            if name in ['qnn','matched_head']:model.fit(X,t.target.to_numpy(),seed)
            else:model.fit(X,t.target)
            fit=perf_counter()-start;start=perf_counter();pred=model.predict(D);predict=perf_counter()-start
            records.append({'model':name,'seed':seed,'training_rows':len(t),'training_sha256':training_digest,
                'validation_sha256':m['files']['validation.csv'],'metrics':score(v.target,pred,LABELS),
                'fit_seconds':fit,'predict_seconds':predict,'preprocessing_seconds':prep_time if name in ['qnn','matched_head'] else 0,
                'resources':model.resources() if hasattr(model,'resources') else {}})
            pd.DataFrame({'sample_id':v.sample_id,'target':v.target,'prediction':pred}).to_csv(results/f'{name}_{seed}_predictions.csv',index=False)
            if hasattr(model,'history'):(results/f'{name}_{seed}_history.json').write_text(json.dumps(model.history,indent=2))
            print(len(t),name,seed,records[-1]['metrics']['macro_f1'],flush=True)
    (results/'runs.json').write_text(json.dumps(records,indent=2))


def metric_vector(y,pred,weights):
    matrices=np.array([np.bincount(3*y+p,weights=weights,minlength=9).reshape(3,3) for p in pred])
    tp=np.diagonal(matrices,axis1=1,axis2=2);support=matrices.sum(axis=2);predicted=matrices.sum(axis=1)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support!=0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros_like(tp),where=(support+predicted)!=0)
    return np.r_[f1.mean(),(tp.sum(axis=1)/matrices.sum(axis=(1,2))).mean(),recall.mean(axis=0)]


def compare(baseline,candidate,samples=2000):
    tables,_,manifest,_=load_data();v=tables['validation'];order=v.sample_id.tolist();encoding={s:i for i,s in enumerate(LABELS)}
    y=v.target.map(encoding).to_numpy();folders=[Path(baseline),Path(candidate)]
    groups=json.loads((ROOT/'data/validation_groups.json').read_text());values=np.array([groups[s] for s in order]);unique=np.unique(values);index=np.searchsorted(unique,values)
    rng=np.random.default_rng(2026);weights=[]
    while len(weights)<samples:
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[index]
        if all(w[y==k].sum()>0 for k in range(3)):weights.append(w)
    report={'policy':'Validation comparison only; report all classes and freeze settings before final test prediction.','models':{}}
    names=['macro_f1','accuracy','recall_alfalfa','recall_maize','recall_wheat']
    for model in MODELS:
        predictions=[]
        for folder in folders:
            records=json.loads((folder/'runs.json').read_text());selected=[r for r in records if r['model']==model]
            if sorted(r['seed'] for r in selected)!=[0,1,2]:raise ValueError('Three complete seeds required')
            if any(r['validation_sha256']!=manifest['files']['validation.csv'] for r in selected):raise ValueError('Validation checksum differs')
            seeds=[]
            for seed in [0,1,2]:
                p=pd.read_csv(folder/f'{model}_{seed}_predictions.csv')
                if not p.sample_id.is_unique or set(p.sample_id)!=set(order):raise ValueError('Prediction IDs differ')
                p=p.set_index('sample_id').loc[order]
                if p.target.tolist()!=v.target.tolist() or not set(p.prediction)<=set(LABELS):raise ValueError('Prediction labels differ')
                seeds.append(p.prediction.map(encoding).to_numpy())
            predictions.append(np.array(seeds))
        delta=metric_vector(y,predictions[1],np.ones(len(y)))-metric_vector(y,predictions[0],np.ones(len(y)))
        boot=np.array([metric_vector(y,predictions[1],w)-metric_vector(y,predictions[0],w) for w in weights])
        report['models'][model]={'delta':dict(zip(names,delta)),'one_sided_95_lower':dict(zip(names,np.quantile(boot,.05,axis=0)))}
    (folders[1]/'comparison.json').write_text(json.dumps(report,indent=2));return report


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('check');check.add_argument('--extension-dir',type=Path)
    fit=sub.add_parser('train')
    fit.add_argument('--size',type=int)
    fit.add_argument('--extension-dir',type=Path)
    fit.add_argument('--results',type=Path,required=True)
    fit.add_argument('--layers',type=int,default=1,help='Data re-uploading layers L (default 1)')
    fit.add_argument('--hidden-units',type=int,default=1,help='Classical MLP control hidden units h (default 1)')
    fit.add_argument('--seeds',type=int,nargs='*',default=[0,1,2],help='Seeds to run (default: 0 1 2)')
    comp=sub.add_parser('compare');comp.add_argument('--baseline',type=Path,required=True);comp.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='check':
        t,v,m=training_rows(extension_dir=args.extension_dir);print(json.dumps({'train':len(t),'validation':len(v),'test':m['rows']['test'],'temporal_policy':m['temporal_policy'],'file_checks':'passed'},indent=2))
    elif args.command=='train':train(args.size,args.results,args.extension_dir,layers=args.layers,hidden_units=args.hidden_units,seeds=args.seeds)
    else:print(json.dumps(compare(args.baseline,args.candidate),indent=2))


if __name__=='__main__':main()
