"""Load the release with explicit schema, hash and split checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def file_sha256(path):
    path=Path(path);content=path.read_bytes()
    if path.suffix.lower() in {'.csv','.json','.md'}:
        content=content.replace(b'\r\n',b'\n')
    return hashlib.sha256(content).hexdigest()


def load_data(root=ROOT):
    root=Path(root);data=root/'data'
    manifest=json.loads((data/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if file_sha256(data/name)!=digest:
            raise ValueError(f'Release checksum mismatch: {name}')
    tables={split:pd.read_csv(data/f'{split}.csv') for split in ['train','validation','test']}
    seen=set();features=manifest['features'];labels=manifest['labels']
    for split,table in tables.items():
        expected=['sample_id']+features+(['target'] if split!='test' else [])
        if list(table)!=expected or len(table)!=manifest['rows'][split]:raise ValueError('Schema/count mismatch')
        if not table.sample_id.is_unique or seen&set(table.sample_id):raise ValueError('Duplicate or overlapping sample IDs')
        seen.update(table.sample_id)
        if not np.isfinite(table[features].to_numpy()).all():raise ValueError('Nonfinite features')
        if split!='test' and set(table.target)!=set(labels):raise ValueError('Every declared crop is required')
    with np.load(data/'views/pca4.npz',allow_pickle=False) as f:view={k:f[k] for k in f.files}
    if 'y_test' in view:raise ValueError('Test labels must remain private')
    for split,table in tables.items():
        np.testing.assert_array_equal(view['ids_'+split],table.sample_id)
        if split!='test':np.testing.assert_array_equal(view['y_'+split],table.target)
    ids=json.loads((data/'starter_ids.json').read_text())
    starter=tables['train'].set_index('sample_id').index.get_indexer(ids)
    if len(set(ids))!=100 or (starter<0).any():raise ValueError('Starter must have 100 unique training rows')
    return tables,view,manifest,starter
