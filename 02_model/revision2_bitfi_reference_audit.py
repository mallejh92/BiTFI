"""CPU audit for the BiTFI reference-selection controls (revision 2, E2).

1. Recomputes the main-run reference selection (NeighborBank.select, k=3) for every
   scored variable-case under the masked input, exactly as infer_stages calls it.
2. Checks that ReferenceBankVariant(mode='topk') returns identical arrays and names.
3. Checks the stored main-run audit (quality/reference_availability.csv, read-only).
4. Fixes the five facilities that supplied most reference slots (exclTop5 list).
5. Records, per variable-case, the eligible ranking and the random3 / exclTop5 picks.
6. Records registry-key collisions (identical availability vectors) and checks none share a run chunk.
"""
import json, time, hashlib
import numpy as np, pandas as pd
import clean_protocol as cp
from revision2_reference_bank import ReferenceBankVariant, observed_key

OUT = cp.OUT / 'revision/bitfi_variants'
ORIGINAL = cp.ROOT / '03_result/reevaluation_fullgap_20260929'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bank = cp.CleanNeighborBank()
    manifest = pd.read_csv(cp.OUT / 'mask_manifest.csv')
    sites = {n: cp.site(n) for n in manifest.greenhouse.unique()}
    topk = ReferenceBankVariant(bank, 'topk', variables=cp.COLS)
    rand = ReferenceBankVariant(bank, 'random', variables=cp.COLS)
    records = []; keys = {}
    for i, r in enumerate(manifest.itertuples()):
        o = sites[r.greenhouse]; mr = cp.masked_case(o, r)
        masked = mr.masked_data[cp.COLS]
        observed = mr.effective_mask[cp.COLS].ne(0).values & np.isfinite(masked.values)
        obs = masked.values.astype(np.float32)
        for v in r.masked_vars.split(','):
            ci = cp.COLS.index(v)
            key = observed_key(v, masked.index, observed[:, ci])
            keys.setdefault(key, []).append(int(r.case_id))
            for w in (topk, rand):
                w.clear(); w.register(r.case_id, r.greenhouse, v, masked.index, observed[:, ci])
            nb, names = bank.select(v, masked.index, obs[:, ci], observed[:, ci], k=3)
            nb2, names2 = topk.select(v, masked.index, obs[:, ci], observed[:, ci], k=3)
            assert names == names2 and nb.shape == nb2.shape and np.array_equal(nb, nb2, equal_nan=True), (r.case_id, v)
            log = topk.take_log()[0]
            _, scores, status = topk.rank(v, masked.index, obs[:, ci], observed[:, ci])
            rand.select(v, masked.index, obs[:, ci], observed[:, ci], k=3); rlog = rand.take_log()[0]
            records.append(dict(case_id=int(r.case_id), greenhouse=r.greenhouse, scenario=r.scenario,
                                gap_length_h=int(r.gap_length_h), variable=v, status=status,
                                n_coverage_candidates=log['n_coverage_candidates'], n_eligible=log['n_eligible'],
                                main_references=','.join(names), main_abs_r=log['abs_r'], main_mean_abs_r=log['mean_abs_r'],
                                eligible_ranking=';'.join(f'{n}:{s:.6f}' for s, n in scores),
                                random3_references=rlog['references'], random3_mean_abs_r=rlog['mean_abs_r'],
                                random3_ranks=rlog['eligible_ranks']))
        if i % 500 == 0:
            print(i, '/', len(manifest), round(time.time() - t0), 's', flush=True)
    d = pd.DataFrame(records)
    collisions = {f'{k[0]}|{k[2]}|{k[4][:12]}': ids for k, ids in keys.items() if len(ids) > 1}
    stored = pd.read_csv(ORIGINAL / 'quality/reference_availability.csv')
    m = d.merge(stored, on=['case_id', 'variable'], validate='one_to_one')
    assert len(m) == len(d) == len(stored)
    same = (m.main_references == m.references.fillna('')).all()
    slots = d.main_references.str.split(',').explode()
    slots = slots[slots != '']
    counts = slots.value_counts()
    top5 = list(counts.index[:5])
    share = float(counts.iloc[:5].sum() / counts.sum())
    # exclTop5 picks recomputed from the stored eligible ranking (top-3 after exclusion)
    def excl(ranking):
        pairs = [x.split(':') for x in ranking.split(';') if x]
        kept = [(n, float(s)) for n, s in pairs if n not in top5]
        return kept[:3], len(kept)
    ex = d.eligible_ranking.map(excl)
    d['exclTop5_references'] = ex.map(lambda z: ','.join(n for n, _ in z[0]))
    d['exclTop5_mean_abs_r'] = ex.map(lambda z: float(np.mean([s for _, s in z[0]])) if z[0] else np.nan)
    d['exclTop5_n_pool'] = ex.map(lambda z: z[1])
    d.to_csv(OUT / 'reference_audit.csv', index=False)
    info = dict(variable_cases=len(d), cases=int(d.case_id.nunique()), reference_slots=int(counts.sum()),
                n_references_distribution={str(k): int(v) for k, v in d.main_references.map(lambda s: len([x for x in s.split(',') if x])).value_counts().sort_index().items()},
                top5_facilities=top5, top5_slot_counts={n: int(counts[n]) for n in top5}, top5_slot_share=share,
                all_slot_counts={n: int(c) for n, c in counts.items()},
                stored_main_audit=str(ORIGINAL / 'quality/reference_availability.csv'),
                stored_main_audit_sha256=hashlib.sha256((ORIGINAL / 'quality/reference_availability.csv').read_bytes()).hexdigest(),
                recomputed_equals_stored_audit=bool(same),
                wrapper_topk_identical_to_NeighborBank_select=True,
                registry_key_collisions=collisions,
                min_eligible=int(d.n_eligible.min()), min_exclTop5_pool=int(d.exclTop5_n_pool.min()),
                exclTop5_cases_below_3=int((d.exclTop5_n_pool < 3).sum()),
                seconds=time.time() - t0)
    # Identical availability vectors occur when a single-variable mask and a multi-variable mask share
    # the gap location. Selection inputs are then identical; the registry is per 16-case chunk, so a
    # duplicate key is only a problem inside one chunk (2 shards by case_id parity, 16 rows per chunk).
    chunk_of = {}
    for s in range(2):
        ids = manifest.case_id[manifest.case_id % 2 == s].tolist()
        for j, c in enumerate(ids):
            chunk_of[c] = (s, j // 16)
    info['registry_key_collisions_within_a_shard_chunk'] = [ids for ids in collisions.values() if len({chunk_of[c] for c in ids}) < len(ids)]
    (OUT / 'top5_reference_facilities.json').write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != 'all_slot_counts'}, indent=2), flush=True)
    assert same and not info['registry_key_collisions_within_a_shard_chunk']


if __name__ == '__main__':
    main()
