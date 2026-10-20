"""Reference-selection controls for BiTFI that wrap an existing NeighborBank.

The wrapper is a drop-in replacement for ``model.bank`` inside
``bitfi_refinement_ablation.infer_stages``. ``select`` keeps the signature and the
return format of ``spatial.NeighborBank.select``: a (k, T) float32 array aligned to
the target timestamps (missing reference values interpolated, then edge-filled)
and the list of selected greenhouse names, or ``(0, T)`` and ``[]``.

Eligibility is exactly that of ``NeighborBank.select``:
  * candidates are training greenhouses with the variable and >=80% coverage of
    the target timestamps (``bank.aligned``; same timestamps, no shifting);
  * the target needs >=500 available (masked-input) observations;
  * a candidate needs >=500 paired finite observations and a finite Pearson r;
  * the target greenhouse itself is never a candidate (test targets are not in
    the training bank, so this is asserted rather than relied upon).

Modes
  topk    : the original rule, top-k by |r| (ties: name descending).
  exclude : drop the names in ``exclude`` from the eligible list, then top-k.
  random  : draw k eligible candidates uniformly without replacement with a seed
            fixed per mask and variable, ``default_rng([seed_base, case_id, var])``.
            Picked names are returned in descending |r| order, the same row
            convention as the original rule.

Every call must be registered (case id, greenhouse) through ``register``; the key
is the variable plus a hash of the masked-input availability vector, which is
invariant to hidden target values. Each call is appended to ``log``.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from spatial import DEFAULT_K, MIN_FIT_POINTS

MODES = ("topk", "exclude", "random")
SEED_BASE = 20261019


def observed_key(var: str, index: pd.Index, observed: np.ndarray) -> tuple:
    """Key of one selection call: variable, target time axis and availability pattern."""
    o = np.ascontiguousarray(np.asarray(observed, dtype=bool))
    first = str(index[0]) if len(index) else ""
    last = str(index[-1]) if len(index) else ""
    return (var, len(index), first, last, hashlib.sha1(o.tobytes()).hexdigest())


class ReferenceBankVariant:
    def __init__(self, bank, mode: str = "topk", exclude=(), seed_base: int = SEED_BASE,
                 variables: list[str] | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.bank = bank
        self.sites = bank.sites
        self.target_vars = list(getattr(bank, "target_vars", variables or []))
        self.mode = mode
        self.exclude = tuple(sorted(exclude))
        unknown = set(self.exclude) - set(self.sites)
        if unknown:
            raise ValueError(f"excluded names are not training references: {sorted(unknown)}")
        self.seed_base = int(seed_base)
        self.variables = list(variables) if variables is not None else self.target_vars
        self._cases: dict[tuple, dict] = {}
        self.log: list[dict] = []

    # NeighborBank compatibility
    def aligned(self, var: str, index: pd.Index) -> pd.DataFrame:
        return self.bank.aligned(var, index)

    # Case registry
    def register(self, case_id: int, greenhouse: str, var: str, index: pd.Index,
                 observed: np.ndarray, **extra) -> tuple:
        if greenhouse in self.sites:
            raise ValueError(f"target greenhouse {greenhouse} is a training reference")
        key = observed_key(var, index, observed)
        info = dict(case_id=int(case_id), greenhouse=greenhouse, variable=var, **extra)
        old = self._cases.get(key)
        if old is not None and old["case_id"] != info["case_id"]:
            raise RuntimeError(f"identical selection inputs for cases {old['case_id']} and {case_id} ({var})")
        self._cases[key] = info
        return key

    def clear(self) -> None:
        self._cases.clear()

    def take_log(self) -> list[dict]:
        out, self.log = self.log, []
        return out

    # Ranking with the original eligibility rule
    def rank(self, var: str, index: pd.Index, target: np.ndarray, observed: np.ndarray):
        """Return (aligned candidates, [(|r|, name)] sorted as NeighborBank.select, status)."""
        cand = self.bank.aligned(var, index)
        if cand.empty:
            return cand, [], "no_candidate_with_80pct_coverage"
        fit = observed & np.isfinite(target)
        if fit.sum() < MIN_FIT_POINTS:
            return cand, [], "target_below_500_observations"
        y = pd.Series(target[fit])
        scores: list[tuple[float, str]] = []
        for name in cand.columns:
            x = cand[name].values[fit]
            m = np.isfinite(x)
            if m.sum() < MIN_FIT_POINTS:
                continue
            r = pd.Series(x[m]).corr(y[m])
            if np.isfinite(r):
                scores.append((abs(float(r)), name))
        scores.sort(reverse=True)
        return cand, scores, ("ok" if scores else "no_candidate_with_500_pairs")

    def select(self, var: str, index: pd.Index, target: np.ndarray, observed: np.ndarray,
               k: int = DEFAULT_K) -> tuple[np.ndarray, list[str]]:
        key = observed_key(var, index, observed)
        info = self._cases.get(key)
        if info is None:
            raise KeyError(f"unregistered reference selection call for {var}")
        cand, scores, status = self.rank(var, index, target, observed)
        eligible = [(r, n) for r, n in scores if n != info["greenhouse"]]
        pool = [(r, n) for r, n in eligible if n not in self.exclude]
        if self.mode == "random":
            rng = np.random.default_rng([self.seed_base, info["case_id"], self.variables.index(var)])
            names = sorted(n for _, n in pool)
            take = min(int(k), len(names))
            chosen = {names[i] for i in rng.choice(len(names), size=take, replace=False)} if take else set()
            picked_scores = [(r, n) for r, n in pool if n in chosen]
        else:
            picked_scores = pool[:int(k)]
        picked = [n for _, n in picked_scores]
        rank_of = {n: i + 1 for i, (_, n) in enumerate(eligible)}
        top = eligible[:DEFAULT_K]
        self.log.append(dict(
            **info, mode=self.mode, k=int(k), status=status,
            n_coverage_candidates=int(cand.shape[1]), n_eligible=len(eligible),
            n_excluded_eligible=len(eligible) - len(pool), n_pool=len(pool),
            n_references=len(picked), shortfall=len(picked) < int(k),
            references=",".join(picked),
            abs_r=";".join(f"{r:.6f}" for r, _ in picked_scores),
            eligible_ranks=";".join(str(rank_of[n]) for n in picked),
            mean_abs_r=float(np.mean([r for r, _ in picked_scores])) if picked else np.nan,
            top3_references=",".join(n for _, n in top),
            top3_mean_abs_r=float(np.mean([r for r, _ in top])) if top else np.nan,
            seed=(f"[{self.seed_base},{info['case_id']},{self.variables.index(var)}]" if self.mode == "random" else ""),
        ))
        if not picked:
            return np.zeros((0, len(index)), dtype=np.float32), []
        arr = cand[picked].interpolate(limit_direction="both").ffill().bfill()
        return arr.values.T.astype(np.float32), picked
