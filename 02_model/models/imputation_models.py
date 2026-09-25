"""Official SAITS / MOMENT adapters for retrospective masked-window imputation.

Neither adapter reads the runner's base cache. Artificial and natural missing
values are zeroed with explicit masks. MOMENT channels are separate batch items
because its observation mask has no channel axis.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
WINDOW = 512

def saits_network(device="cuda", window=WINDOW):
    sys.path.insert(0, str(ROOT / "third_party/SAITS"))
    from modeling.saits import SAITS
    return SAITS(
        n_groups=2, n_group_inner_layers=1, d_time=window, d_feature=5,
        d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1,
        input_with_mask=True, param_sharing_strategy="inner_group", MIT=True,
        diagonal_attention_mask=True, device=device,
    ).to(device)

def extract_window(masked, effective, artificial, window=WINDOW):
    positions = np.flatnonzero(artificial.to_numpy().any(axis=1))
    if not len(positions):
        raise ValueError("No artificial gap")
    start, end = int(positions[0]), int(positions[-1]) + 1
    if end - start > window:
        raise ValueError("Artificial gap exceeds model window")
    lo = max(0, min((start + end - window) // 2, len(masked) - window))
    hi = min(lo + window, len(masked))
    raw = masked.iloc[lo:hi].to_numpy(dtype=np.float32)
    observed = (effective.iloc[lo:hi].to_numpy() != 0) & np.isfinite(raw)
    observed &= ~artificial.iloc[lo:hi].to_numpy(dtype=bool)
    x = np.zeros((window, raw.shape[1]), dtype=np.float32)
    mask = np.zeros_like(x)
    x[:hi-lo] = np.where(observed, raw, 0)
    mask[:hi-lo] = observed
    valid_time = np.zeros(window, dtype=np.float32)
    valid_time[:hi-lo] = 1
    return x, mask, valid_time, lo, hi

class WindowImputer:
    uses_ext_covariates = False
    window = WINDOW

    def compute_base(self, data, valid_mask):
        # Interface marker only: this contains no observed values or ground truth.
        return pd.DataFrame(index=data.index, columns=data.columns, dtype=float)

    def impute_artificial(self, masked_data, mask_matrix, artificial_bool, base):
        x, mask, valid, lo, hi = extract_window(
            masked_data, mask_matrix, artificial_bool, self.window)
        pred = self.predict_window(x, mask, valid)
        if not np.isfinite(pred[:hi-lo][artificial_bool.iloc[lo:hi].to_numpy(dtype=bool)]).all():
            raise FloatingPointError(f"{self.name}: non-finite reconstruction")
        result = masked_data.copy()
        block = result.iloc[lo:hi].to_numpy(copy=True)
        target = artificial_bool.iloc[lo:hi].to_numpy(dtype=bool)
        block[target] = pred[:hi-lo][target]
        result.iloc[lo:hi] = block
        return result

class SAITSImputation(WindowImputer):
    name = "SAITS"

    def __init__(self, checkpoint: Path, device="cuda"):
        self.device = device
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.window = state["window"]
        self.model = saits_network(device, self.window)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict_window(self, x, mask, valid):
        inputs = {"X": torch.from_numpy(x[None]).to(self.device),
                  "missing_mask": torch.from_numpy(mask[None]).to(self.device)}
        return self.model.impute(inputs)[0][0].float().cpu().numpy()

class MOMENTImputation(WindowImputer):
    name = "MOMENT"

    def __init__(self, snapshot: Path, device="cuda"):
        sys.path.insert(0, str(ROOT / "third_party/moment"))
        from momentfm import MOMENTPipeline
        self.device = device
        self.model = MOMENTPipeline.from_pretrained(
            str(snapshot), model_kwargs={"task_name": "reconstruction"})
        self.model.init()
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.window = int(self.model.config.seq_len)
        if self.window != WINDOW:
            raise ValueError(f"Unexpected MOMENT window {self.window}")

    @torch.inference_mode()
    def predict_window(self, x, mask, valid):
        # Independent channel masks: (channels, 1, time), not one shared mask.
        tx = torch.from_numpy(x.T[:, None, :].copy()).to(self.device)
        tm = torch.from_numpy(mask.T.copy()).to(self.device)
        tv = torch.from_numpy(np.broadcast_to(valid, mask.T.shape).copy()).to(self.device)
        result = self.model(x_enc=tx, mask=tm, input_mask=tv)
        return result.reconstruction[:, 0, :].float().cpu().numpy().T

class MOMENTFineTunedImputation(MOMENTImputation):
    name = "MOMENT-FT"

    def __init__(self, snapshot: Path, checkpoint: Path, device="cuda"):
        super().__init__(snapshot, device=device)
        if not (checkpoint.parent / "complete.json").exists():
            raise RuntimeError("MOMENT head training is incomplete")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["epoch"] <= 0:
            raise RuntimeError("Selected head is unchanged from zero-shot")
        self.model.head.load_state_dict(state["head_state_dict"], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)


class ClassicalImputation(WindowImputer):
    """Reuse existing trained models, with ctx720 inference and ctx1440 masks."""
    window = 720
    def __init__(self, name):
        self.name = name
        if name == "LI":
            from models.linear_interpolation import LinearInterpolation
            self.model = LinearInterpolation()
        elif name == "SeasonalNaive":
            from models.seasonal_naive import SeasonalNaiveImputation
            self.model = SeasonalNaiveImputation(season_length=24, context_len=720)
        else:
            from models.autogluon_model import AutoGluonImputation
            self.model = AutoGluonImputation.load(
                ROOT/"03_result/comparison/models/AG-LightGBM",
                context_len=720, name="AG-LightGBM")
            from functools import partial
            self.model.predictor.predict = partial(self.model.predictor.predict, use_cache=False)
    def compute_base(self, data, valid_mask):
        if hasattr(self.model,"compute_base"):
            return self.model.compute_base(data,valid_mask)
        return super().compute_base(data,valid_mask)
    def impute_artificial(self, masked_data, mask_matrix, artificial_bool, base):
        if hasattr(self.model,"impute_artificial"):
            return self.model.impute_artificial(masked_data,mask_matrix,artificial_bool,base)
        return self.model.impute(masked_data,mask_matrix)
