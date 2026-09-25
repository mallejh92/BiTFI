"""
patchtst_model.py
PatchTST 기반 결측 보간 모델 (v2)

참고 논문: Nie et al. (ICLR 2023)
  "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

v2 개선 사항 (LSTM 대비 취약점 보완):
  (1) Mask-aware 입력: concat(patch * obs_mask, obs_mask) → 2P 입력
      · 패치 내 부분 관측 정보를 보존 — 기존 "패치 전체 0" 방식 폐기
      · Individual masking에서 컨텍스트 소실 문제 해결
  (2) patch_len 16 → 8: 개별 결측 해상도 향상
  (3) 2-term loss: masked_MSE + recon_weight × observed_MSE
  (4) Per-patch bias correction: LSTM의 window-level 보정을 패치 단위로 적용
  (5) Gradient clipping: Transformer 학습 안정화

설계 원칙 (유지):
  · Channel-independence: 각 변수 독립 처리, Transformer 가중치 공유
  · Instance Normalization: 윈도우별 z-score 정규화
  · 오버랩 슬라이딩 윈도우 추론 → 겹치는 구간 평균 집계
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 구성 블록
# ──────────────────────────────────────────────

class _PatchEmbedding(nn.Module):
    """패치 입력(2P 차원)을 d_model 차원으로 선형 투영"""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, input_dim) → (B, N, D)
        return self.dropout(self.proj(x))


class _LearnablePositionalEncoding(nn.Module):
    """학습 가능한 위치 인코딩 — 논문의 Wpos"""

    def __init__(self, max_n_patches: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_n_patches, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        N   = x.size(1)
        pos = torch.arange(N, device=x.device)
        return x + self.pe(pos)  # (B, N, D)


class _PatchTSTNet(nn.Module):
    """
    PatchTST Transformer 인코더 + 재구성 헤드 (v2).

    입력 : (B, N, 2P) — concat(patch * obs_mask, obs_mask)
           obs_mask = 1이면 관측, 0이면 결측 (훈련 시 마스킹된 패치는 0으로 설정)
    출력 : (B, N, P) — 원본 패치 재구성
    """

    def __init__(
        self,
        patch_len:     int,
        max_n_patches: int,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        d_ff:          int   = 128,
        n_layers:      int   = 3,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.patch_len = patch_len

        # 입력 차원: patch_len * 2 (data + obs_mask)
        self.patch_embedding = _PatchEmbedding(patch_len * 2, d_model, dropout)
        self.pos_encoding    = _LearnablePositionalEncoding(max_n_patches, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 재구성 헤드 D → P
        self.head = nn.Linear(d_model, patch_len)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # x_in: (B, N, 2P) — 이미 마스킹 정보 포함
        z = self.patch_embedding(x_in)  # (B, N, D)
        z = self.pos_encoding(z)        # (B, N, D)
        z = self.encoder(z)             # (B, N, D)
        return self.head(z)             # (B, N, P)


# ──────────────────────────────────────────────
# Dataset — Mask-aware 자기지도 마스킹 재구성
# ──────────────────────────────────────────────

class _PatchTSTDataset(Dataset):
    """
    Mask-aware 자기지도 마스킹 재구성 Dataset.

    각 샘플 처리:
      1. 단일 변수(channel-independence)의 길이 W 윈도우 추출
      2. Instance Normalization
      3. 비오버랩 패치 분할: N = W // P 패치
      4. mask_ratio 비율의 패치를 obs_mask=0으로 마스킹
      5. x_in = concat(patches * obs_mask, obs_mask) → (N, 2P)
    """

    def __init__(
        self,
        data:           np.ndarray,  # (T, F) 전처리된 시계열
        context_window: int,
        patch_len:      int,
        mask_ratio:     float = 0.4,
        mask_type:      str   = "mixed",   # 'random' | 'block' | 'mixed'
        random_seed:    int   = 42,
    ):
        self.data    = data.astype(np.float32)
        self.W       = context_window
        self.P       = patch_len
        self.N       = context_window // patch_len   # 비오버랩 패치 수
        self.T, self.F = data.shape
        self.mask_ratio = mask_ratio
        self.mask_type  = mask_type
        self.rng     = np.random.RandomState(random_seed)

        self.n_win = max(0, self.T - self.W + 1)

    def __len__(self) -> int:
        return self.n_win * self.F

    def __getitem__(self, idx: int):
        var_idx = idx % self.F
        win_idx = idx // self.F

        # 단일 변수 윈도우 추출 (W,)
        series = self.data[win_idx : win_idx + self.W, var_idx].copy()

        # Instance Normalization
        mu  = series.mean()
        std = series.std() + 1e-8
        series = (series - mu) / std

        # 비오버랩 패치 생성: (N, P)
        patches = series[: self.N * self.P].reshape(self.N, self.P)

        # 훈련 데이터는 clean → obs_mask 초기값은 모두 1
        obs_mask = np.ones((self.N, self.P), dtype=np.float32)

        # 마스킹 전략 선택
        # 'random': 랜덤 패치 마스킹 (원래 방식)
        # 'block':  연속 블록 마스킹 (테스트 분포 일치)
        # 'mixed':  샘플마다 랜덤/블록 혼합 → 일반화 향상
        use_block = (
            self.mask_type == "block" or
            (self.mask_type == "mixed" and self.rng.rand() < 0.5)
        )

        if use_block:
            # 테스트 갭 길이 [6,12,24,72,168]h를 patch 단위로 변환해 샘플
            # patch_len P 기준: [ceil(6/P), ceil(12/P), ceil(24/P), ceil(72/P), ceil(168/P)]
            P = self.P
            gap_patches = [max(1, int(np.ceil(h / P))) for h in [6, 12, 24, 72, 168]]
            gap_patches = [g for g in gap_patches if g <= self.N]
            if not gap_patches:
                gap_patches = [1]
            block_len   = int(self.rng.choice(gap_patches))
            block_start = int(self.rng.randint(0, max(1, self.N - block_len + 1)))
            mask_idx    = np.arange(block_start, min(block_start + block_len, self.N))
        else:
            n_mask   = max(1, int(self.N * self.mask_ratio))
            mask_idx = self.rng.choice(self.N, size=n_mask, replace=False)

        patch_mask = np.zeros(self.N, dtype=bool)
        patch_mask[mask_idx] = True
        obs_mask[mask_idx]   = 0.0

        # 모델 입력: concat(patches * obs_mask, obs_mask) → (N, 2P)
        x_in = np.concatenate([patches * obs_mask, obs_mask], axis=-1).astype(np.float32)

        return (
            torch.tensor(x_in,       dtype=torch.float32),  # (N, 2P)
            torch.tensor(patches,    dtype=torch.float32),   # (N, P) 재구성 타깃
            torch.tensor(patch_mask, dtype=torch.bool),      # (N,)  손실 마스크
        )


# ──────────────────────────────────────────────
# 메인 모델 클래스
# ──────────────────────────────────────────────

class PatchTSTImputation:
    """
    PatchTST 기반 결측 보간 모델 v2 (Nie et al., ICLR 2023).

    v2 핵심 변경:
      · Mask-aware 입력: concat(data*obs_mask, obs_mask) — 부분 관측 정보 보존
      · patch_len=8: 개별 결측 해상도 향상
      · 2-term loss: masked_MSE + recon_weight × observed_MSE
      · Per-patch bias correction: 관측 타임스텝 기준 예측 편향 제거
      · Gradient clipping (max_norm=1.0): 학습 안정화
    """

    name = "PatchTST"

    def __init__(
        self,
        n_features:     int   = 5,
        patch_len:      int   = 24,   # 24h = 1일 주기 정렬 (이전: 8)
        d_model:        int   = 128,  # 용량 확대 (이전: 64)
        n_heads:        int   = 4,
        d_ff:           int   = 256,  # 용량 확대 (이전: 128)
        n_layers:       int   = 3,
        dropout:        float = 0.2,
        mask_ratio:     float = 0.4,
        mask_type:      str   = "mixed",  # 'random'|'block'|'mixed' (이전: random만)
        recon_weight:   float = 0.3,
        epochs:         int   = 100,
        patience:       int   = 15,   # 조기종료 여유 확대 (이전: 10)
        val_ratio:      float = 0.2,
        batch_size:     int   = 32,
        lr:             float = 5e-4,
        context_window: int   = 720,
        random_seed:    int   = 42,
    ):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.n_features     = n_features
        self.patch_len      = patch_len
        self.d_model        = d_model
        self.n_heads        = n_heads
        self.d_ff           = d_ff
        self.n_layers       = n_layers
        self.dropout        = dropout
        self.mask_ratio     = mask_ratio
        self.mask_type      = mask_type
        self.recon_weight   = recon_weight
        self.epochs         = epochs
        self.patience       = patience
        self.val_ratio      = val_ratio
        self.batch_size     = batch_size
        self.lr             = lr
        self.context_window = context_window
        self.random_seed    = random_seed
        self.model: _PatchTSTNet | None = None
        self._ctx_window: int = context_window

    # ── fit ──────────────────────────────────────

    def fit(self, train_data: pd.DataFrame) -> None:
        if len(train_data) == 0:
            print("  [PatchTST] 훈련 데이터 없음 → 스킵")
            self.model = None
            return

        # NaN(원본 결측) → 선형 보간으로 채움
        data = (
            train_data.copy()
            .interpolate(method="linear", limit_direction="both")
            .ffill().bfill().fillna(0.0)
            .values.astype(np.float32)
        )
        self.n_features = data.shape[1]
        T = len(data)
        P = self.patch_len

        # context_window를 데이터 크기에 맞게 조정 (최소 2 패치)
        W = min(self.context_window, (T // (2 * P)) * P)
        W = max(W, P * 2)
        self._ctx_window = W
        N = W // P

        # Temporal validation split
        min_val = W * 2
        n_val   = max(min_val, int(T * self.val_ratio))
        if T > n_val + W * 2:
            train_arr = data[:-n_val]
            val_arr   = data[-n_val:]
        else:
            train_arr = data
            val_arr   = None

        train_ds = _PatchTSTDataset(train_arr, W, P, self.mask_ratio, self.mask_type, self.random_seed)
        if len(train_ds) == 0:
            print("  [PatchTST] 유효 훈련 샘플 없음 → 스킵")
            self.model = None
            return

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        val_loader = None
        if val_arr is not None and len(val_arr) >= W:
            val_ds = _PatchTSTDataset(
                val_arr, W, P, self.mask_ratio, self.mask_type, self.random_seed + 1
            )
            if len(val_ds) > 0:
                val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # 모델 초기화 (입력 차원: patch_len * 2)
        self.model = _PatchTSTNet(
            patch_len=P,
            max_n_patches=N,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction="none")

        best_val     = float("inf")
        patience_cnt = 0
        best_state   = None

        for epoch in range(self.epochs):
            # ── Train ──
            self.model.train()
            total_loss, n_b = 0.0, 0

            for x_in, target, patch_mask in train_loader:
                x_in       = x_in.to(DEVICE)        # (B, N, 2P)
                target     = target.to(DEVICE)       # (B, N, P)
                patch_mask = patch_mask.to(DEVICE)   # (B, N)

                pred     = self.model(x_in)           # (B, N, P)
                loss_raw = criterion(pred, target)    # (B, N, P)

                # 1) 마스킹된 패치에 대한 재구성 손실 (primary)
                mask_exp = patch_mask.unsqueeze(-1).expand_as(loss_raw)
                n_masked = mask_exp.sum().clamp(min=1)
                loss_masked = (loss_raw * mask_exp).sum() / n_masked

                # 2) 관측된 패치에 대한 재구성 손실 (학습 안정화)
                obs_exp = (~patch_mask).unsqueeze(-1).expand_as(loss_raw)
                n_obs   = obs_exp.sum().clamp(min=1)
                loss_recon = (loss_raw * obs_exp).sum() / n_obs

                loss = loss_masked + self.recon_weight * loss_recon

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                n_b += 1

            train_loss = total_loss / max(n_b, 1)

            # ── Validation ──
            if val_loader is not None:
                self.model.eval()
                total_val, n_v = 0.0, 0
                with torch.no_grad():
                    for x_in, target, patch_mask in val_loader:
                        x_in       = x_in.to(DEVICE)
                        target     = target.to(DEVICE)
                        patch_mask = patch_mask.to(DEVICE)
                        pred       = self.model(x_in)
                        loss_raw   = criterion(pred, target)
                        mask_exp   = patch_mask.unsqueeze(-1).expand_as(loss_raw)
                        n_masked   = mask_exp.sum().clamp(min=1)
                        loss_masked = (loss_raw * mask_exp).sum() / n_masked
                        obs_exp    = (~patch_mask).unsqueeze(-1).expand_as(loss_raw)
                        n_obs      = obs_exp.sum().clamp(min=1)
                        loss_recon = (loss_raw * obs_exp).sum() / n_obs
                        loss       = loss_masked + self.recon_weight * loss_recon
                        total_val += loss.item()
                        n_v += 1
                val_loss = total_val / max(n_v, 1)

                if val_loss < best_val:
                    best_val     = val_loss
                    patience_cnt = 0
                    best_state   = copy.deepcopy(self.model.state_dict())
                else:
                    patience_cnt += 1

                if (epoch + 1) % 10 == 0:
                    print(
                        f"    [PatchTST] epoch {epoch+1}/{self.epochs}, "
                        f"train={train_loss:.4f}, val={val_loss:.4f}"
                    )

                if patience_cnt >= self.patience:
                    print(
                        f"    [PatchTST] Early stopping @ epoch {epoch+1} "
                        f"(best val={best_val:.4f})"
                    )
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(
                        f"    [PatchTST] epoch {epoch+1}/{self.epochs}, "
                        f"loss={train_loss:.4f}"
                    )

        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ── impute ──────────────────────────────────

    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        결측 위치(mask_matrix == 0)를 PatchTST 예측값으로 채움.

        추론 흐름:
          1. 선형 보간으로 초기값 설정
          2. 변수별 독립 처리 (channel-independence)
          3. 오버랩 슬라이딩 윈도우로 스윕
          4. 실제 obs_mask를 concat 입력으로 — 부분 관측 정보 보존
          5. Per-patch bias correction
          6. 결측 위치에만 평균 예측값 대체
        """
        if self.model is None:
            from .linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)

        T, F = masked_data.shape
        P    = self.patch_len
        W    = min(self._ctx_window, (T // P) * P)
        W    = max(W, P)
        N    = W // P
        stride = max(P, W // 2)

        if T < P:
            from .linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)

        # 선형 보간 초기값
        init_data = (
            masked_data.copy()
            .interpolate(method="linear", limit_direction="both")
            .ffill().bfill().fillna(0.0)
            .values.astype(np.float32)
        )
        mask_vals = mask_matrix.values.astype(np.float32)
        mask_vals[np.isnan(mask_vals)] = 0.0

        accum  = np.zeros((T, F), dtype=np.float64)
        counts = np.zeros((T, F), dtype=np.float64)

        self.model.eval()

        with torch.no_grad():
            for col_idx in range(F):
                series   = init_data[:, col_idx]   # (T,)
                col_mask = mask_vals[:, col_idx]    # 1=유효, 0=결측

                for win_start in range(0, T, stride):
                    win_end = min(win_start + W, T)
                    seg_len = win_end - win_start

                    seg    = series[win_start:win_end].copy()
                    seg_mk = col_mask[win_start:win_end].copy()

                    # 윈도우에 결측 없으면 스킵
                    if seg_mk.min() == 1:
                        continue

                    # 짧은 윈도우 패딩
                    if seg_len < W:
                        fill_v = float(seg[-1]) if seg_len > 0 else 0.0
                        seg    = np.append(seg,    np.full(W - seg_len, fill_v,  dtype=np.float32))
                        seg_mk = np.append(seg_mk, np.ones(W - seg_len,          dtype=np.float32))

                    # Instance Normalization (훈련과 동일)
                    mu  = seg.mean()
                    std = seg.std() + 1e-8
                    seg_norm = (seg - mu) / std

                    # 비오버랩 패치 생성
                    patches = seg_norm[:N * P].reshape(N, P)       # (N, P)
                    obs_mk  = seg_mk[:N * P].reshape(N, P)         # (N, P) 실제 obs_mask

                    # 모델 입력: concat(patches * obs_mk, obs_mk) → (N, 2P)
                    x_in = np.concatenate([patches * obs_mk, obs_mk], axis=-1).astype(np.float32)
                    x_in_t = torch.tensor(x_in).unsqueeze(0).to(DEVICE)  # (1, N, 2P)

                    pred = self.model(x_in_t).squeeze(0).cpu().numpy()   # (N, P)

                    # 역정규화
                    pred_denorm = pred * std + mu  # (N, P)

                    # ── Per-patch bias correction ──
                    # 각 패치 내 관측 타임스텝에서 (예측 − 실제) 평균 편향 제거
                    seg_actual = seg[:N * P].reshape(N, P)  # 원스케일 실제값
                    for p_idx in range(N):
                        obs_ts = obs_mk[p_idx].astype(bool)  # (P,)
                        if obs_ts.sum() >= 2:
                            bias = (pred_denorm[p_idx][obs_ts] - seg_actual[p_idx][obs_ts]).mean()
                            pred_denorm[p_idx] -= bias

                    # 결측 위치에만 예측값 누적
                    for p_idx in range(N):
                        p_start = win_start + p_idx * P
                        p_end   = p_start + P
                        if p_end > T:
                            break
                        missing_ts = col_mask[p_start:p_end] == 0  # (P,) bool
                        if not missing_ts.any():
                            continue
                        accum[p_start:p_end, col_idx]  += pred_denorm[p_idx] * missing_ts
                        counts[p_start:p_end, col_idx] += missing_ts

        # 최종 결과: 관측값 보존 + 결측 위치는 평균 예측값으로 대체
        imputed = init_data.copy()
        filled  = counts > 0
        imputed[filled] = accum[filled] / counts[filled]

        result = pd.DataFrame(imputed, index=masked_data.index, columns=masked_data.columns)
        assert result.shape == masked_data.shape, (
            f"[PatchTST] output shape {result.shape} != input shape {masked_data.shape}"
        )
        # 남은 NaN fallback (시계열 끝단 패치 미포함 등)
        for col in result.columns:
            result[col] = result[col].interpolate(method="linear", limit_direction="both")
        return result.ffill().bfill()
