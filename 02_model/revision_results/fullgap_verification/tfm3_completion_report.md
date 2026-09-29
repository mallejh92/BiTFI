# Full-gap TimesFM3 implementation and re-evaluation

All four existing TimesFM3 configurations completed the same 3,421 mask cases and 6,205 scored variable-cases. Context remained 1,900 h, selected by the existing Figure S2 validation. No new model family, weights, masks, scalers or reference-selection policy was introduced.

| Existing configuration | Previous NMAE | Full-gap NMAE | Native calls | Maximum requested span (h) |
|---|---:|---:|---:|---:|
| TimesFM3.0 | 0.0600491713 | 0.0599794915 | 6205 | 168 |
| TimesFM3.0-MV | 0.0564015171 | 0.0563538860 | 6195 | 191 |
| TimesFM3.0-COV | 0.0499462751 | 0.0492194769 | 12400 | 1776 |
| TimesFM3.0-COV-SPA | 0.0416576203 | 0.0409527580 | 12400 | 1776 |

The primary local + cross comparison is now 0.0409527580; BiTFI remains 0.0345965650. The relative reduction is 15.52079346% (paired greenhouse bootstrap, 10,000 resamples, seed 42: 11.79869265–19.71020676%). The final manuscript analysis regenerates all intervals and contrasts from the complete current result set; the independent publication checks pass for all 21 primary configurations.

## Implementation

`forecasting_mode="full_gap"` is the default for TimesFM2.5 and all TimesFM3 comparison adapters. Every target forecast and auxiliary univariate initialization requests the complete applicable gap. `forecasting_mode="rolling"` explicitly retains the former chunked implementation for historical reproduction. The retained `horizon_len=128` attribute is only the rolling-mode chunk size, not an active full-gap restriction.

The TimesFM2.5 native forecast capacity starts at 256 positions and grows in 128-position multiples, up to the retained continuous quantile head limit of 1,024. Primary TimesFM2.5 requests are only 6–168 h. Requests beyond that head limit raise an explicit error rather than silently switching to rolling calls. The model checks the shared backend's actual compiled capacity.

## Leading merged-gap correction

There are 18 artificial-variable intervals joined to natural missingness. Ten leading initializations, in mask cases 2817, 2866, 2915, 2964, 3013 and 3062, span 1,776 positions; the largest nonleading merged span is 191 h. The final covariate call retains the existing no-past-context skip for leading intervals. Its univariate fallback uses reversed post-gap context. That forecast must be reversed back before it is assigned to chronological positions. This alignment was missing in the earlier generic helper.

The correction is explicitly enabled for full-gap TimesFM helpers; unrelated active models and BiTFI paths are unaffected. It directly changes ten leading initializations and can also change co-masked targets through their covariates. The univariate primary route forecasts the artificial interval from pre-gap context, while the multivariate route retains masked-input interpolation at these leading intervals; neither uses this corrected initialization path.

The partial covariate run before the alignment fix was interrupted and retained under `before_leading_orientation_fix`; its results are excluded. Both covariate configurations were then evaluated from the beginning.

## Verification

For every configuration, all 15 scenario-duration conditions passed hidden-truth perturbation, serial/batch, batch order, observed-value preservation and A–B–A checks at the unchanged tolerance (rtol 1e-5, atol 1e-6). All native covariate and initialization outputs were checked for sufficient output length; no 128-h requests occurred.

Both covariate configurations additionally passed direct native-forecast alignment checks for all ten leading initializations: reversed future context, a native 1,776-h forecast, and reversed output matched the chronological reconstruction. These six mask cases also passed hidden-truth, serial/batch, order and A–B–A checks. The largest alignment difference was 7.1526e-7. A synthetic asymmetric sequence independently verified both helper functions and their legacy opt-out.

BiTFI was replayed on the 15 conditions plus all six leading cases (21 masks). Maximum difference from stored masked predictions was 5.9605e-7; hidden-truth perturbation was bitwise invariant, and order differences remained within the original tolerance. Stored BiTFI predictions were not modified. Its implementation, stage runner, data protocol and dispatcher match the prior public code byte-for-byte.

Figure 6 reused exactly the same interval and selection. Only `TimesFM3.0-COV-SPA.csv` changed; the selection and BiTFI, SAITS, MOMENT and Spatial ridge exports are byte-identical.

## Hardware

Univariate and multivariate TimesFM3 re-evaluations used an RTX A6000. Both covariate evaluations used an RTX PRO 6000 Blackwell. All used the same highest-precision matrix setting after construction. BiTFI reuse verification and the Figure 6 export used A6000 GPUs. Driver elapsed times are not a hardware-matched runtime comparison.

The detailed model reports are in the corresponding `fullgap_verification/<model>/` folders; BiTFI reuse is in `BiTFI-reuse/verification.json`.
