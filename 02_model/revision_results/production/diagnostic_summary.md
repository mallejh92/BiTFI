# Selected-depth production verification

The original diagnostic set float32 matrix multiplication to `highest` before model construction, but `BiTFITimesFM3.__init__` reset it to `high`. The production experiment set `highest` again after construction. The corrected diagnostic follows the actual production order. No prediction, trained parameter, context length, refinement depth or original tolerance was changed.

Thirty fixed cases cover all 15 scenario-duration combinations. At the original tolerance (rtol=1e-5, atol=1e-6), every hidden-target, order, stored-output and observation check passes. Same-batch repeat and hidden-target perturbation predictions are bitwise identical. Maximum normalized differences: reversed batch 4.76837158e-7; regrouped stored output 1.07288361e-6; serial versus singleton-batch API 3.57627869e-7. Observed inputs differ by at most 2.94049581e-8 from stored float64 inputs because production converts them to float32.

An explicit `high`-precision control on the first 16 cases reproduces the previous reversed-batch maximum exactly (5.53280115e-5). The largest high-versus-highest prediction difference in that control is 1.96158886e-4. This isolates the precision setting as the source of the earlier failed check.

Replaying an original 16-case A6000 shard batch gives bitwise identical stored predictions (maximum difference zero). Replaying a 16-case batch originally evaluated on Blackwell gives a maximum difference of 1.07288361e-6 on A6000, still within the unchanged original tolerance. The stored prediction file's SHA-256 is unchanged before and after all checks. These results support numerical equivalence of the selected 1,900-h / five-pass production and direct serial APIs for the checked cases; they do not claim bitwise identity across arbitrary hardware and batch grouping.

The diagnostic's newly added bitwise-observation guard initially rejected the expected float64-to-float32 observation cast after writing the complete report. That diagnostic guard was corrected to the original allclose condition, and the complete report independently validated on CPU. All original tolerances were retained. Details, every case-level difference and both original-batch comparisons are recorded in `verification.json`.

MOMENT Fig. 6 was then exported with the validation-selected head. Only `figure6_common_window/MOMENT-FT.csv` changed; all other model hashes, the selected interval, masks and truth remain identical. Its previous CSV is preserved in this verification directory. GPU 1 has been released to the AutoGluon task.
