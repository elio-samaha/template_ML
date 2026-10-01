# ML research workbench

General templates: this decision guide and [ML_workbench.ipynb](ML_workbench.ipynb). Matching section numbers make code easy to find.

## Sailboat pricing project

Start with the **[boat-pricing guide](boats/README.md)** and **[question-by-question notebook](boats/Boat_pricing.ipynb)**. They cover numeric text extraction, all supplied EDA questions, pricing models for unseen makes/variants, diagnostics, and ten-boat predictions. The default dataset is synthetic; set the CSV paths to obtain real answers.

## Navigation

- [00 — Start here](#section-00)
- [01 — Imports and configuration](#section-01)
- [02 — Load data or run the demo](#section-02)
- [03 — Integrity and schema](#section-03)
- [04 — Reserve the holdout and design folds](#section-04)
- [05 — Target and missingness EDA](#section-05)
- [06 — Relationships, redundancy and categories](#section-06)
- [07 — Optional time and feature recipes](#section-07)
- [08 — Fold-safe preprocessing](#section-08)
- [09 — Baselines and candidate models](#section-09)
- [10 — Compare out-of-fold predictions](#section-10)
- [11 — Development diagnostics](#section-11)
- [12 — Optional small tuning / PCA / robust loss](#section-12)
- [13 — Importance and ablation](#section-13)
- [14 — Optional statistical tests](#section-14)
- [15 — Optional uncertainty and robustness](#section-15)
- [16 — Optional shift and finance diagnostics](#section-16)
- [17 — Freeze and evaluate the holdout once](#section-17)
- [18 — Prediction, export and research summary](#section-18)


<a id="section-00"></a>

## 00 — Start here

Purpose: a copy/paste workbench for unfamiliar tabular prediction problems. Run the core cells in order; optional recipes are disabled until selected. The default is a small synthetic regression demo, never a claim about real financial data.

Fast path: 01–06 → 08–11 → 13–15 → 18. Change configuration in 01, replace the demo in 02, and follow the same numbered headings in this guide and notebook. Each code block states its dependencies through the variables it uses. Keep notebook outputs clear when sharing real data.

Six-hour exercise: 0–30 min framing/integrity; 30–75 train-only EDA and split design; 75–150 baselines; 150–240 one nonlinear model and diagnostics; 240–300 robustness/interpretation; final hour results/presentation. Adapt to your actual deadline.

Before coding ask: what does a row represent, what is predicted, at what time, using what available information, and what decision does prediction support? Write the answers down. Never infer availability from a column name alone.

Dependencies: Python 3.8+; numpy, pandas, matplotlib, scipy, scikit-learn. Optional inference recipes use statsmodels. Install in your own environment with `python -m pip install numpy pandas matplotlib scipy scikit-learn statsmodels jupyter`. Check package versions on the supplied machine; the encoder recipe supports old and new sklearn APIs.

**Code:** notebook section 00.

<a id="section-01"></a>

## 01 — Imports and configuration

Set TASK to regression or classification; SPLIT to iid, time, or group. TIME identifies prediction timestamps; GROUP identifies entities that must stay apart for unseen-entity evaluation. For binary classification POS_LABEL is the event you care about. Choose features explicitly after inspecting the schema. Set GAP_TIME_BLOCKS from label overlap/availability, not by tuning performance.

**Code:** notebook section 01.

<a id="section-02"></a>

## 02 — Load data or run the demo

Replace only the demo block with pd.read_csv/path, read_parquet, read_excel, or your existing DataFrame. Preserve a raw copy. Parse dates explicitly and verify timezone/units. Join external tables using validate='many_to_one' and inspect unmatched rows; a many-to-many join can silently inflate your dataset.

**Code:** notebook section 02.

<a id="section-03"></a>

## 03 — Integrity and schema

Question → code: corrupt values? dtype/range/infinity checks. Missingness? missing fractions and patterns. Duplicate observations? duplicated plus a domain key. Repeated IDs? nunique and group sizes. Do not delete repeated rows automatically: repetitions may be legitimate. Structural checks can cover the input, but keep target relationship exploration in development data. Missing targets are excluded from supervised fitting; never impute them.

**Code:** notebook section 03.

<a id="section-04"></a>

## 04 — Reserve the holdout and design folds

IID: shuffled folds only if rows are exchangeable. Time: split distinct timestamps so simultaneous rows never straddle a boundary. The gap below counts timestamp blocks, not rows; irregular times/overlapping horizon labels require purging by actual label-end timestamps. Group: hold out whole entities for unseen-entity deployment. Forecasting known entities requires temporal validation instead. If both time and entity separation are required, implement a custom splitter matching deployment. Never use a random split to claim forecasting performance.

The final holdout is for one frozen evaluation. Use development folds to choose models/features. EDA on development data can still overfit model selection; keep experiments few and record them.

**Code:** notebook section 04.

<a id="section-05"></a>

## 05 — Target and missingness EDA

Does target shape change the loss? Regression: inspect tails, quantiles and skew; MSE emphasizes large errors, MAE the conditional median. Classification: inspect imbalance and practical error costs. Missingness can be informative but can also represent a changing collection process. Add missing indicators inside preprocessing. Investigate outliers before winsorizing; fit any thresholds on each training fold. Avoid dropping difficult validation observations.

**Code:** notebook section 05.

<a id="section-06"></a>

## 06 — Relationships, redundancy and categories

Linear signal? Pearson correlation. Monotone signal? Spearman. Nonlinear structure? scatter/binned target means, optionally mutual information (training only; noisy, not a significance test). Redundant predictors? correlation/conditioning; Ridge often helps. Categories? show counts alongside target averages: tiny groups produce misleading extremes. Zero correlation does not imply independence; correlated features make importance attribution ambiguous.

**Code:** notebook section 06.

<a id="section-07"></a>

## 07 — Optional time and feature recipes

Set RUN_TIME_EDA / RUN_FEATURE_RECIPE to True only when appropriate. Temporal trends? target rolling mean/std and feature-target correlations per time bucket. Lags and rolling means must respect entity boundaries, chronological order and actual availability. A shifted target is allowed only when that past label has matured at prediction time. Duplicate entity timestamps need an explicit aggregation/order rule. Log1p is only defined for x > -1 and is not automatically suitable for signed returns. Do not run cointegration on returns when the hypothesis concerns stationary combinations of nonstationary price levels: check integration order first.

**Code:** notebook section 07.

<a id="section-08"></a>

## 08 — Fold-safe preprocessing

Everything learned from values belongs in a pipeline: imputation, scaling, encoding, PCA, feature selection and clipping thresholds. Numeric-coded categories must be recast explicitly. IDs are not automatically predictors. Unseen categories use handle_unknown='ignore'; high-cardinality one-hot features can exhaust memory. This intentionally dense template fits small interview data; use sparse preprocessing plus compatible models for large data. All-missing numeric columns should be investigated and deliberately excluded if unusable.

**Code:** notebook section 08.

<a id="section-09"></a>

## 09 — Baselines and candidate models

Begin with mean/median or class-prior baseline, then a linear/regularized model and one nonlinear competitor. Ridge handles correlated predictors with shrinkage; Lasso encourages sparse coefficients but can select unstably among correlated features. Logistic regression provides class probabilities; evaluate calibration too. Forests capture nonlinear interactions but may overfit small samples and do not extrapolate well. PCA preserves feature variance, not target signal; validate it rather than assuming it helps. Avoid broad hyperparameter searches on a short exercise.

**Code:** notebook section 09.

<a id="section-10"></a>

## 10 — Compare out-of-fold predictions

Record fold metrics and variability, not only the best score. Classification probabilities must be aligned by class label. A temporal/group fold missing classes can make a classifier or AUC invalid: redesign/report it; never silently discard the fold. MSE/MAE/R² answer different questions. Below R² uses validation mean (sklearn convention); skill vs a training-mean baseline is separately computed in 11. Classification accuracy can hide minority-class failure; log loss measures probability quality. Folds overlap in training, so fold-score standard deviations are descriptive, not independent confidence intervals.

**Code:** notebook section 10.

<a id="section-11"></a>

## 11 — Development diagnostics

Use out-of-fold residuals, not training residuals. Patterns vs fitted values suggest misspecification; a fan shape suggests changing error variance. Tail errors motivate robust loss or distributional predictions. Segment errors by time/entity/category and show sample counts. Negative R² is possible. For classification inspect confusion matrix and reliability diagram; tune thresholds on development data only. For temporal ACF sort residuals and aggregate simultaneous observations; ordinary row-order ACF on a panel is misleading.

**Code:** notebook section 11.

<a id="section-12"></a>

## 12 — Optional small tuning / PCA / robust loss

Enable only a few hypotheses supported by EDA. This cell tunes on the existing development folds and changes the candidate model; rerun 10–11 to regenerate honest fold diagnostics (tuned-model OOF evaluation should use nested CV for an unbiased selection estimate). The untouched holdout remains the final assessment. Robust alternatives: HuberRegressor for numeric dense regression, or GradientBoostingRegressor(loss='huber'); positive-target log transformations via TransformedTargetRegressor require careful back-transformation since exp(E[log Y]) is not E[Y]. PCA goes inside preprocessing after scaling; never fit it globally.

**Code:** notebook section 12.

<a id="section-13"></a>

## 13 — Importance and ablation

Why does a feature help? Permute raw columns on validation rows through the whole pipeline; this naturally groups a categorical feature's one-hot columns. Correlated features may substitute for each other and reduce measured importance; permutation can also create unrealistic combinations. Use drop-feature/group ablation across folds for incremental value. Neither coefficients nor importance establish causality. Partial dependence assumes meaningful feature combinations; correlated features make it misleading.

**Code:** notebook section 13.

<a id="section-14"></a>

## 14 — Optional statistical tests

Tests need a stated null, sampling assumptions and effect size. Do not indiscriminately run every test. Welch t-test: difference in independent-group means, unequal variance allowed. Spearman: monotonic association, iid inference. Chi-square: categorical independence with adequate expected counts. ADF: unit-root null; KPSS: stationarity null; use chronological single series with explicit deterministic trend choice. Ljung–Box: residual serial dependence; Breusch–Pagan: heteroskedasticity in a specified regression. HAC standard errors address some serial/variance dependence for coefficient inference, not leakage or wrong functional form. Adjust p-values across related exploratory hypotheses; low p is not economic usefulness or causality. Block/cluster inference is required for dependent observations.

**Code:** notebook section 14.

<a id="section-15"></a>

## 15 — Optional uncertainty and robustness

Compare losses on the same observations (paired differences). Resample iid rows only for iid data; contiguous block resampling is a rough stationary time-series recipe, with block length checked for sensitivity. For panels resample time blocks including all entities or use appropriate cluster inference. This interval conditions on already fitted predictions: it does not include training/model-selection uncertainty. For correlated groups, bootstrap whole groups instead. Check different windows, simpler models, ablations, extreme-value sensitivity and performance by segment; repeated experimentation consumes validation information.

**Code:** notebook section 15.

<a id="section-16"></a>

## 16 — Optional shift and finance diagnostics

Before final evaluation use development fold-to-fold comparisons for shift. After freezing, descriptive development/holdout differences can explain failure, but adapting then requires a new holdout. KS tests compare marginal numeric distributions under iid assumptions; dependence, large n and many features complicate p-values. Covariate shift changes P(X), concept drift changes P(Y|X). Marginal drift alone does not prove a failed model.

For finance ask: horizon, execution delay, tradable prices, spread/fees/slippage, turnover, exposure, capacity and overlapping labels. Correlation/IC is not P&L. Sharpe must use equally spaced portfolio returns, not arbitrary prediction rows. Annual sqrt scaling assumes suitable dependence; show tail loss/drawdown and net costs. Do not claim profitability from anonymized regression data.

**Code:** notebook section 16.

<a id="section-17"></a>

## 17 — Freeze and evaluate the holdout once

Enable RUN_HOLDOUT only once choices are frozen. Fit preprocessing/model on all development data; evaluate the reserved holdout once, alongside a training-only baseline. Avoid interpreting sklearn R² as skill against a training mean; both are reported. New feature choices based on this result invalidate its status as an untouched estimate. Inspect shift after this point without claiming a second tuned score is final.

**Code:** notebook section 17.

<a id="section-18"></a>

## 18 — Prediction, export and research summary

Prediction uses the exact saved feature schema and fitted preprocessing. Preserve row IDs and ordering. Do not refit on new prediction data. Store artifacts only in an appropriate private location; never commit interview/company data or fitted sensitive artifacts to this public repository. Pickle/joblib files must come from trusted sources. Export code is off by default.

Present: objective and observation grain → 2–4 findings that changed decisions → split and leakage controls → baseline/model fold table → diagnostics/importance → uncertainty and limitations → next experiments. Explain why the selected model is justified, what could invalidate the finding and what you would test with another day. Keep an experiment log of hypothesis, change, fold result, decision and time. No signal is a valid outcome when supported by honest evaluation.

**Code:** notebook section 18.
