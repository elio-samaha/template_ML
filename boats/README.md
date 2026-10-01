# Sailboat pricing project

**Dataset not supplied: executable solution template, not numerical findings.**

[Open the notebook](Boat_pricing.ipynb).

## Navigation

- [00 — What this project answers](#section-00)
- [01 — Setup and editable settings](#section-01)
- [02 — Map columns and load data](#section-02)
- [03 — Part 1 Q1: numeric extraction and units](#section-03)
- [04 — Part 1 Q2: categorical cleanup and consistency](#section-04)
- [05 — Part 1 Q1a: Beneteau mainsail and category draft](#section-05)
- [06 — Part 1 Q3: three largest feature correlations](#section-06)
- [07 — Listing-price distribution and expensive boats](#section-07)
- [08 — Geographic pricing and consistency](#section-08)
- [09 — Mean and median by build year](#section-09)
- [10 — Part 2: target, features and unseen-make splits](#section-10)
- [11 — Baselines and pricing models](#section-11)
- [12 — Compare make and variant validation](#section-12)
- [13 — Residuals, segments, robustness and feature importance](#section-13)
- [14 — Freeze and reserved unseen-make evaluation](#section-14)
- [15 — Predict ten boats and report average/median](#section-15)
- [16 — Write the conclusions and next steps](#section-16)

<a id="section-00"></a>

## 00 — What this project answers

This is a complete, adaptable solution skeleton for the supplied sailboat questions. **No real dataset was supplied. All default data and any default outputs are synthetic demonstrations.** Run with your CSV to obtain numerical answers; do not present demo results as Kaggle findings.

Navigation: change settings in 01, replace the demo using DATA_PATH in 02, inspect parsing/audit in 03–04, run Part 1 in 05–09 and Part 2 in 10–16. Every section is matched in the guide and notebook. The code supports Python 3.8 syntax and both older/newer OneHotEncoder APIs.

Assumptions to verify: each row is a listing; target is already converted to USD; dimensions are metric unless explicit units say otherwise; 'model' means hull type only if the schema confirms it; 'main sale area' means mainsail area; the ten boats are supplied separately, otherwise ten held-out listings illustrate predictions. The phrase 'advanced, average and median' is ambiguous: this solution reports individual predictions, total, average and median; no unidentified 'advanced' statistic is invented.

Part 1 is a full-dataset descriptive exercise. Part 2 uses make/variant-aware validation, but if you adapt model choices after looking at full-data Part 1, the reserved holdout is **not truly blind**. For a strict research protocol do Part 1 on development data and get a separate final evaluation dataset.

Requirements: `python -m pip install numpy pandas scipy matplotlib scikit-learn statsmodels nbformat jupyter`. No uploads, APIs, or network calls are needed during analysis. Keep real data, trained artifacts and prediction files outside this public repository.

<a id="section-01"></a>

## 01 — Setup and editable settings

Set DATA_PATH to your CSV and TEN_BOATS_PATH to the ten unpriced boats if supplied. Units for bare numbers must come from the dataset documentation; never guess from magnitude. REFERENCE_YEAR should be the listing/valuation year, not automatically today's year for a historical dataset. If listings span years, compute age using a row-specific listing-year column. Locale DECIMAL_MARK='.' handles 1,234.5; choose ',' for 1.234,5. An isolated comma such as 1,234 is ambiguous unless locale is confirmed.

**Matching code:** notebook section 01.

<a id="section-02"></a>

## 02 — Map columns and load data

Aliases cover the spoken column names. If an exact name is missed, add COLUMN_OVERRIDES, e.g. {'your exact column': 'hull_type'}. Duplicate mappings raise an error so nothing is silently overwritten. Preserve the raw frame. Review the printed map before analysis. Hull type/model/variant can mean different things in actual files; edit the map after looking at their values.

**Matching code:** notebook section 02.

<a id="section-03"></a>

## 03 — Part 1 Q1: numeric extraction and units

Use dimensional parsing, not 'strip every non-digit': that would turn '10.5 m / 34 ft' into nonsense. Prefer explicitly metric representations, then convert feet/inches, square feet, tonnes/lb and hp/CV to m, m², kg and kW. Twin-engine power is summed only when written explicitly as a multiplier. Values with ranges, inequalities, conflicting equivalent units or several ambiguous numbers are flagged, not silently averaged. Bare numeric strings use only configured documented units. This recipe cannot infer the meaning of every free-text description; examine the audit and extend exact patterns as required. Comma decimals require confirmed locale. Units in column headers must be incorporated into the configuration/mapping.

**Matching code:** notebook section 03.

<a id="section-04"></a>

## 04 — Part 1 Q2: categorical cleanup and consistency

Normalize whitespace/case/accents, unify only known equivalent labels and preserve the raw source. 'Beneteau' and 'Bénéteau' become the same make. Do not fuzzy-merge manufacturers or variants without review. Check impossible years, nonpositive physical measurements, bad currencies, suspicious ratios and duplicate listings. Flag rather than deleting plausible luxury boats. Target values that are nonfinite/nonpositive cannot be modeled as positive USD prices. Exact duplicates are removed with a log; listings sharing make/variant/specifications are not automatically duplicates. If a listing ID exists, check/reconcile duplicates by ID. Missing category/specification values remain missing for Part 1 (denominators reported), then are imputed within training folds for modeling.

Deck area here is length × beam, a rectangular-envelope proxy; verify an existing column against the stated definition. If inconsistent, use the recomputed value and retain the discrepancy. This deterministic feature is not independent of length and beam and may dominate correlation rankings mechanically. Last-built-year is model-level metadata; future archive information may be unavailable at listing time, so it is excluded from the predictive default.

**Matching code:** notebook section 04.

<a id="section-05"></a>

## 05 — Part 1 Q1a: Beneteau mainsail and category draft

Compute the mean mainsail area among valid Beneteau listings; report total listings, observed count and units. Because specifications repeat across listings, also show an equally weighted model-variant estimate to expose sampling bias. Answer the category question using the largest observed median draft, show all ties, counts and missingness; a second table with a minimum count is a robustness check, not a replacement for the literal answer. No category or Beneteau value can be inferred without the real rows.

**Matching code:** notebook section 05.

<a id="section-06"></a>

## 06 — Part 1 Q3: three largest feature correlations

Rank unique numerical-feature pairs by absolute Pearson correlation; show signs and complete-pair counts, then compare Spearman ranking and scatter plots. Exclude target, row IDs and duplicated deck-area source values from feature-pair ranking; price associations are a separate table. Include the derived deck proxy for the literal feature question and show a second ranking without it. Require enough paired observations, examine top-three coverage and missingness. Strong scale relationships (length/displacement/sail area/beam) are physically plausible hypotheses, not verified rankings. Repeat at make+variant level as a robustness check because multiple listings of the same model can dominate. Avoid causal claims and avoid claiming top-three certainty when sample correlations are nearly tied.

**Matching code:** notebook section 06.

<a id="section-07"></a>

## 07 — Listing-price distribution and expensive boats

Plot prices in dollars and log dollars (positive values only); report percentiles. Define expensive as the top 5% of valid asking prices, report the dollar threshold and sensitivity to top 10%/1%. Describe numeric medians and categorical prevalence versus the remainder, with counts and prevalence ratios. This is descriptive selection by target, not an unbiased discovery of pricing causes. Expensive boats may be larger, newer, multihulls or from premium makes; check these hypotheses rather than asserting them. Means are sensitive to very large yachts; medians and log distributions make concentration clearer.

**Matching code:** notebook section 07.

<a id="section-08"></a>

## 08 — Geographic pricing and consistency

Start with counts/mean/median and box plots on log prices. Then compare like with like: region composition can differ in age, size, hull type and make/variant. Estimate a descriptive log-price regression controlling for physical characteristics, age, hull type and make+variant fixed effects; cluster standard errors by make+variant. The region coefficient exp(beta)-1 is an adjusted conditional association relative to the reference region, not a causal premium. Verify that comparable models span regions; disconnected/thin support and country-region collinearity make effects unidentified or unstable. Do not include country and region simultaneously without a clear design.

Consistency checks: same-variant cross-region comparisons; region ratios within size/age/hull strata with minimum counts; cluster bootstrap of raw regional medians. Stability of sign/magnitude across these is evidence of consistency. No consistency claim is guaranteed. Cluster bootstrap assumes model families are reasonable independent units; few groups need caution. Region can proxy condition, taxes, seller mix, equipment, currency conversion and listing dates. These omitted variables limit interpretation.

**Matching code:** notebook section 08.

<a id="section-09"></a>

## 09 — Mean and median by build year

Group by boat build year, not model's last-built year. Report count, mean, median and interquartile range; plot both with sparse years highlighted. Missing/invalid years are excluded and their count reported. This is cross-sectional cohort composition, not a depreciation curve: newer boats can be different sizes/models and survivorship/condition can differ. Compare within size/hull strata or fit the adjusted pricing model. Without listing dates, age assumes the configured valuation year; actual longitudinal depreciation needs repeated comparable transactions/listings and time controls.

**Matching code:** notebook section 09.

<a id="section-10"></a>

## 10 — Part 2: target, features and unseen-make splits

Target = positive asking price in USD. MAE predicts a conditional median under absolute loss; squared-dollar loss favors a conditional mean and is dominated by expensive boats. Use MAE as dealer-facing default, RMSE/log error/R² as complements; inspect percentage error by price band, since MAPE can be unstable on cheap boats. Models trained on log(price) and exponentiated produce a geometric-center estimate (a conditional median under suitable log-error symmetry), not automatically mean fair value. This code also includes a raw-dollar forest to compare objectives.

Features: physical dimensions, sail areas, engine power, displacement/headroom, age and hull/material/rig/category; optional brand/region. Exclude exact variant as a default predictive input, fine location/designer high-cardinality identifiers and last-built-year/archive data of questionable historical availability. Include brand only in a separately validated model; unknown one-hot values fall back to physical/general features. Do not target-encode using all rows. Age = reference year - year built, deck proxy, beam/length, sail-area/displacement scale ratios are deterministic features; all imputation/scaling is train-only.

Primary deployment test: whole makes held out. Secondary: whole make+variant combinations held out. If repeated listings of the same boat span model groups, deduplicate by boat ID first. A random row split would allow specification memorization. GroupKFold estimates cold-start behavior but cannot guarantee reliable prices for arbitrary boats far outside observed size/age/type ranges.

**Matching code:** notebook section 10.

<a id="section-11"></a>

## 11 — Baselines and pricing models

Compare a raw median baseline, interpretable log-Ridge, log-Random Forest and a raw-dollar Random Forest. Ridge has smooth scale effects and can extrapolate but may underfit nonlinear relationships; trees capture interactions but cannot extrapolate usefully beyond training support. Compare a physical-only/general model to optional brand+region, so cold-start behavior and geographic contribution are visible. Keep one-hot cardinality bounded. Dense encoding is intended for modest tabular challenges; switch to sparse/compatible estimators or CatBoost if available for large data. Unknown categories are handled, not declared familiar. No claims about the best model precede actual CV.

**Matching code:** notebook section 11.

<a id="section-12"></a>

## 12 — Compare make and variant validation

Print fold distributions and both row-weighted and make-weighted errors; a make with many listings should not be mistaken for universal performance. Select on development unseen-make MAE and use complexity/stability to resolve near ties. Variant folds answer a different deployment question and may incidentally contain unseen makes too; report the actual fraction. Do not tune using the reserved holdout. Fold training sets overlap, so fold SD is descriptive and does not form an ordinary iid confidence interval. Selection among these models still induces optimism in the CV winner; the held-out group test or a separate dataset is required for stronger evidence.

**Matching code:** notebook section 12.

<a id="section-13"></a>

## 13 — Residuals, segments, robustness and feature importance

Inspect unseen-make out-of-fold predictions: dollar/log residuals, actual vs predicted, error vs size/age/price band and group. Compare selected-minus-baseline losses by whole-make bootstrap, interpreting it as conditional on fitted predictions and model selection. Permute original raw feature columns on validation rows; this groups one-hot representations correctly. Strongly correlated size measures substitute, so individual importance is ambiguous; optional grouped-size permutation and feature ablation test incremental value. Importance is predictive, not causal. If baseline wins, say so and skip meaningless importance rather than interpreting arbitrary features.

**Matching code:** notebook section 13.

<a id="section-14"></a>

## 14 — Freeze and reserved unseen-make evaluation

RUN_HOLDOUT is off by default. Freeze features/cleaning/model before enabling it. The test makes are absent from all training rows. Compare baseline and selected model, segment the test errors and record make identities/counts. If earlier full-dataset EDA influenced your choices, explicitly report this as an internal benchmark with possible selection contamination, not a pristine independent test. Report observed out-of-support features separately; handling unknown categories does not solve domain shift.

**Matching code:** notebook section 14.

<a id="section-15"></a>

## 15 — Predict ten boats and report average/median

With TEN_BOATS_PATH supplied, reuse exactly the same schema mapping, deterministic cleaning and units, then predict with the development-fitted model. Refit on all labeled data only after the evaluation and model choice are finalized (explicitly do so yourself). Never include price in predictive features. No file supplied: select ten reserved boats using features/seed only and illustrate comparison to held-out actual prices. They must not be used for tuning. Report individual prices, total, mean and median across the ten point predictions; these are not estimates of the dealer's unknown aggregate realized sales. Flag unseen makes/variants and numeric extrapolation/missingness. Prediction intervals require a separate group-aware calibration protocol; ordinary iid conformal guarantees do not automatically apply to new manufacturers.

**Matching code:** notebook section 15.

<a id="section-16"></a>

## 16 — Write the conclusions and next steps

Use the following evidence-backed answer structure after running on real data:

1. **Cleaning:** enumerate raw unit/type issues, parse failures, currency checks, invalid years, duplicate treatment and deck-proxy discrepancies with counts. Explain that unresolved values remain missing and model imputation is fold-specific.
2. **Q1a:** give Beneteau average mainsail area (m²) plus sample count/missingness; highest-median category draft (m) plus ties/count and robustness.
3. **Q3 correlations:** name top three unique pairs with signed Pearson r, Spearman r and n; explain mechanical dependence for deck proxy and repeated model specifications. Describe price tail and top-quantile patterns with comparisons.
4. **Region/year:** separate raw composition effects from adjusted associations, discuss overlap/confounders and instability. Build-year trends are cross-sectional, not measured depreciation.
5. **Pricing:** state asking-price target, physical/general features and excluded identifiers; show baseline vs models under unseen-make and unseen-variant evaluation. Explain model objective, unknown-category fallback, error scale and uncertainty.
6. **Ten boats:** present ten predictions, flags, mean/median/total; state whether they were provided boats or an illustrative held-out sample.

Limitations: listing price vs completed sale price, omitted condition/refits/equipment/engine hours, listing time/inflation/FX and regional tax differences, dataset selection/survivorship, duplicate ads, model-size extrapolation and few independent brands. A useful dealer model should add condition/equipment/sales history, calibrate against realized transaction prices, use group-aware uncertainty and abstain/refer to expert appraisal for unfamiliar boats outside support.

Expected hypotheses are not findings: size/newness may predict prices; deck/length/beam may be highly correlated; raw regions may reflect composition. Fill numerical claims from actual outputs only. If the unseen-make baseline wins, report failure to justify a more complex model.
