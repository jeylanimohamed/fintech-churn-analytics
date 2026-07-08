# Fintech Product Analytics: Churn-Label Classification & Simulated A/B Test

**Mohamed Jeylani**

This is an independent product-analytics portfolio project built on **fully synthetic digital-banking data**. It demonstrates an end-to-end workflow — synthetic data generation, SQL analysis, classification of an inactivity-based churn label, and a simulated A/B test of 30-day transaction activity. The generator intentionally encodes several behavioural relationships and a treatment effect, so the resulting metrics **demonstrate analytical methods rather than real customer behaviour or causal product impact**. No proprietary employer data, and no real customer records, were used.

> My professional analytics experience motivated me to expand my portfolio into product analytics. This project is a separate, self-directed exercise and is not connected to any employer or their data.

---

## Business question (illustrative)

Framed as a product analyst might: which behavioural signals are associated with an inactivity-based churn label, and does a simulated onboarding variant change 30-day transaction activity? Both questions are answered **only within a generated dataset**, to exercise the methods.

---

## Dataset (synthetic)

All data are generated locally by `data/generate_data.py`. The generator creates an illustrative digital-banking scenario using **assumed** distributions for plans, product-feature adoption, transaction activity and account inactivity. These assumptions are **not** calibrated to or validated against proprietary data from any real financial institution (including any named fintech).

| Table | Rows | Notes |
|---|---|---|
| `users` | 8,000 | plan, country, signup date, adopted features, inactivity, churn label |
| `transactions` | 60,000 | category and amount per transaction |
| `ab_test` | 2,000 | users sampled into a simulated experiment (control/treatment) |

Verify counts by running the project (see below).

**Churn label definition.** A user is labelled churned if they have been inactive for **more than 90 days** relative to a fixed cutoff (2026-01-01). Several model features are derived from the same transaction history that defines this label, so they share information with it (see Limitations).

---

## Repository structure

```
fintech-churn-analytics/
├── README.md
├── NOTES.md                  # short methodology / decision log
├── requirements.txt
├── analysis.py               # SQL analysis, churn-label classifier, simulated A/B test, charts
├── data/
│   └── generate_data.py      # synthetic data generator (git-ignored CSV output)
└── sql/
    ├── analysis_queries.sql  # business-style SQL queries
    └── load_to_sqlite.py     # builds data/fintech.db so the SQL runs standalone
```

---

## Methodology

1. **Generate** synthetic users, transactions and a simulated A/B sample (fixed seed for reproducibility).
2. **SQL analysis** via SQLite: churn label by plan/country, feature-adoption buckets, active-vs-churned behaviour, cohort retention.
3. **Churn-label classification**: a Random Forest classifies the inactivity-based label. Split: `train_test_split`, `test_size=0.25`, `random_state=42` (no stratification). This is classification of a label defined at one observation point — **not** a prospective forecast.
4. **Simulated A/B test**: the generator injects a higher expected 30-day transaction count into the treatment group. Analysis reports means, relative difference, a Welch t-test, a 95% confidence interval on the mean difference, Cohen's d, and a sample-ratio-mismatch (SRM) check.

Reported outputs describe **the generated sample only**.

---

## Results (outputs of the generated sample)

These recover relationships written into the generator; they are not causal findings and not real customer behaviour.

**Churn-label classifier (synthetic test set)**
- ROC-AUC **0.821**, accuracy 0.837.
- Features most associated with the label (RF importance, not causal): `tenure_months` 0.522, `txn_count` 0.175.

**Churn label by segment (generated)**
- By plan: Standard 19.8% vs Metal 10.7%.
- By feature adoption: 1–2 features 18.1%, 3–4 features 14.5%, 5+ features 15.9% (non-monotonic; the 5+ bucket is small).

**Simulated A/B test — primary metric: 30-day transaction count**
- Control n = 964, Treatment n = 1,036; SRM check p = 0.107 (balanced).
- Control mean 13.20 vs Treatment mean 15.06 (relative difference 14.2%).
- Mean difference **1.87** transactions, 95% CI **1.54 to 2.20**; Cohen's d **0.490**; Welch t = 10.99; p < 0.0001.
- The treatment effect was **injected by the generator**, so the low p-value shows the pipeline detects a **known** difference. It is **not** evidence that a real onboarding change works, and the primary metric is 30-day transaction activity, **not** retention.

---

## Limitations

- **Synthetic data.** Everything is generated; results are not real-world business, financial, or behavioural evidence.
- **Label circularity / potential leakage.** The churn label is inactivity-based and some predictors come from the same transaction history, so this is closer to classifying an inactivity status than forecasting future churn.
- **Feature importance is not causality.** Associations reflect the generator's encoded structure.
- **The A/B "effect" is injected.** A significant p-value confirms the analysis detects the encoded difference; it does not justify any real rollout.
- **Metric scope.** The experiment measures 30-day transaction count, not retention or revenue causally.

---

## Reproduce from a clean clone

```bash
pip install -r requirements.txt

python data/generate_data.py     # writes data/*.csv (git-ignored)
python analysis.py               # SQL analysis, classifier, simulated A/B test, charts -> output/

# Optional: run the SQL file standalone
python sql/load_to_sqlite.py     # builds data/fintech.db
# then, if the sqlite3 CLI is installed:
# sqlite3 data/fintech.db < sql/analysis_queries.sql
```

A typical run reports 8,000 users, 60,000 transactions, churn-label rate 17.0%, classifier ROC-AUC 0.821, and the simulated A/B results above.

---

## Technologies

Python, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, SQL, SQLite

---

## Potential next steps

- Redesign churn as a **prospective** task: a fixed observation window for features and a later outcome window for the label, with a temporal train/test split.
- Add guardrail metrics, a power-analysis example, and randomisation-balance checks to the experiment section.
- Compare the Random Forest against a regularised logistic-regression baseline.

---

## Résumé summary

Independent product-analytics project on **fully synthetic** digital-banking data: SQL analysis, Random Forest classification of an inactivity-based churn label (ROC-AUC 0.82 on a synthetic test set), and a **simulated** A/B test of 30-day transaction activity with effect size and confidence intervals. Metrics demonstrate the analytical workflow, not real customer behaviour.
