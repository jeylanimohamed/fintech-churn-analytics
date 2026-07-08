# Fintech Product Analytics: Churn Classification & a Simulated A/B Test

**Mohamed Jeylani**

This is a personal product-analytics project I built to practise the kind of work a fintech analyst does day to day: writing SQL against a warehouse, building a churn model, and reading out an A/B test. All of the data is synthetic and generated locally — there's no employer data, no real customers, nothing proprietary. I wrote the data generator myself, and I deliberately baked a few behavioural patterns and a treatment effect into it, so the numbers below are really me recovering things I put into the data on purpose. Treat them as a demonstration of the methods, not as findings about how real customers behave.

> My day job got me interested in analytics, so I started this on the side to build out a product-analytics portfolio. It's entirely separate from any employer and uses none of their data.

## The question I was answering

If I were a product analyst, I'd want to know two things: which behaviours tend to go with an inactivity-based churn label, and whether a simulated onboarding change moves 30-day transaction activity. Both answers only hold inside the dataset I generated — the point is to run the analysis end to end, not to draw a real conclusion.

## The data (synthetic)

Everything comes out of `data/generate_data.py`. It builds a made-up digital-banking scenario with assumed distributions for plans, feature adoption, transaction activity and inactivity. None of it is calibrated against a real bank's numbers.

| Table | Rows | What's in it |
|---|---|---|
| `users` | 8,000 | plan, country, signup date, adopted features, inactivity, churn label |
| `transactions` | 60,000 | category and amount per transaction |
| `ab_test` | 2,000 | users pulled into the simulated experiment (control/treatment) |

You can check the counts yourself by running the project (below).

A user counts as churned if they've gone more than 90 days without a transaction, measured against a fixed cutoff of 2026-01-01. Worth flagging up front: several of the model features come from the same transaction history that defines the label, so they share information with it. More on that in the limitations.

## Repository structure

```
fintech-churn-analytics/
├── README.md
├── NOTES.md                  # short methodology / decision log
├── requirements.txt
├── analysis.py               # SQL analysis, churn classifier, simulated A/B test, charts
├── data/
│   └── generate_data.py      # synthetic data generator (CSV output is git-ignored)
└── sql/
    ├── analysis_queries.sql  # business-style SQL queries
    └── load_to_sqlite.py     # builds data/fintech.db so the SQL runs on its own
```

## How I approached it

1. **Generate** the users, transactions and the A/B sample, with a fixed seed so it's reproducible.
2. **SQL analysis** in SQLite: churn by plan and country, feature-adoption buckets, active-vs-churned behaviour, cohort retention.
3. **Churn classification**: a Random Forest on the inactivity-based label, split with `train_test_split` (`test_size=0.25`, `random_state=42`, no stratification). This is classifying a label defined at one point in time — not forecasting who'll churn next quarter.
4. **Simulated A/B test**: the generator gives the treatment group a higher expected 30-day transaction count. I report the group means, the relative difference, a Welch t-test, a 95% confidence interval on the difference, Cohen's d, and a sample-ratio-mismatch check.

Everything below describes the generated sample only.

## Results (on the generated data)

These recover the relationships I wrote into the generator. They aren't causal and they aren't real customer behaviour.

**Churn classifier (synthetic test set)**

- ROC-AUC 0.821, accuracy 0.837.
- Most important features (RF importance, not causation): `tenure_months` 0.522, `txn_count` 0.175.

**Churn by segment (generated)**

- By plan: Standard 19.8% vs Metal 10.7%.
- By feature adoption: 1–2 features 18.1%, 3–4 features 14.5%, 5+ features 15.9%. Not monotonic, and the 5+ bucket is small.

**Simulated A/B test — 30-day transaction count**

- Control n = 964, Treatment n = 1,036; SRM check p = 0.107, so the split is balanced.
- Control mean 13.20 vs Treatment mean 15.06 — a 14.2% relative difference.
- Difference of 1.87 transactions, 95% CI 1.54 to 2.20; Cohen's d 0.490; Welch t = 10.99; p < 0.0001.
- Since I injected that effect myself, the tiny p-value just means the pipeline correctly detects a difference I already know is there. It says nothing about whether a real onboarding change would work, and the metric here is transaction activity, not retention.

## Limitations

- It's all synthetic, so none of it is real-world evidence.
- The churn label is inactivity-based and some predictors come from the same transaction history, so this is closer to labelling an inactivity status than predicting future churn.
- Feature importance reflects the structure I encoded, not cause and effect.
- The A/B effect is injected — a significant result confirms detection, it doesn't justify a rollout.
- The experiment measures 30-day transaction count, not retention or revenue.

## Running it from a clean clone

```bash
pip install -r requirements.txt

python data/generate_data.py     # writes data/*.csv (git-ignored)
python analysis.py               # SQL analysis, classifier, simulated A/B test, charts -> output/

# optional: run the SQL file on its own
python sql/load_to_sqlite.py     # builds data/fintech.db
# then, if you have the sqlite3 CLI:
# sqlite3 data/fintech.db < sql/analysis_queries.sql
```

A normal run reports 8,000 users, 60,000 transactions, a 17.0% churn rate, classifier ROC-AUC 0.821, and the A/B numbers above.

## Tools

Python, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, SQL, SQLite

## Things I'd do next

- Reframe churn as a proper forward-looking problem: a fixed window for features, a later window for the outcome, and a time-based train/test split.
- Add guardrail metrics, a power analysis, and randomisation-balance checks to the experiment.
- Compare the Random Forest against a regularised logistic-regression baseline.

## One-line summary

Personal product-analytics project on fully synthetic banking data: SQL analysis, a Random Forest churn classifier (ROC-AUC 0.82 on a synthetic test set), and a simulated A/B test of 30-day transaction activity with effect size and confidence intervals. The numbers demonstrate the workflow, not real customer behaviour.
