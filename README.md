# Fintech Product Analytics: Churn & A/B Testing

**Mohamed Jeylani** | July 2026

Analysis of user churn and an A/B test for a digital banking product — 8,000 users, 60,000 transactions. Built because I wanted to understand the kind of product analytics work that fintechs like Revolut do day-to-day.

---

## Why I Built This

My first project was healthcare analytics (makes sense given my internships at UnitedHealth Group and Optum). But I'm targeting fintech roles too, and I didn't have anything that showed I could do product analytics. This project fills that gap.

The two things I focused on:
1. **Churn analysis** — what makes users leave a digital banking app?
2. **A/B testing** — does a new onboarding flow actually improve retention?

---

## What's In Here

| File | What It Does |
|---|---|
| `analysis.py` | Full pipeline — data loading, SQL queries, churn modeling, A/B test evaluation, 4 charts |
| `sql/analysis_queries.sql` | 6 business questions (MAU trends, churn by plan/country, cohort retention, etc.) |
| `data/generate_data.py` | Synthetic data generator — modeled after real fintech user patterns |
| `output/` | 4 charts (see below) |

---

## Key Findings

### 1. Churn Drivers
- **Tenure** is the #1 predictor — newer users churn way more. Makes sense: if someone's been around 18 months, they're probably staying.
- **Transaction count** is #2 — active users stick around.
- **Plan tier matters**: Metal plan users churn at 10.7% vs. 19.8% for Standard. Higher commitment = lower churn.

### 2. Feature Adoption Reduces Churn
Users who adopt 3+ product features churn less than those using only 1-2. This feels obvious in hindsight — the more value someone gets from the product, the less likely they are to leave.

### 3. A/B Test: New Onboarding Flow → **14.2% Transaction Lift**
- Treatment group averaged 15.1 transactions in 30 days vs 13.2 for control
- 15.1% higher spend ($451 vs $392)
- p < 0.0001 — this is a real effect, not noise
- **Recommendation**: Roll out the new flow to all users

---

## How to Run

```bash
pip install -r requirements.txt
python3 data/generate_data.py   # generate the synthetic data
python3 analysis.py             # run the full analysis
```

Output goes to `output/` — 4 PNG charts. SQL queries in `sql/analysis_queries.sql` can be run against any SQLite setup.

---

## Things I'd Improve With More Time

- [ ] Add an actual dashboard (thinking Streamlit or a Power BI embed)
- [ ] Implement a proper ETL pipeline instead of one-shot data loading
- [ ] Test more models (XGBoost, logistic regression) not just Random Forest
- [ ] Add a time-to-churn survival analysis (Kaplan-Meier curves)
- [ ] Simulate a multi-arm bandit instead of a simple A/B test

---

## Tech

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, scipy, SQL, SQLite

---

## Context

I built this for my job search — data analyst and product analytics roles, mostly at fintechs in London and the Netherlands. I've also got a healthcare analytics project (see my other repo) from my UnitedHealth Group / Optum experience. Between the two, I wanted to show I can work in different industries without needing a ton of ramp-up time.
