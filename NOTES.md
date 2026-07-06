# Project Journal — kept while building this

## July 5, 2026
Starting this project tonight. I wanted something fintech-focused since my other project is healthcare. Revolut's job posting mentions Python + SQL + product analytics specifically, so I'm building toward that.

Decided on churn analysis + A/B testing. These are two of the most common things product analysts do at any tech company.

## July 6
Generated the synthetic data. 8,000 users, 60,000 transactions. Modeled it after what I'd expect to see in a banking app — different plan tiers, feature adoption rates, transaction categories.

The churn rate came out to 17%. That feels realistic for a digital banking product based on what I've read. Metal plan users churn way less (10.7%) which makes sense — higher commitment, more features.

One thing I noticed: the data generator is kind of slow for 60k transactions. If I were doing this for real, I'd probably use a proper database instead of building everything in pandas DataFrames. But for a portfolio project, this works.

## July 7
Finished the analysis. A few surprises:
- Tenure is way more predictive of churn than I expected (52% feature importance). I thought transaction count would be #1.
- The A/B test result is stronger than I thought it'd be — 14% lift with p < 0.0001. Almost too clean. If this were real data I'd be suspicious of some confounding factor.
- Feature adoption buckets didn't come out as clean as I wanted. 5+ features shows 15.9% churn vs 14.5% for 3-4 features. The direction is right but the curve isn't perfectly monotonic. Could be because the 5+ bucket is small (only 527 users).

Still want to add:
- [ ] Cohort retention curves (already wrote the query, just need to plot it)
- [ ] A survival analysis for time-to-churn
- [ ] Maybe a Streamlit dashboard if I have time before applications open

## July 8
Tidied up the code and wrote the README. Tried to keep things readable — I hate portfolio projects where the code is unreadable mess. Added comments where the logic isn't obvious.

Overall, happy with this. It's not production code but it shows I can:
- Write SQL to answer business questions
- Build a churn prediction model that's actually useful
- Run and interpret an A/B test properly (with statistical testing, not just eyeballing means)
- Present findings in a way that a PM or VP could act on

Two projects down. Might do one more in Power BI if I have time before Goldman's applications open August 15.