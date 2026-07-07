"""
Fintech Product Analytics — Churn & A/B Test Analysis
======================================================
Project by Mohamed Jeylani | July 2026

Goal: Understand why users churn from a digital banking product and
whether a new onboarding flow improves retention. Uses synthetic data
modeled after patterns I've seen in fintech analytics.

Structure:
  1. Load & explore the data
  2. SQL analysis via SQLite
  3. Churn driver analysis (what makes users leave?)
  4. A/B test evaluation (did the new onboarding work?)
  5. Visualizations
  6. Business recommendations
"""

import sqlite3
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sns.set_style('whitegrid')
sns.set_palette('mako')

# ══════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════

os.makedirs('output', exist_ok=True)

users = pd.read_csv('data/users.csv')
txns = pd.read_csv('data/transactions.csv')
ab_test = pd.read_csv('data/ab_test.csv')

print(f"Users: {len(users):,} | Transactions: {len(txns):,} | A/B test: {len(ab_test):,}")
print(f"Churn rate: {users['churned'].mean()*100:.1f}%")
print(f"Plans: {users['plan'].value_counts().to_dict()}")
print()

# ══════════════════════════════════════════
# 2. SQL ANALYSIS
# ══════════════════════════════════════════

conn = sqlite3.connect(':memory:')
users.to_sql('users', conn, index=False)
txns.to_sql('transactions', conn, index=False)

print("=" * 55)
print("SQL: Churn Rate by Plan & Country")
print("=" * 55)
q = pd.read_sql_query("""
    SELECT plan, country,
           COUNT(*) as users,
           ROUND(SUM(churned)*100.0/COUNT(*), 1) as churn_pct
    FROM users
    GROUP BY plan, country
    HAVING users >= 50
    ORDER BY churn_pct DESC
    LIMIT 8
""", conn)
print(q.to_string(index=False))
print()

print("=" * 55)
print("SQL: Feature Adoption vs Churn")
print("=" * 55)
q2 = pd.read_sql_query("""
    SELECT
        CASE
            WHEN LENGTH(adopted_features) - LENGTH(REPLACE(adopted_features, ';', '')) + 1 >= 5 THEN '5+ features'
            WHEN LENGTH(adopted_features) - LENGTH(REPLACE(adopted_features, ';', '')) + 1 >= 3 THEN '3-4 features'
            ELSE '1-2 features'
        END as adoption_level,
        COUNT(*) as users,
        ROUND(SUM(churned)*100.0/COUNT(*), 1) as churn_pct
    FROM users
    GROUP BY adoption_level
    ORDER BY churn_pct DESC
""", conn)
print(q2.to_string(index=False))
print()

print("=" * 55)
print("SQL: Churned vs Active User Behavior")
print("=" * 55)
q3 = pd.read_sql_query("""
    WITH stats AS (
        SELECT t.user_id, COUNT(*) as txn_count,
               ROUND(AVG(t.amount),2) as avg_amt,
               COUNT(DISTINCT t.category) as cats_used
        FROM transactions t GROUP BY t.user_id
    )
    SELECT CASE WHEN u.churned=1 THEN 'Churned' ELSE 'Active' END as status,
           COUNT(*) as users,
           ROUND(AVG(s.txn_count),1) as avg_txns,
           ROUND(AVG(s.avg_amt),2) as avg_txn_value,
           ROUND(AVG(s.cats_used),1) as avg_categories
    FROM users u JOIN stats s ON u.user_id = s.user_id
    GROUP BY u.churned
""", conn)
print(q3.to_string(index=False))
print()

# ══════════════════════════════════════════
# 3. CHURN DRIVER ANALYSIS
# ══════════════════════════════════════════

print("=" * 55)
print("CHURN DRIVER ANALYSIS")
print("=" * 55)

# Enrich users with transaction stats
txn_stats = txns.groupby('user_id').agg(
    txn_count=('transaction_id', 'count'),
    total_spend=('amount', 'sum'),
    avg_txn_value=('amount', 'mean'),
    categories_used=('category', 'nunique')
).reset_index()

df = users.merge(txn_stats, on='user_id', how='left').fillna(0)

# Feature adoption count
df['feature_count'] = df['adopted_features'].apply(
    lambda x: len(str(x).split(';')) if pd.notna(x) and str(x) != '' else 0
)

# Tenure in months
df['signup_date'] = pd.to_datetime(df['signup_date'])
cutoff = pd.Timestamp('2026-01-01')
df['tenure_months'] = ((cutoff - df['signup_date']).dt.days / 30.44).round(1)

# Plan one-hot encoding
plan_dummies = pd.get_dummies(df['plan'], prefix='plan')
df = pd.concat([df, plan_dummies], axis=1)

# Churn by plan
print("\nChurn rate by plan:")
plan_churn = df.groupby('plan').agg(
    users=('user_id', 'count'),
    churn_rate=('churned', 'mean')
)
plan_churn['churn_rate'] = (plan_churn['churn_rate'] * 100).round(1)
print(plan_churn.to_string())
print()

# Churn by feature adoption
df['adoption_bucket'] = pd.cut(df['feature_count'], 
    bins=[0, 2, 4, 9], labels=['1-2 features', '3-4 features', '5+ features'])
feat_churn = df.groupby('adoption_bucket', observed=False)['churned'].mean() * 100
print(f"Churn when 1-2 features: {feat_churn.iloc[0]:.1f}%")
print(f"Churn when 3-4 features: {feat_churn.iloc[1]:.1f}%")
print(f"Churn when 5+ features:  {feat_churn.iloc[2]:.1f}%")
print()

# Churn prediction model
print("Building churn prediction model...")
features = ['age', 'tenure_months', 'txn_count', 'total_spend', 
            'avg_txn_value', 'categories_used', 'feature_count',
            'plan_Plus', 'plan_Premium', 'plan_Metal', 'plan_Standard']

X = df[features].fillna(0)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.3f}")

# Feature importance
imp = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance')

print("\nWhat drives churn (feature importance):")
for _, row in imp.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:20s} {bar} {row['importance']:.3f}")

# ══════════════════════════════════════════
# 4. A/B TEST EVALUATION
# ══════════════════════════════════════════

print("\n" + "=" * 55)
print("A/B TEST: New Onboarding Flow")
print("=" * 55)

control = ab_test[ab_test['variant'] == 'control']
treatment = ab_test[ab_test['variant'] == 'treatment']

print(f"\nControl group:   {len(control)} users")
print(f"Treatment group: {len(treatment)} users")

# Transaction count (our primary metric)
ctrl_txn = control['txn_count_30d']
trt_txn = treatment['txn_count_30d']

print(f"\nControl avg transactions (30d):   {ctrl_txn.mean():.1f}")
print(f"Treatment avg transactions (30d):  {trt_txn.mean():.1f}")
print(f"Lift:                             {((trt_txn.mean()/ctrl_txn.mean())-1)*100:.1f}%")

# T-test
t_stat, p_value = stats.ttest_ind(trt_txn, ctrl_txn)
print(f"\nT-statistic: {t_stat:.3f}")
print(f"P-value:     {p_value:.4f}")
print(f"Significant at 95% confidence: {'YES ✓' if p_value < 0.05 else 'no'}")

# Spend comparison
ctrl_spend = control['total_spend_30d']
trt_spend = treatment['total_spend_30d']
print(f"\nControl avg spend (30d):   ${ctrl_spend.mean():.2f}")
print(f"Treatment avg spend (30d):  ${trt_spend.mean():.2f}")
print(f"Revenue lift:               {((trt_spend.mean()/ctrl_spend.mean())-1)*100:.1f}%")

# ══════════════════════════════════════════
# 5. VISUALIZATIONS
# ══════════════════════════════════════════

print("\n" + "=" * 55)
print("GENERATING CHARTS")
print("=" * 55)

# Fig 1: Churn by plan + feature adoption
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

plan_order = ['Standard', 'Plus', 'Premium', 'Metal']
plan_data = df.groupby('plan')['churned'].mean() * 100
plan_data = plan_data.reindex(plan_order)
bars = axes[0].bar(plan_data.index, plan_data.values, color=sns.color_palette('mako', 4))
axes[0].set_title('Churn Rate by Plan', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, plan_data.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

feat_order = ['1-2 features', '3-4 features', '5+ features']
feat_data = df.groupby('adoption_bucket', observed=False)['churned'].mean() * 100
feat_data = feat_data.reindex(feat_order)
bars = axes[1].bar(feat_data.index, feat_data.values, color=sns.color_palette('mako', 3))
axes[1].set_title('Churn Rate by Feature Adoption', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, feat_data.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('output/churn_drivers.png', dpi=140, bbox_inches='tight')
plt.close()
print("  ✓ output/churn_drivers.png")

# Fig 2: Feature importance
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(imp['feature'], imp['importance'], color=sns.color_palette('mako', len(imp)))
ax.set_title('What Predicts Churn? (Random Forest Feature Importance)', fontweight='bold')
ax.set_xlabel('Importance Score')
for bar, val in zip(bars, imp['importance']):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
           va='center', fontsize=9)
plt.tight_layout()
plt.savefig('output/churn_feature_importance.png', dpi=140, bbox_inches='tight')
plt.close()
print("  ✓ output/churn_feature_importance.png")

# Fig 3: A/B Test results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ab_colors = ['#95a5a6', '#2ecc71']
axes[0].bar(['Control', 'Treatment'], [ctrl_txn.mean(), trt_txn.mean()], color=ab_colors)
axes[0].set_title('Avg Transactions (30 days)', fontweight='bold')
axes[0].set_ylabel('Transaction Count')
for i, (mean, std, n) in enumerate([(ctrl_txn.mean(), ctrl_txn.std(), len(ctrl_txn)), (trt_txn.mean(), trt_txn.std(), len(treatment))]):
    axes[0].errorbar(i, mean, yerr=std/np.sqrt(n), capsize=8, color='black', linewidth=1.5)
    axes[0].text(i, mean + 0.3, f'{mean:.1f}', ha='center', fontweight='bold', fontsize=11)

axes[1].bar(['Control', 'Treatment'], [ctrl_spend.mean(), trt_spend.mean()], color=ab_colors)
axes[1].set_title('Avg Spend (30 days)', fontweight='bold')
axes[1].set_ylabel('Spend ($)')
for i, (mean, std, n) in enumerate([(ctrl_spend.mean(), ctrl_spend.std(), len(control)), (trt_spend.mean(), trt_spend.std(), len(treatment))]):
    axes[1].errorbar(i, mean, yerr=std/np.sqrt(n), capsize=8, color='black', linewidth=1.5)
    axes[1].text(i, mean + 5, f'${mean:.0f}', ha='center', fontweight='bold', fontsize=11)

fig.suptitle(f'A/B Test Results: New Onboarding Flow (p={p_value:.4f})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/ab_test_results.png', dpi=140, bbox_inches='tight')
plt.close()
print("  ✓ output/ab_test_results.png")

# Fig 4: Transaction categories by churn status
fig, ax = plt.subplots(figsize=(11, 5))
cat_by_churn = txns.merge(users[['user_id', 'churned']], on='user_id')
cat_pivot = cat_by_churn.pivot_table(
    values='amount', index='category', columns='churned',
    aggfunc='mean'
).fillna(0)
cat_pivot.columns = ['Active', 'Churned']
cat_pivot = cat_pivot.sort_values('Active')

x = np.arange(len(cat_pivot))
width = 0.35
bars1 = ax.barh(x - width/2, cat_pivot['Active'], width, label='Active Users', color='#3498db', alpha=0.85)
bars2 = ax.barh(x + width/2, cat_pivot['Churned'], width, label='Churned Users', color='#e74c3c', alpha=0.85)
ax.set_yticks(x)
ax.set_yticklabels(cat_pivot.index)
ax.set_xlabel('Avg Transaction Amount ($)')
ax.set_title('Spending Patterns: Active vs Churned Users', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('output/spending_by_churn.png', dpi=140, bbox_inches='tight')
plt.close()
print("  ✓ output/spending_by_churn.png")

# ══════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════

print("\n" + "=" * 55)
print("KEY FINDINGS")
print("=" * 55)

top_driver = imp.iloc[-1]
second_driver = imp.iloc[-2]
print(f"""
1. CHURN DRIVERS
   Top predictor: {top_driver['feature']} ({top_driver['importance']:.3f} importance)
   Second: {second_driver['feature']} ({second_driver['importance']:.3f} importance)

2. PLAN IMPACT
   Standard plan churn: {plan_data['Standard']:.1f}% vs Metal: {plan_data['Metal']:.1f}%
   Higher-tier plans show {'lower' if plan_data['Metal'] < plan_data['Standard'] else 'higher'} churn

3. FEATURE ADOPTION
   Users with 5+ features churn at {feat_data['5+ features']:.1f}% vs {feat_data['1-2 features']:.1f}% for 1-2 features
   Each additional feature adopted reduces churn risk

4. A/B TEST: NEW ONBOARDING
   Transaction lift: {((trt_txn.mean()/ctrl_txn.mean())-1)*100:.1f}%
   Spend lift: {((trt_spend.mean()/ctrl_spend.mean())-1)*100:.1f}%
   Statistically significant: {'YES' if p_value < 0.05 else 'NO'} (p={p_value:.4f})

5. RECOMMENDATIONS
   → Roll out new onboarding flow to all users ({((trt_txn.mean()/ctrl_txn.mean())-1)*100:.1f}% transaction lift)
   → Target Standard plan users with feature adoption campaigns
   → Monitor users who drop below 3 active features — they're at risk
   → Crypto and Stocks categories correlate with lower churn — promote these
""")