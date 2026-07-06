"""
Generate synthetic fintech user data for churn/product analytics project.
Models patterns seen in digital banking apps — signups, transactions,
product feature adoption, and churn.

Run: python3 data/generate_data.py
Output: data/users.csv, data/transactions.csv, data/ab_test.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(530)

N_USERS = 8000
N_TRANSACTIONS = 60000

# --- Users ---
def generate_users(n):
    ages = np.random.normal(34, 11, n).clip(18, 75).astype(int)
    
    # Countries roughly proportional to Revolut's markets
    countries = np.random.choice(
        ['UK', 'Spain', 'Poland', 'Portugal', 'France', 'Germany', 'Other'],
        n, p=[0.28, 0.16, 0.15, 0.10, 0.12, 0.10, 0.09]
    )
    
    # Signup dates over 2 years
    start = datetime(2024, 1, 1)
    end = datetime(2025, 12, 31)
    signup_dates = [start + timedelta(days=int(x)) 
                    for x in np.random.randint(0, (end-start).days, n)]
    
    # Plan type
    plans = np.random.choice(['Standard', 'Plus', 'Premium', 'Metal'], n, 
                             p=[0.45, 0.25, 0.20, 0.10])
    
    # Features adopted (of 8 possible)
    features = ['Card Payments', 'FX Transfer', 'Crypto', 'Stocks', 
                'Budgeting', 'Vaults', 'Rewards', 'Travel Insurance']
    
    # Adoption probability increases with plan tier
    plan_weights = {'Standard': 0.15, 'Plus': 0.25, 'Premium': 0.35, 'Metal': 0.50}
    
    adopted_features = []
    for plan in plans:
        n_feat = np.random.binomial(8, plan_weights[plan])
        picked = list(np.random.choice(features, max(1, n_feat), replace=False))
        adopted_features.append('; '.join(picked))
    
    # Churn: users who haven't transacted in 90+ days by the cutoff date
    # We'll compute this properly after generating transactions
    # For now, placeholder
    churned = [0] * n
    
    # Days since last activity (will be updated after transactions)
    days_inactive = np.random.exponential(30, n).astype(int)
    
    return pd.DataFrame({
        'user_id': [f'U{str(i).zfill(5)}' for i in range(1, n+1)],
        'signup_date': signup_dates,
        'country': countries,
        'age': ages,
        'plan': plans,
        'adopted_features': adopted_features,
        'churned': churned,  # placeholder
        'days_inactive': days_inactive,  # placeholder
    })

# --- Transactions ---
def generate_transactions(users_df, n_txns):
    user_ids = users_df['user_id'].values
    plans = dict(zip(users_df['user_id'], users_df['plan']))
    signups = dict(zip(users_df['user_id'], users_df['signup_date']))
    
    # Plan affects activity level
    plan_activity = {'Standard': 1.0, 'Plus': 1.4, 'Premium': 1.8, 'Metal': 2.2}
    
    categories = ['Shopping', 'Food & Drink', 'Bills', 'Transport', 
                  'Entertainment', 'Transfer', 'Crypto', 'Stocks', 'Travel']
    cat_weights = [0.28, 0.20, 0.12, 0.10, 0.08, 0.10, 0.05, 0.03, 0.04]
    
    cutoff = datetime(2026, 1, 1)
    txns = []
    
    for i in range(n_txns):
        uid = np.random.choice(user_ids)
        signup = signups[uid]
        
        # Weighted toward active users
        activity_mult = plan_activity.get(plans[uid], 1.0)
        if np.random.random() > 0.3 * activity_mult:
            # More likely to pick active users
            uid = np.random.choice(user_ids)
            signup = signups[uid]
        
        max_date = min(cutoff, signup + timedelta(days=730))
        days_since_signup = (max_date - signup).days
        if days_since_signup <= 0:
            continue
            
        txn_date = signup + timedelta(days=np.random.randint(0, days_since_signup))
        
        cat = np.random.choice(categories, p=np.array(cat_weights)/sum(cat_weights))
        
        # Amount varies by category
        amount_ranges = {
            'Shopping': (5, 200), 'Food & Drink': (3, 80), 'Bills': (20, 500),
            'Transport': (2, 60), 'Entertainment': (5, 150), 'Transfer': (10, 2000),
            'Crypto': (10, 1000), 'Stocks': (20, 5000), 'Travel': (50, 3000)
        }
        lo, hi = amount_ranges[cat]
        amount = round(np.random.uniform(lo, hi), 2)
        
        txns.append({
            'transaction_id': f'TXN{str(i+1).zfill(7)}',
            'user_id': uid,
            'date': txn_date.strftime('%Y-%m-%d'),
            'category': cat,
            'amount': amount
        })
    
    return pd.DataFrame(txns)

# --- A/B Test Data ---
def generate_ab_test(users_df):
    """Simulate an A/B test: new onboarding flow vs old one.
    Metric: 30-day transaction volume"""
    n_test = min(2000, len(users_df))
    test_users = users_df.sample(n_test, random_state=42).copy()
    
    test_users['variant'] = np.random.choice(['control', 'treatment'], n_test)
    
    # Treatment group has ~18% higher 30-day transaction volume on average
    results = []
    for _, user in test_users.iterrows():
        base_volume = np.random.poisson(12)
        if user['variant'] == 'treatment':
            volume = base_volume + np.random.poisson(3)
        else:
            volume = base_volume + np.random.poisson(1)
        
        revenue = volume * np.random.uniform(15, 45)
        
        results.append({
            'user_id': user['user_id'],
            'variant': user['variant'],
            'txn_count_30d': volume,
            'total_spend_30d': round(revenue, 2),
            'days_active_30d': np.random.randint(1, 31)
        })
    
    return pd.DataFrame(results)

# --- Main ---
if __name__ == '__main__':
    print("Generating users...")
    users = generate_users(N_USERS)
    
    print("Generating transactions...")
    txns = generate_transactions(users, N_TRANSACTIONS)
    
    # Calculate real churn and inactivity
    cutoff = datetime(2026, 1, 1)
    txns['date'] = pd.to_datetime(txns['date'])
    last_txn = txns.groupby('user_id')['date'].max()
    
    for uid in users['user_id']:
        if uid in last_txn.index:
            days = (cutoff - last_txn[uid]).days
            users.loc[users['user_id'] == uid, 'days_inactive'] = days
            users.loc[users['user_id'] == uid, 'churned'] = 1 if days > 90 else 0
        else:
            # Never transacted
            signup = users.loc[users['user_id'] == uid, 'signup_date'].values[0]
            days = (cutoff - pd.Timestamp(signup)).days
            users.loc[users['user_id'] == uid, 'days_inactive'] = days
            users.loc[users['user_id'] == uid, 'churned'] = 1 if days > 90 else 0
    
    print("Generating A/B test data...")
    ab_test = generate_ab_test(users)
    
    users.to_csv('data/users.csv', index=False)
    txns.to_csv('data/transactions.csv', index=False)
    ab_test.to_csv('data/ab_test.csv', index=False)
    
    churn_rate = users['churned'].mean() * 100
    print(f"\nDone! {len(users):,} users, {len(txns):,} transactions")
    print(f"Churn rate: {churn_rate:.1f}%")
    print(f"Files: data/users.csv, data/transactions.csv, data/ab_test.csv")