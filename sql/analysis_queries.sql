-- Fintech Product Analytics — SQL Queries
-- Business questions a product analyst would answer at a digital bank

-- =========================================
-- 1. Monthly Active Users (MAU) trend
-- =========================================
SELECT
    strftime('%Y-%m', t.date) AS month,
    COUNT(DISTINCT t.user_id) AS monthly_active_users,
    COUNT(t.transaction_id) AS total_transactions,
    ROUND(SUM(t.amount), 2) AS total_volume,
    ROUND(AVG(t.amount), 2) AS avg_transaction_value
FROM transactions t
GROUP BY month
ORDER BY month;

-- =========================================
-- 2. Churn rate by plan and country
-- =========================================
SELECT
    u.plan,
    u.country,
    COUNT(*) AS total_users,
    SUM(u.churned) AS churned_users,
    ROUND(SUM(u.churned) * 100.0 / COUNT(*), 1) AS churn_rate_pct,
    ROUND(AVG(u.days_inactive), 0) AS avg_days_inactive
FROM users u
GROUP BY u.plan, u.country
HAVING total_users >= 50
ORDER BY churn_rate_pct DESC;

-- =========================================
-- 3. Feature adoption vs churn
-- =========================================
SELECT
    CASE
        WHEN LENGTH(u.adopted_features) - LENGTH(REPLACE(u.adopted_features, ';', '')) + 1 >= 5 THEN '5+ features'
        WHEN LENGTH(u.adopted_features) - LENGTH(REPLACE(u.adopted_features, ';', '')) + 1 >= 3 THEN '3-4 features'
        ELSE '1-2 features'
    END AS feature_adoption,
    COUNT(*) AS user_count,
    ROUND(SUM(u.churned) * 100.0 / COUNT(*), 1) AS churn_rate_pct,
    ROUND(AVG(u.days_inactive), 0) AS avg_days_inactive
FROM users u
GROUP BY feature_adoption
ORDER BY churn_rate_pct DESC;

-- =========================================
-- 4. User transaction behavior before churn
-- =========================================
WITH user_activity AS (
    SELECT
        t.user_id,
        COUNT(t.transaction_id) AS total_txns,
        ROUND(SUM(t.amount), 2) AS total_spend,
        COUNT(DISTINCT t.category) AS categories_used,
        ROUND(AVG(t.amount), 2) AS avg_txn_value,
        MAX(t.date) AS last_txn_date
    FROM transactions t
    GROUP BY t.user_id
)
SELECT
    CASE WHEN u.churned = 1 THEN 'Churned' ELSE 'Active' END AS status,
    COUNT(*) AS users,
    ROUND(AVG(ua.total_txns), 1) AS avg_txns,
    ROUND(AVG(ua.total_spend), 2) AS avg_total_spend,
    ROUND(AVG(ua.categories_used), 1) AS avg_categories_used,
    ROUND(AVG(ua.avg_txn_value), 2) AS avg_txn_value
FROM users u
JOIN user_activity ua ON u.user_id = ua.user_id
GROUP BY u.churned;

-- =========================================
-- 5. Top spending categories by plan
-- =========================================
SELECT
    u.plan,
    t.category,
    COUNT(t.transaction_id) AS txn_count,
    ROUND(SUM(t.amount), 2) AS total_spend,
    ROUND(AVG(t.amount), 2) AS avg_spend,
    COUNT(DISTINCT t.user_id) AS unique_users
FROM transactions t
JOIN users u ON t.user_id = u.user_id
GROUP BY u.plan, t.category
ORDER BY u.plan, total_spend DESC;

-- =========================================
-- 6. Cohort retention (users who transact in month N after signup)
-- =========================================
WITH user_first_month AS (
    SELECT user_id, strftime('%Y-%m', signup_date) AS cohort
    FROM users
),
monthly_activity AS (
    SELECT DISTINCT user_id, strftime('%Y-%m', date) AS activity_month
    FROM transactions
)
SELECT
    ufm.cohort,
    COUNT(DISTINCT ufm.user_id) AS cohort_size,
    COUNT(DISTINCT CASE WHEN ma.activity_month = ufm.cohort THEN ufm.user_id END) AS month_0,
    COUNT(DISTINCT CASE WHEN ma.activity_month = strftime('%Y-%m', date(ufm.cohort || '-01', '+1 month')) THEN ufm.user_id END) AS month_1,
    COUNT(DISTINCT CASE WHEN ma.activity_month = strftime('%Y-%m', date(ufm.cohort || '-01', '+3 months')) THEN ufm.user_id END) AS month_3,
    COUNT(DISTINCT CASE WHEN ma.activity_month = strftime('%Y-%m', date(ufm.cohort || '-01', '+6 months')) THEN ufm.user_id END) AS month_6
FROM user_first_month ufm
LEFT JOIN monthly_activity ma ON ufm.user_id = ma.user_id
GROUP BY ufm.cohort
ORDER BY ufm.cohort;