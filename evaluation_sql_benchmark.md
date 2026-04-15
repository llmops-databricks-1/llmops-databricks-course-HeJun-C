# SQL Benchmark Set for Churn Analytics Agent

## 1. How many users are in the `train` table?
**Difficulty:** Easy

```sql
SELECT COUNT(*) AS train_user_count
FROM train;
```

## 2. What is the overall churn rate in `train`?
**Difficulty:** Easy

```sql
SELECT ROUND(AVG(is_churn), 4) AS overall_churn_rate
FROM train;
```

## 3. How many rows in `members` have invalid / outlier ages?
**Difficulty:** Easy

```sql
SELECT COUNT(*) AS invalid_age_count
FROM members
WHERE bd IS NULL OR bd NOT BETWEEN 10 AND 100;
```

## 4. What is the gender distribution in `members` after treating null as `'unknown'`?
**Difficulty:** Easy

```sql
WITH gender_counts AS (
    SELECT
        COALESCE(gender, 'unknown') AS gender,
        COUNT(*) AS user_count
    FROM members
    GROUP BY COALESCE(gender, 'unknown')
)
SELECT
    gender,
    user_count,
    ROUND(user_count * 100.0 / SUM(user_count) OVER (), 2) AS pct_of_users
FROM gender_counts
ORDER BY user_count DESC, gender;
```

## 5. What is the average valid age in `members`?
**Difficulty:** Easy

```sql
SELECT ROUND(AVG(bd), 2) AS avg_valid_age
FROM members
WHERE bd BETWEEN 10 AND 100;
```

## 6. How many `train` users have a matching row in `members`?
**Difficulty:** Easy

```sql
SELECT COUNT(*) AS matched_train_members_users
FROM train t
INNER JOIN members m
    ON t.msno = m.msno;
```

## 7. How many transactions happened in February 2017?
**Difficulty:** Easy

```sql
SELECT COUNT(*) AS feb_2017_transaction_count
FROM transactions
WHERE transaction_date BETWEEN DATE '2017-02-01' AND DATE '2017-02-28';
```

## 8. What is the average `actual_amount_paid` for auto-renew transactions before March 2017?
**Difficulty:** Easy

```sql
SELECT ROUND(AVG(actual_amount_paid), 2) AS avg_paid_auto_renew
FROM transactions
WHERE is_auto_renew = 1
  AND transaction_date < DATE '2017-03-01';
```

## 9. What is the total listening time in seconds in February 2017?
**Difficulty:** Easy

```sql
SELECT ROUND(SUM(total_secs), 2) AS total_listening_secs_feb_2017
FROM user_logs
WHERE date BETWEEN DATE '2017-02-01' AND DATE '2017-02-28';
```

## 10. What is the average number of unique songs played per active user-day in February 2017?
**Difficulty:** Easy

```sql
SELECT ROUND(AVG(num_unq), 2) AS avg_daily_unique_songs_feb_2017
FROM user_logs
WHERE date BETWEEN DATE '2017-02-01' AND DATE '2017-02-28';
```

## 11. Among `train` users with a member profile, what is the churn rate by gender?
**Difficulty:** Medium

```sql
SELECT
    COALESCE(m.gender, 'unknown') AS gender,
    COUNT(*) AS user_count,
    ROUND(AVG(t.is_churn), 4) AS churn_rate
FROM train t
INNER JOIN members m
    ON t.msno = m.msno
GROUP BY COALESCE(m.gender, 'unknown')
ORDER BY user_count DESC, gender;
```

## 12. Among `train` users with a member profile, what is the churn rate by registration method?
**Difficulty:** Medium

```sql
SELECT
    m.registered_via,
    COUNT(*) AS user_count,
    ROUND(AVG(t.is_churn), 4) AS churn_rate
FROM train t
INNER JOIN members m
    ON t.msno = m.msno
GROUP BY m.registered_via
ORDER BY user_count DESC, m.registered_via;
```

## 13. For all `train` users, what is the average February 2017 listening time by churn status?
**Difficulty:** Medium

```sql
WITH feb_logs AS (
    SELECT
        msno,
        SUM(total_secs) AS total_secs_feb
    FROM user_logs
    WHERE date BETWEEN DATE '2017-02-01' AND DATE '2017-02-28'
    GROUP BY msno
)
SELECT
    t.is_churn,
    COUNT(*) AS user_count,
    ROUND(AVG(COALESCE(f.total_secs_feb, 0)), 2) AS avg_total_secs_feb
FROM train t
LEFT JOIN feb_logs f
    ON t.msno = f.msno
GROUP BY t.is_churn
ORDER BY t.is_churn;
```

## 14. For all `train` users, what is the average number of transactions before March 2017 by churn status?
**Difficulty:** Medium

```sql
WITH txn_counts AS (
    SELECT
        msno,
        COUNT(*) AS txn_count
    FROM transactions
    WHERE transaction_date < DATE '2017-03-01'
    GROUP BY msno
)
SELECT
    t.is_churn,
    COUNT(*) AS user_count,
    ROUND(AVG(COALESCE(x.txn_count, 0)), 2) AS avg_txn_count_pre_march
FROM train t
LEFT JOIN txn_counts x
    ON t.msno = x.msno
GROUP BY t.is_churn
ORDER BY t.is_churn;
```

## 15. For churners vs non-churners, what percentage had at least one cancellation transaction before March 2017?
**Difficulty:** Medium

```sql
WITH cancel_flag AS (
    SELECT
        msno,
        1 AS had_cancel
    FROM transactions
    WHERE is_cancel = 1
      AND transaction_date < DATE '2017-03-01'
    GROUP BY msno
)
SELECT
    t.is_churn,
    COUNT(*) AS user_count,
    SUM(CASE WHEN c.had_cancel = 1 THEN 1 ELSE 0 END) AS users_with_cancel,
    ROUND(AVG(CASE WHEN c.had_cancel = 1 THEN 1.0 ELSE 0.0 END), 4) AS pct_with_cancel
FROM train t
LEFT JOIN cancel_flag c
    ON t.msno = c.msno
GROUP BY t.is_churn
ORDER BY t.is_churn;
```

## 16. Using each user’s latest transaction before March 2017, what is the churn rate by whether that latest transaction was a cancellation?
**Difficulty:** Hard

```sql
WITH latest_txn AS (
    SELECT
        msno,
        is_cancel,
        ROW_NUMBER() OVER (
            PARTITION BY msno
            ORDER BY transaction_date DESC,
                     membership_expire_date DESC,
                     actual_amount_paid DESC,
                     payment_plan_days DESC
        ) AS rn
    FROM transactions
    WHERE transaction_date < DATE '2017-03-01'
)
SELECT
    CASE
        WHEN l.msno IS NULL THEN 'no_transaction'
        WHEN l.is_cancel = 1 THEN 'latest_txn_cancelled'
        ELSE 'latest_txn_not_cancelled'
    END AS latest_txn_cancel_status,
    COUNT(*) AS user_count,
    ROUND(AVG(t.is_churn), 4) AS churn_rate
FROM train t
LEFT JOIN latest_txn l
    ON t.msno = l.msno
   AND l.rn = 1
GROUP BY
    CASE
        WHEN l.msno IS NULL THEN 'no_transaction'
        WHEN l.is_cancel = 1 THEN 'latest_txn_cancelled'
        ELSE 'latest_txn_not_cancelled'
    END
ORDER BY latest_txn_cancel_status;
```

## 17. Using each user’s latest transaction before March 2017, what is the churn rate by latest `payment_plan_days` bucket?
**Difficulty:** Hard

```sql
WITH latest_txn AS (
    SELECT
        msno,
        payment_plan_days,
        ROW_NUMBER() OVER (
            PARTITION BY msno
            ORDER BY transaction_date DESC,
                     membership_expire_date DESC,
                     actual_amount_paid DESC,
                     payment_plan_days DESC
        ) AS rn
    FROM transactions
    WHERE transaction_date < DATE '2017-03-01'
)
SELECT
    CASE
        WHEN l.msno IS NULL THEN 'no_transaction'
        WHEN l.payment_plan_days < 30 THEN '<30'
        WHEN l.payment_plan_days = 30 THEN '30'
        WHEN l.payment_plan_days BETWEEN 31 AND 89 THEN '31-89'
        ELSE '90+'
    END AS plan_days_bucket,
    COUNT(*) AS user_count,
    ROUND(AVG(t.is_churn), 4) AS churn_rate
FROM train t
LEFT JOIN latest_txn l
    ON t.msno = l.msno
   AND l.rn = 1
GROUP BY
    CASE
        WHEN l.msno IS NULL THEN 'no_transaction'
        WHEN l.payment_plan_days < 30 THEN '<30'
        WHEN l.payment_plan_days = 30 THEN '30'
        WHEN l.payment_plan_days BETWEEN 31 AND 89 THEN '31-89'
        ELSE '90+'
    END
ORDER BY plan_days_bucket;
```

## 18. For all `train` users, what is the churn rate by February 2017 listening-time quartile?
**Difficulty:** Hard

```sql
WITH feb_logs AS (
    SELECT
        msno,
        SUM(total_secs) AS total_secs_feb
    FROM user_logs
    WHERE date BETWEEN DATE '2017-02-01' AND DATE '2017-02-28'
    GROUP BY msno
),
base AS (
    SELECT
        t.msno,
        t.is_churn,
        COALESCE(f.total_secs_feb, 0) AS total_secs_feb
    FROM train t
    LEFT JOIN feb_logs f
        ON t.msno = f.msno
),
ranked AS (
    SELECT
        msno,
        is_churn,
        total_secs_feb,
        NTILE(4) OVER (ORDER BY total_secs_feb) AS engagement_quartile
    FROM base
)
SELECT
    engagement_quartile,
    COUNT(*) AS user_count,
    ROUND(MIN(total_secs_feb), 2) AS min_total_secs_in_quartile,
    ROUND(MAX(total_secs_feb), 2) AS max_total_secs_in_quartile,
    ROUND(AVG(is_churn), 4) AS churn_rate
FROM ranked
GROUP BY engagement_quartile
ORDER BY engagement_quartile;
```

## 19. For all `train` users, what is the churn rate by January-to-February 2017 listening change segment?
**Difficulty:** Hard

```sql
WITH jan_logs AS (
    SELECT
        msno,
        SUM(total_secs) AS total_secs_jan
    FROM user_logs
    WHERE date BETWEEN DATE '2017-01-01' AND DATE '2017-01-31'
    GROUP BY msno
),
feb_logs AS (
    SELECT
        msno,
        SUM(total_secs) AS total_secs_feb
    FROM user_logs
    WHERE date BETWEEN DATE '2017-02-01' AND DATE '2017-02-28'
    GROUP BY msno
),
base AS (
    SELECT
        t.msno,
        t.is_churn,
        COALESCE(j.total_secs_jan, 0) AS total_secs_jan,
        COALESCE(f.total_secs_feb, 0) AS total_secs_feb
    FROM train t
    LEFT JOIN jan_logs j
        ON t.msno = j.msno
    LEFT JOIN feb_logs f
        ON t.msno = f.msno
)
SELECT
    CASE
        WHEN total_secs_jan = 0 AND total_secs_feb = 0 THEN 'no_activity_both_months'
        WHEN total_secs_feb > total_secs_jan THEN 'increased_in_feb'
        WHEN total_secs_feb < total_secs_jan THEN 'decreased_in_feb'
        ELSE 'flat'
    END AS listening_change_segment,
    COUNT(*) AS user_count,
    ROUND(AVG(is_churn), 4) AS churn_rate
FROM base
GROUP BY
    CASE
        WHEN total_secs_jan = 0 AND total_secs_feb = 0 THEN 'no_activity_both_months'
        WHEN total_secs_feb > total_secs_jan THEN 'increased_in_feb'
        WHEN total_secs_feb < total_secs_jan THEN 'decreased_in_feb'
        ELSE 'flat'
    END
ORDER BY listening_change_segment;
```

## 20. Using each user’s latest transaction before March 2017, what is the churn rate by whether the user was active in the 7 days before their membership expiry date?
**Difficulty:** Hard

```sql
WITH latest_txn AS (
    SELECT
        msno,
        membership_expire_date,
        ROW_NUMBER() OVER (
            PARTITION BY msno
            ORDER BY transaction_date DESC,
                     membership_expire_date DESC,
                     actual_amount_paid DESC,
                     payment_plan_days DESC
        ) AS rn
    FROM transactions
    WHERE transaction_date < DATE '2017-03-01'
),
activity_flag AS (
    SELECT
        l.msno,
        MAX(CASE WHEN ul.total_secs > 0 THEN 1 ELSE 0 END) AS active_last_7d_before_expiry
    FROM latest_txn l
    LEFT JOIN user_logs ul
        ON l.msno = ul.msno
       AND ul.date BETWEEN DATE_SUB(l.membership_expire_date, 6) AND l.membership_expire_date
    WHERE l.rn = 1
    GROUP BY l.msno
)
SELECT
    CASE
        WHEN a.msno IS NULL THEN 'no_transaction'
        WHEN a.active_last_7d_before_expiry = 1 THEN 'active_last_7d_before_expiry'
        ELSE 'inactive_last_7d_before_expiry'
    END AS pre_expiry_activity_segment,
    COUNT(*) AS user_count,
    ROUND(AVG(t.is_churn), 4) AS churn_rate
FROM train t
LEFT JOIN activity_flag a
    ON t.msno = a.msno
GROUP BY
    CASE
        WHEN a.msno IS NULL THEN 'no_transaction'
        WHEN a.active_last_7d_before_expiry = 1 THEN 'active_last_7d_before_expiry'
        ELSE 'inactive_last_7d_before_expiry'
    END
ORDER BY pre_expiry_activity_segment;
```
