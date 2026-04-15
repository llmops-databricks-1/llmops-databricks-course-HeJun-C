# Churn Analysis — Domain Knowledge Skill

General best practices for exploratory data analysis of consumer churn when
you have user demographics, engagement/usage logs, and transaction history.

---

## 1. Key Analysis Dimensions

### Demographics
- **Age**: Compare age distributions between churned and retained users.
  Watch for outlier values (negatives, implausibly large numbers); filter
  or cap before analysis.
- **Gender**: Calculate churn rates by gender category (including "unknown").
  Large proportions of unknown gender should be treated as a distinct segment,
  not dropped.
- **City / Region**: Identify geographic variation in churn. Consider grouping
  low-count cities to avoid noisy estimates.
- **Registration Channel**: Different sign-up methods may attract users with
  different retention profiles.

### Engagement Intensity
- **Listening Frequency**: How many days per week/month does the user engage?
  Users with low frequency are typically at higher churn risk.
- **Session Depth**: Among songs started, what fraction is played to
  completion (>98.5%)? High skip rates suggest dissatisfaction.
- **Unique Content**: Users exploring a wide catalog may be more engaged;
  users listening to very few unique songs may be disengaged or niche.
- **Total Listening Time**: Aggregate seconds over a period. Consider both
  mean and trend (declining usage is a strong churn signal).

### Payment Behavior
- **Plan Type & Price**: Users on shorter or cheaper plans may churn more
  easily due to lower switching cost.
- **Discount Sensitivity**: Compare list price vs. actual amount paid. Large
  discounts may indicate price-sensitive users.
- **Auto-Renew Status**: Users without auto-renew must actively choose to
  stay, making them structurally more likely to churn.
- **Cancellation History**: Prior cancellations (even if followed by renewal)
  signal risk.

### Tenure & Lifecycle
- **Registration Date → Tenure**: Compute tenure at the observation date.
  New users and very long-tenured users may have different churn dynamics.
- **Cohort Analysis**: Group users by registration month/quarter and compare
  retention curves across cohorts.

---

## 2. Recommended Analytical Patterns

### Segment Comparison (Churned vs. Retained)
For every feature, compare the distribution between `is_churn=1` and
`is_churn=0`. Use:
- Averages, medians, and percentiles for continuous features.
- Frequency tables and proportion charts for categorical features.

### Cohort Analysis
Group users by a shared time-based attribute (e.g., registration quarter)
and measure churn rate within each cohort to detect temporal trends.

### Correlation / Association
- Compute point-biserial correlation between numeric features and the churn
  binary label.
- Use chi-squared or Cramér's V for categorical features vs. churn.
- Be cautious interpreting correlation as causation.

### Trend Analysis
For users with time-series data (daily logs, repeated transactions):
- Compare recent activity levels to historical baseline.
- Look for declining engagement trends in the weeks before churn.

### Feature Importance (Lightweight)
Fit a simple model (logistic regression or a single decision tree) and
inspect coefficients or feature importances. This is exploratory, not
production modeling — use it to prioritize which features warrant deeper
investigation.

---

## 3. Aggregation Strategies

- **Always aggregate event-level tables to user level** before joining with
  the churn label. Joining raw event rows with the label duplicates the
  label across rows and inflates apparent sample size.
- Common user-level aggregations for engagement logs:
  - Total / mean / median listening seconds
  - Count of active days
  - Mean songs played per session
  - Completion rate (num_100 / total songs started)
  - Unique songs ratio (num_unq / total songs)
- Common user-level aggregations for transactions:
  - Number of transactions
  - Average payment amount
  - Whether auto-renew was ever enabled
  - Total plan days purchased
  - Most recent transaction recency

---

## 4. Statistical Approaches

| Comparison Type | Suggested Method |
|---|---|
| Continuous feature, churned vs. retained | t-test (if roughly normal) or Mann-Whitney U |
| Categorical feature vs. churn | Chi-squared test, Cramér's V |
| Multiple features at once | Logistic regression coefficients |
| Feature ranking | Decision tree feature importance |
| Distribution visualization | Histograms, box plots, violin plots |

---

## 5. Common Pitfalls

- **Survivorship Bias**: If the observation window is too short, users who
  *would* churn but haven't yet are counted as retained, biasing churn rates
  downward.
- **Confounding Variables**: A feature may correlate with churn only because
  it correlates with a third variable (e.g., tenure).
- **Outlier Contamination**: Fields like age may contain sentinel or garbage
  values. Always inspect value ranges and filter before computing statistics.
- **Conflating Cancellation with Churn**: A transaction-level cancellation
  flag does not necessarily mean the user churned. Churn is defined at the
  membership level (did the user fail to renew within 30 days of expiration).
- **Plan Length Differences**: Users on annual plans and monthly plans have
  very different renewal cadences. Comparing raw churn rates without
  controlling for plan length can be misleading.
- **Ignoring Unknown Categories**: Dropping rows with unknown gender or city
  discards a large portion of users and introduces selection bias. Treat
  unknowns as a valid category.
