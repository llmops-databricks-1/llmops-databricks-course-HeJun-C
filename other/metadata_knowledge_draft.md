Section 1: Tables Overview

Table Name: members
Description: User information. Note that not every user in the dataset is available in other datasets.
Column Explanations:
msno: user id, string type
city: city id, integer type, as we use numbers to represent cities
bd: age, integer type,  note: this column has outlier values ranging from -7000 to 2015, please use your judgement
gender: represent sex, string type, has three types of values (“male”, “female”, null). We treat null values in this column as “unknown”
registered_via: registration method, integer type
registration_init_time: initial registration date, date type, with 2011-09-15 format
Row Grain: Each row represents a unique user. Each user only has one single row. The msno is the user id primary key for this table.
Null Values: No null values in the table except for the meaning for null values in the gender column representing “Unknown”

Table Name: user_logs
Description: Daily user logs describing listening behaviors of a user. Data collected until 2/28/2017
Column Explanations:
msno: user id, string type
date: date when the streaming behavior happened, date type, with 2011-09-15 format
num_25: # of songs played less than 25% of the song length, integer type
num_50: # of songs played between 25% to 50% of the song length, integer type
num_75: # of songs played between 50% to 75% of the song length, integer type
num_985: # of songs played between 75% to 98.5% of the song length, integer type
num_100: # of songs played over 98.5% of the song length, integer type
num_unq: # of unique songs played, integer type
total _secs: total seconds played, double type
Row Grain: Potential multiple rows per user; typically one row per user per day. No obvious primary key.
Null Values: No null values in the table

Table Name: transactions
Description: transactions of users up until 2/28/2017
Column Explanations:
msno: user id, string type
payment_method_id: payment method, integer type
payment_plan_days: length of membership plan in days, integer type
plan_list_price: the list price of the plan, integer type
actual_amount_paid: actual amount paid for the transaction, integer type
is_auto_renew: binary values representing whether the plan is auto renew plan or not (1: yes, 0: no), integer type
transaction_date: date when the transaction happened, date type, with 2011-09-15 format
membership_expire_date: date when the membership expires, date type, with 2011-09-15 format
is_cancel: binary values representing whether or not the user canceled the membership in this transaction (1: yes, 0: no), integer type
Row Grain: Potential multiple rows per user, one row per transaction record, no obvious primary key.
Null Values: No null values in the table

Table Name: train
Description: containing the user ids and whether they have churned for March 2017
Column Explanations:
msno: user id, string type
is_churn: this is the target variable, churn is defined as whether the user did not continue the subscription within 30 days of expiration. is_churn = 1 means churn,is_churn = 0 means renewal, integer type
Row Grain: One row per user
Null Values: No null values in the table

Section 2: Table Join Logic:
joins on msno
train ↔ members: roughly user-level join
train ↔ transactions: one-to-many
train ↔ user_logs: one-to-many
avoid joining raw transactions and user_logs as it can multiply rows not useful for analysis
no all msno exist in all tables; user inner join for analysis to only analyze users in all the useful tables

Section 3: Business Rules
is_cancel is not the same as is_churn
bd has obvious bad/outlier values and should be treated carefully
gender = null should be interpreted as unknown
user_logs and transactions tables usually need aggregation before joining to user-level churn analysis
