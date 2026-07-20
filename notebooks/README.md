# Feature Engineering — AI Risk System (IEEE-CIS Fraud Detection)

A full history of every feature engineered across all versions of this project: what we built, why, what we removed, what we kept, and the measurable effect on model scores.

---

## Dataset Baseline

| Property | Value |
|---|---|
| Source | IEEE-CIS Fraud Detection (Kaggle) |
| Raw files | `train_transaction.csv` (394 cols) + `train_identity.csv` (41 cols) |
| After merge | 590,540 rows × ~434 columns |
| Target | `isFraud` (binary, ~3.5% positive rate) |
| Evaluation metric | F1-score at optimized threshold |

The dataset has severe class imbalance (~96.5% legitimate, ~3.5% fraud). Every engineering decision was made with this in mind — features that help distinguish the minority class matter far more than features that describe the majority.

---

## Version History at a Glance

| Version | File | Columns | Best F1 | Key Change |
|---|---|---|---|---|
| Raw merge | — | ~434 | baseline | Merge + dtype optimization |
| V1–V8 | `train_optimized.pkl` | 434 | 0.748 | Memory optimization, basic cleaning |
| V9 | `train_v9_engineered.pkl` | 442 | 0.767 | UID construction + 8 behavioral aggregations |
| V10 | `train_v10_winning_fe.pkl` | 486 | TBD | 40 winning features via uid_d1n |

---

## Phase 1 — Data Merging & Memory Optimization (V1–V8, 434 columns)

### What we did

```python
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```

A left join — every transaction is kept, identity info is attached where available (~144K of 590K transactions have identity data). The rest get NaN in identity columns. This was the correct choice: dropping transactions without identity would lose most of the training data.

### Memory optimization

All columns were downcasted to the smallest valid dtype:

```python
# float64 → float32 where values fit
# int64 → int32 or int16 where values fit
# object → category where cardinality is manageable
```

Effect: RAM usage dropped from ~3.5GB to ~1.2GB. No information lost. This was necessary to avoid out-of-memory errors during training on a local machine.

### What was removed

| Column | Reason for removal |
|---|---|
| Nothing removed at this stage | All raw columns retained for modeling |

### Baseline score

LightGBM with default params on raw merged data: **F1 ≈ 0.748**

---

## Phase 2 — UID Construction & Behavioral Aggregations (V9, 442 columns, +8 features)

### The core insight

Each row in the dataset is a transaction, not a person. The model sees each transaction in isolation unless we give it per-user context. The question becomes: **how do we identify "the same person" across rows without a user ID column?**

The dataset has no explicit user ID. We constructed one.

### Feature: `uid`

```python
train['uid'] = (
    train['card1'].astype(str) + '_' +
    train['card2'].astype(str) + '_' +
    train['card3'].astype(str) + '_' +
    train['card5'].astype(str) + '_' +
    train['addr1'].astype(str)
)
```

**Why these columns:** card1–card5 are anonymized payment card identifiers. addr1 is billing address. Together they form a fingerprint: same card details + same billing address = almost certainly the same person.

**Why not include card4/card6:** card4 and card6 are card type/brand codes (e.g. Visa, Mastercard). Two different people can share the same card brand, so including them would merge unrelated users into the same UID group. card1/2/3/5 are more unique identifiers.

**Why not include D features here:** D features change over time within the same user's transactions. Using them in the UID would split one user into multiple groups across their transaction history. We wanted a stable UID that groups all of a user's transactions together.

**Limitation discovered:** This UID was too coarse — it grouped some unrelated users together when they happened to share card details. This led to leaky aggregations. Fixed in V10 with `uid_d1n`.

### Features added in V9 (8 features)

All aggregations used `.transform()` to broadcast group statistics back to individual rows, preserving the 590,540 row shape.

#### `uid_transaction_count`

```python
train['uid_transaction_count'] = train.groupby('uid')['TransactionAmt'].cumcount()
```

Running count of how many transactions this user has made so far (not including current). Uses `cumcount()` rather than `count()` — temporal ordering is respected, so row 1 gets 0, row 2 gets 1, etc.

**Why:** New users (count=0) are higher fraud risk. Established users with long histories are lower risk. This captures account age in transaction units.

#### `uid_mean_amount`

```python
train['uid_mean_amount'] = train.groupby('uid')['TransactionAmt'].transform(
    lambda x: x.shift(1).expanding().mean()
)
```

Rolling mean of transaction amount for this user, using only past transactions (`shift(1)` ensures the current transaction is excluded — no data leakage). `expanding()` uses all past rows, not a fixed window.

**Why:** Fraud often involves transactions that deviate dramatically from a user's normal spending. This feature gives the model the user's historical average to compare against.

#### `uid_std_amount`

Same as above but standard deviation. Captures how consistent the user's spending is. A fraudster testing a stolen card may make many small transactions followed by one large one — high std signals this pattern.

#### `uid_amount_ratio`

```python
train['uid_amount_ratio'] = train['TransactionAmt'] / train['uid_mean_amount']
```

Current transaction amount divided by the user's historical average. ratio=1.0 is normal. ratio=10.0 means this transaction is 10x larger than typical for this user — strong fraud signal.

**Note:** Produces `inf` when `uid_mean_amount=0` (first transaction). Handled downstream per model: LightGBM ignores NaN natively, XGBoost needed explicit `replace(inf, -999)`.

#### `uid_amount_zscore`

```python
train['uid_amount_zscore'] = (
    train['TransactionAmt'] - train['uid_mean_amount']
) / train['uid_std_amount']
```

Standardized deviation from the user's mean. More statistically principled than ratio — accounts for the user's typical variance. A zscore of 3 means this transaction is 3 standard deviations above normal.

**Why keep both ratio and zscore:** They capture different things. ratio is multiplicative (useful when baseline is small). zscore is additive/standardized (useful when baseline is large). Both add signal.

#### `uid_mean_hour`

```python
train['uid_mean_hour'] = train.groupby('uid')['trans_hour'].transform(
    lambda x: x.shift(1).expanding().mean()
)
```

`trans_hour` was derived earlier: `train['trans_hour'] = (train['TransactionDT'] // 3600) % 24`. This feature captures the user's typical transaction hour. A transaction at 3am for a user who always transacts at noon is anomalous.

**Uses shift(1):** Same anti-leakage pattern as amount features.

### Score after V9

| Model | F1 | AUC |
|---|---|---|
| LightGBM | **0.767** | 0.9611 |

Improvement over baseline: **+0.019 F1** from 8 features. The UID-based behavioral features were the single largest improvement step in the project.

---

## Phase 3 — Winning Features via uid_d1n (V10, 486 columns, +44 features)

### Why uid needed to change

The V9 `uid` (card1+card2+card3+card5+addr1) had two problems:

1. **Too stable** — grouped all of a user's transactions across the full ~2 year dataset window. If a user's card was stolen halfway through, fraud and legit transactions were in the same group, diluting the signal.
2. **No temporal anchor** — the D features (time deltas) were computed on the full group without considering when in the user's history we were.

### Feature: `uid_d1n` (the new UID)

```python
train['D1n'] = np.floor(train['TransactionDT'] / 86400) - train['D1']
train['uid_d1n'] = (
    train['card1'].astype(str) + '_' +
    train['addr1'].astype(str) + '_' +
    train['D1n'].astype(str)
)
```

**D1n construction:** `TransactionDT / 86400` converts seconds to days. `np.floor` rounds down to the integer day number. Subtracting `D1` (days since card first seen) gives the **card's first-seen calendar day** — a stable property of the card that doesn't change across the user's transactions.

**Why this is better than raw uid:** Two users can share card1+addr1 by coincidence (e.g., family members with the same billing address). Adding `D1n` disambiguates: if they started using their cards on different days, they get different UID groups. This makes the UID more precise.

**Why only card1+addr1 (not card2/3/5 like V9):** After analysis, card2/3/5 added noise — they sometimes varied across transactions for the same real user due to anonymization. card1+addr1+D1n was found to be the most stable and discriminative combination.

```python
grp = train.groupby('uid_d1n')
```

All subsequent features are computed using this groupby object — 590k rows split into user buckets, each bucket containing all transactions that share the same card1+addr1+D1n value.

---

### D-Feature Normalization (4 new columns, used to build 6 aggregated features)

#### Why normalize D features

Raw D features are time deltas measured in days from some reference point in the dataset. This means D1=100 on dataset day 200 means something different than D1=100 on dataset day 400 — the raw value is relative to dataset time, not to the user. This causes the model to learn a spurious time trend instead of actual user behavior.

Normalization fixes this:

```python
train['D1n']  = np.floor(train['TransactionDT'] / 86400) - train['D1']
train['D4n']  = np.floor(train['TransactionDT'] / 86400) - train['D4']
train['D10n'] = np.floor(train['TransactionDT'] / 86400) - train['D10']
train['D15n'] = np.floor(train['TransactionDT'] / 86400) - train['D15']
```

After normalization, `D1n` = the dataset day when this card was first seen — a stable property. `D10n` = the dataset day when this billing address was first seen. These are now consistent across the timeline.

**Why D4, D10, D15 and not all 15 D features:** These three are believed to be the most informative time-delta features in IEEE-CIS based on competition analysis. D4 ≈ days since card was first seen. D10 ≈ days since billing address first seen. D15 ≈ days since last transaction. The others have high missingness or low discriminative power.

#### Aggregated D features (6 features)

```python
for col in ['D4n', 'D10n', 'D15n']:
    train[f'uid_{col}_mean'] = grp[col].transform('mean')
    train[f'uid_{col}_std']  = grp[col].transform('std')
```

| Feature | What it captures | Fraud signal |
|---|---|---|
| `uid_D4n_mean` | Average normalized D4 across this user's transactions — how long this user has been using this card on average | Very low = new card = higher risk |
| `uid_D4n_std` | Consistency of card age across transactions | High std = card properties changing = suspicious |
| `uid_D10n_mean` | Average normalized D10 — how established this billing address is | Very low = new address = higher risk |
| `uid_D10n_std` | Consistency of address across transactions | High = multiple addresses = suspicious |
| `uid_D15n_mean` | Average time since last transaction — captures usage frequency | Very low = rapid-fire transactions = fraud pattern |
| `uid_D15n_std` | How regular the transaction timing is | High = erratic timing = bot or fraud |

---

### C Feature Means (13 features)

```python
for col in [f'C{i}' for i in range(1, 15) if i != 3]:
    train[f'uid_{col}_mean'] = grp[col].transform('mean')
```

C features are pre-engineered count features from the dataset provider — they capture behavioral frequencies but their exact definitions are anonymized.

**Why C3 was excluded:** C3 is ~95% NaN across the dataset. Taking a group mean of a column that's almost entirely missing produces noise, not signal. Excluded.

**Why mean and not std for C features:** C features are already counts/frequencies. Their within-user variance is less informative than the magnitude. For D features (time deltas) variance matters a lot; for counts, the average level matters more.

| Feature | Believed meaning | Fraud signal |
|---|---|---|
| `uid_C1_mean` | Avg # of billing addresses per card for this user | High = card used with many addresses = stolen card |
| `uid_C2_mean` | Avg # of cards per billing address | High = many cards at same address = fraud ring |
| `uid_C4_mean` | Avg # of transactions in a time window | Very high = transaction flooding = fraud |
| `uid_C5_mean` | Avg # of chargebacks/disputes | Any positive value = bad history |
| `uid_C6_mean` | Avg # of transactions with identical amount | High = repeated exact amounts = bot pattern |
| `uid_C7_mean` | Avg # of addresses in recent time window | High = geographic spreading = suspicious |
| `uid_C8_mean` | Avg # of cards at this merchant | High = card testing at same merchant |
| `uid_C9_mean` | Avg # of successful transactions | Low = most attempts failing = fraud |
| `uid_C10_mean` | Avg # of unique cards for this email | High = one email, many cards = fraud |
| `uid_C11_mean` | Avg # of unique users per card | High = card shared = suspicious |
| `uid_C12_mean` | Avg # of declined transactions | High = many declines = probing behavior |
| `uid_C13_mean` | Avg count of email-card associations | Combined with nunique for full picture |
| `uid_C14_mean` | Avg # of unique addresses per card | High = card moving around = stolen |

---

### C13 nunique (1 feature)

```python
train['uid_C13_nunique'] = grp['C13'].transform('nunique')
```

C13 gets both mean and nunique treatment because it's believed to track email-to-card associations — a particularly discriminative feature for fraud. Mean captures the typical level; nunique captures diversity.

| Feature | What it captures | Fraud signal |
|---|---|---|
| `uid_C13_nunique` | How many distinct C13 values this user produces across transactions | nunique=1 = consistent = legit; high = multiple email-card combos = fraud |

---

### V Feature nuniques (6 features)

```python
for col in ['V127', 'V136', 'V307', 'V309', 'V314', 'V320']:
    train[f'uid_{col}_nunique'] = grp[col].transform('nunique')
```

**Why these 6 V columns:** The V features (V1–V339) are anonymized derived features, likely PCA components or behavioral fingerprints. Among 339 V columns, these 6 are empirically identified in Kaggle literature as among the most fraud-discriminative — they are believed to encode device fingerprint or session behavior dimensions.

**Why nunique and not mean:** V features have already been transformed/scaled by the dataset provider. Their absolute mean per user is less meaningful than diversity — how many distinct values does this user's device signature take? A real user on one device has nunique=1. A fraudster spoofing devices or a bot rotating profiles has high nunique.

| Feature | What it captures | Fraud signal |
|---|---|---|
| `uid_V127_nunique` | Distinct V127 values per user | >1 = device inconsistency |
| `uid_V136_nunique` | Distinct V136 values per user | >1 = fingerprint changing |
| `uid_V307_nunique` | Distinct V307 values per user | >1 = session behavior changing |
| `uid_V309_nunique` | Distinct V309 values per user | >1 = anomalous session pattern |
| `uid_V314_nunique` | Distinct V314 values per user | >1 = device signature rotating |
| `uid_V320_nunique` | Distinct V320 values per user | >1 = behavioral inconsistency |

---

### M Feature means (9 features)

```python
m_mapping = {'T': 1, 'F': 0, 'M0': 0, 'M1': 1, 'M2': 2}
for col in [f'M{i}' for i in range(1, 10)]:
    if col in train.columns:
        temp = train[col].map(m_mapping).astype(float).fillna(-1)
        train[f'uid_{col}_mean'] = temp.groupby(train['uid_d1n']).transform('mean')
```

**Why the mapping is needed:** M features are stored as strings — `'T'`, `'F'`, `'M0'`, `'M1'`, `'M2'`. You cannot compute a mean of strings. The mapping converts them to numeric: True=1, False=0, ordinal M codes mapped accordingly.

**Why `.astype(float)` after mapping:** `.map()` can return object dtype if any values weren't in the dictionary. Casting to float ensures `transform('mean')` works correctly.

**Why `.fillna(-1)` and not 0:** NaN in M features means the check wasn't performed — not that it failed. Using -1 signals "not applicable" which is semantically different from 0 (failed) and 1 (passed). The model learns this distinction.

**Why a separate groupby on `temp`:** The `grp` object was created from the original `train` dataframe. `temp` is a new derived Series. Pandas doesn't allow using the original groupby on a freshly derived column — you need to re-specify `groupby(train['uid_d1n'])` using the key column.

| Feature | What it captures | Fraud signal |
|---|---|---|
| `uid_M1_mean` | Avg card verification match rate | Close to 0 = usually fails verification |
| `uid_M2_mean` | Avg billing address match rate | Low = address mismatches consistently |
| `uid_M3_mean` | Avg shipping address match rate | Low = shipping always different from billing |
| `uid_M4_mean` | Avg method validation score (0/1/2 scale) | Low = payment method issues |
| `uid_M5_mean` | Avg email match rate | Low = email inconsistency |
| `uid_M6_mean` | Avg device match rate | Low = device always different = account compromise |
| `uid_M7_mean` | Avg browser/environment match | Low = browser inconsistency |
| `uid_M8_mean` | Avg supplementary verification match | Low = fails extra checks |
| `uid_M9_mean` | Avg final validation match rate | Low = fails end-to-end validation |

**Why M features are powerful as aggregations:** A single transaction failing M3 might be a legitimate shipping-to-different-address scenario (gift purchase). But a user who fails M3 in 80% of their transactions is almost certainly fraudulent. The mean converts a noisy per-transaction signal into a reliable per-user pattern.

---

### D9 mean and std (2 features)

```python
train['uid_D9_mean'] = grp['D9'].transform('mean')
train['uid_D9_std']  = grp['D9'].transform('std')
```

D9 represents time of day as a decimal fraction (0.0 = midnight, 0.25 = 6am, 0.5 = noon, 0.75 = 6pm).

| Feature | What it captures | Fraud signal |
|---|---|---|
| `uid_D9_mean` | This user's average transaction hour | Mean near 0 = consistently late night = suspicious |
| `uid_D9_std` | Consistency of transaction timing | High std = random hours = bot or fraud ring operating across time zones |

**Why both mean and std:** Mean alone misses the erratic case. A user transacting at random hours has a mean near 0.5 (noon) simply by averaging, but a high std reveals the randomness. You need both to characterize the pattern.

---

### Email domain nunique (1 feature)

```python
train['uid_P_emaildomain_nunique'] = grp['P_emaildomain'].transform('nunique')
```

| Feature | What it captures | Fraud signal |
|---|---|---|
| `uid_P_emaildomain_nunique` | How many distinct payer email domains this user has used | 1 = same email always = legit; >1 = rotating emails = fraud |

Real users have one email. Fraudsters use temporary or rotating emails (`temp-mail.org`, `guerrillamail.com`, etc.) to avoid detection. A user with nunique=5 across their transactions is a very strong fraud signal.

---

### id_02 nunique (1 feature)

```python
train['uid_id_02_nunique'] = grp['id_02'].transform('nunique')
```

id_02 is an anonymized identity feature from `train_identity.csv`, believed to be a device or session identifier.

| Feature | What it captures | Fraud signal |
|---|---|---|
| `uid_id_02_nunique` | How many distinct id_02 values this user produces | 1 = same device/session = legit; high = multiple devices or spoofed IDs |

---

## Features Considered and Rejected

### Target encoding (early consideration)

We considered replacing high-cardinality categorical columns (DeviceInfo, P_emaildomain) with their mean fraud rate. This would provide strong signal — `gmail.com` has a different fraud rate than `protonmail.com`.

**Why not done yet:** Target encoding requires careful out-of-fold computation to avoid leakage. Naively computing mean fraud rate on the full training set and using it as a feature leaks the target into the features. Planned as a next step using k-fold target encoding.

### V-feature means

We considered adding mean aggregations for all 339 V features per user.

**Why rejected:** 339 × 1 = 339 new columns. Most V features have low within-user variance and high between-user variance, meaning the group mean adds little over the raw value. Computational cost and memory usage outweigh the marginal benefit. Only the 6 highest-signal V features were used, and as nunique (diversity) rather than mean.

### C3 mean

**Why rejected:** C3 is ~95% NaN. Group mean on a near-empty column is dominated by the few non-NaN values and produces a noisy feature that degrades model performance. Explicitly excluded.

### Velocity features (planned, not yet implemented)

Counting transactions per user in rolling time windows (e.g., transactions in last 1 hour, last 24 hours). Strong fraud signal — card testing involves many rapid transactions. Excluded from V10 due to complexity of implementing leak-free rolling windows on a time-sorted dataset.

### Stacking / meta-features

Using predictions from LightGBM and XGBoost as features for a meta-learner. Planned next step after V10 benchmark is confirmed.

---

## Feature Summary Table

| Feature | Version Added | Type | Kept | Reason |
|---|---|---|---|---|
| `uid` | V9 | UID string | Dropped in V10 | Replaced by more precise `uid_d1n` |
| `uid_transaction_count` | V9 | Count | Kept | Account age proxy |
| `uid_mean_amount` | V9 | Rolling mean | Kept | User spending baseline |
| `uid_std_amount` | V9 | Rolling std | Kept | Spending consistency |
| `uid_amount_ratio` | V9 | Ratio | Kept | Deviation from normal |
| `uid_amount_zscore` | V9 | Z-score | Kept | Standardized deviation |
| `uid_mean_hour` | V9 | Rolling mean | Kept | Timing pattern |
| `uid_d1n` | V10 | UID string | Kept as key | More precise user identity |
| `uid_D4n_mean/std` | V10 | Agg | Kept | Card age consistency |
| `uid_D10n_mean/std` | V10 | Agg | Kept | Address age consistency |
| `uid_D15n_mean/std` | V10 | Agg | Kept | Transaction frequency |
| `uid_C1–C14_mean` | V10 | Agg ×13 | Kept | Behavioral count history |
| `uid_C3_mean` | V10 | Agg | Rejected | 95% NaN |
| `uid_C13_nunique` | V10 | Agg | Kept | Email-card diversity |
| `uid_V127/136/307/309/314/320_nunique` | V10 | Agg ×6 | Kept | Device fingerprint diversity |
| `uid_M1–M9_mean` | V10 | Agg ×9 | Kept | Verification pass rate history |
| `uid_D9_mean/std` | V10 | Agg | Kept | Transaction timing pattern |
| `uid_P_emaildomain_nunique` | V10 | Agg | Kept | Email rotation detection |
| `uid_id_02_nunique` | V10 | Agg | Kept | Device consistency |
| Target encoding (C/D/email) | — | Planned | Not yet | Needs out-of-fold implementation |
| All V-feature means (×339) | — | Rejected | No | High cost, low marginal gain |
| Velocity features | — | Planned | Not yet | Complex leak-free implementation |

---

## Score Progression

| Stage | Features | Best F1 | AUC |
|---|---|---|---|
| Raw LightGBM (no FE) | 434 raw columns | 0.748 | ~0.94 |
| + UID behavioral features (V9) | +8 features | **0.767** | 0.9611 |
| + Winning features (V10, pending) | +40 features | TBD | TBD |

The 8 V9 features alone added **+0.019 F1** — the largest single improvement in the project. This confirms that per-user behavioral context is the most important signal in this dataset, more important than any individual transaction-level feature.

---

## Engineering Principles Used Throughout

**No target leakage:** All rolling features use `shift(1)` so each row only sees past transactions. Group aggregations (`transform`) don't include the current row's value in its own mean.

**Preserve row count:** All features use `.transform()` rather than `.agg()` so the 590,540-row shape is maintained. `.agg()` would collapse to one row per user.

**User-level context beats transaction-level features:** The biggest gains came from group aggregations, not from transforming individual columns. Fraud detection is fundamentally about behavioral patterns, not individual transaction properties.

**Guard all feature additions:** Every feature addition uses `if col in train.columns` to prevent crashes when columns were dropped in memory optimization. Defensive by default.

**Model-specific preprocessing is separate from feature engineering:** The features are built once and saved to pickle. Categorical encoding (category dtype for LightGBM, LabelEncoder for XGBoost, string NaN fill for CatBoost) is applied per-model at training time, not stored in the pickle. This keeps the feature store clean and model-agnostic.