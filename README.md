# Song Popularity Prediction — Regression Project

Regression project predicting Spotify-style song popularity (0-100 scale) from audio features. Three algorithm families covered in the ML course are compared under CRISP-DM: Linear Regression, Decision Tree Regressor, and MLPRegressor (Linear Neural Network).

---

## Problem

A record company's marketing department wants a model that predicts the popularity of a song from its audio features, so promotion budgets can be allocated more efficiently. Songs with higher predicted popularity would receive more budget; songs with lower predicted popularity, less.

The prediction alone should not decide the budget — cultural context, artist notoriety, marketing spend, and timing all shape real-world popularity and are absent from the dataset. But used as one signal among several, a well-calibrated model is a useful piece of the decision.

## Dataset

- 18,835 songs
- 15 features (song name, duration, and 13 audio attributes: acousticness, danceability, energy, instrumentalness, key, liveness, loudness, audio mode, speechiness, tempo, time signature, audio valence)
- Target: `song_popularity` (0-100)
- `originality` has 70% missing values and is dropped
- `instrumentalness` has fewer than 1% missing and is imputed with the median

## Result

| Model | MAE (test) | RMSE (test) | R² (test) |
|---|---|---|---|
| Linear Regression | 15.33 | 19.23 | 0.002 |
| **Decision Tree (tuned)** | **15.17** | **19.07** | **0.018** |
| MLPRegressor | 15.31 | 19.39 | -0.015 |

The three models sit within one MAE point of each other. The absolute error hovers around 15 popularity points on the 0-100 scale, and R² stays close to zero. This is the natural ceiling of what audio features alone can predict — song popularity is driven by many factors this dataset does not capture (marketing, artist notoriety, playlist inclusion, cultural context). No single audio feature correlates with `song_popularity` above 0.11 in absolute value, which forecasts the low R² before any model is trained.

The tuned Decision Tree was chosen for the final deployment. It has the lowest MAE, the highest R², and its `GridSearchCV` tuning of `max_depth` and `min_samples_leaf` kept training and test error close (train MAE 15.21 vs test MAE 15.17). The MLPRegressor, by contrast, produced negative R² on the test set and predictions that occasionally exceed the 0-100 range — a sign that a neural network has more flexibility than the audio signal justifies.

## Methodology

The work follows CRISP-DM:

1. Business Understanding. Marketing use case and problem definition.
2. Data Understanding. Target distribution analysis, feature distributions, boxplots for outliers, correlation heatmap.
3. Data Preparation. Drop `originality`, impute `instrumentalness`, drop duplicates, feature engineering (`song_duration_min`).
4. Modeling. Three algorithms: Linear Regression (with RFECV feature selection and standardised coefficients), Decision Tree Regressor (tuned with `GridSearchCV`), MLPRegressor. Split → outlier removal on training set only → `StandardScaler` fit on train.
5. Evaluation. Comparison of the three models on the held-out test set with MAE, RMSE, R² and Max Error. (MAPE was excluded because 272 songs have `song_popularity = 0`, which makes the metric divide by zero.)
6. Deployment. Best model persisted with `joblib`. Predictions on the test set exported to CSV.

## Why these three algorithms

The mix matches the three families of regressors taught in the ML course:

- Linear Regression — a linear parametric model, the interpretable baseline.
- Decision Tree Regressor — a non-parametric model that captures non-linear splits.
- MLPRegressor — a neural network baseline (from the "Linear Neural Networks" section of the course).

Comparing across families shows whether the problem is linear (Linear Regression wins), tree-like (Decision Tree wins), or complex enough to reward a neural network. Here, all three land within one MAE point of each other, and R² stays near zero for all of them. The ceiling is set by the data, not by the model choice — and the MLP's negative test R² tells us that added complexity can actually hurt on datasets this noisy.

## What surprised me during the analysis

Popularity is only weakly correlated with any single audio feature (top absolute correlation ~0.11 — instrumentalness). This was visible in the correlation heatmap before any modelling, and it correctly predicted that the linear model would have a near-zero R². It is a useful reminder that EDA is not decoration — a two-minute look at correlations forecasted the ceiling of the whole modelling effort.

`loudness` and `energy` correlate at ~0.76 with each other. Multicollinearity of this scale distorts linear regression coefficients and can flip their signs. Here both variables were kept to match the course brief (compare three model families), but a production version would drop one or move to a regularised model.

The Decision Tree default (unlimited depth) gives a training MAE close to zero and a test MAE far higher — the textbook overfitting signature. Tuning `max_depth` and `min_samples_leaf` with `GridSearchCV` closes that gap almost completely at the cost of about one MAE point on test.

## In production

If this were productised for the record company:

1. Ingestion. Daily pull of new tracks from a streaming API, joined with the audio features (already available via the Spotify API).
2. Scoring. Batch job runs the persisted Decision Tree over new tracks.
3. Action. Predicted popularity feeds one input of a broader promotion-budget scoring model — never used as the sole determinant.
4. Monitoring. Track (a) MAE on the previous month's actualised popularity, (b) distribution drift on `loudness` and `tempo` (both trend over years in music production), (c) coverage of the exclusion cases (tracks with `song_popularity = 0` behave differently from the rest and probably need a separate model).
5. Retraining. Quarterly retrain with the latest month's actualised popularity as new ground truth.

## Scope and limitations

The dataset contains only audio features. Real-world popularity depends on marketing investment, artist notoriety, playlist placement, cultural context, and release timing — none of which are here. The ceiling of any purely audio-based model on this dataset is around R² ≈ 0.02 on unseen data. Reporting a much higher R² would only be possible through data leakage or overfitting.

Songs with `song_popularity = 0` are unrated or unreleased tracks. They behave differently from the main distribution. A two-stage model (first classify rated vs unrated, then predict popularity conditional on rated) would likely improve overall accuracy but was out of scope.

## Stack

Python 3.10+, pandas, numpy, scikit-learn (LinearRegression, DecisionTreeRegressor, MLPRegressor, RFECV, GridSearchCV, StandardScaler, learning_curve), matplotlib, seaborn, joblib.

---

Developed as part of Machine Learning for Marketing at NOVA IMS (Lisbon, 2023) and refactored for public portfolio sharing.
Author: Camilla Alves.
