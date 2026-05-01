## 3 notebooks I found insightful/helpful:
1. https://www.kaggle.com/code/aliafzal9323/s6e4-0-970-stacked-lgb-xgb-cat-feature-engine
2. https://www.kaggle.com/code/mohit78241/s6e4-ensemble-voting-transfer-0-981-lb
3. https://www.kaggle.com/code/manasi197/s6e4-xgboost

s6e4-xgboost: Introduced the domain threshold flags and magic_score composite feature that I directly used. 

s6e4-stacked-lgb-xgb-cat: Showed the value of blending the original source dataset with competition data and using a LGB+XGB+CatBoost stack with a meta-learner. I'm currently using only XGBoost, so adding more base models is a major gain still available to me.

s6e4-ensemble-voting-transfer: Highest scorer, showed that combining multiple strong submissions via conditional voting (consensus when models agree, stronger model as fallback when they don't) outperforms any single model. Once I have a few solid models, ensembling predictions is worth doing over picking one winner.

# HOMEWORK 2
## EDA and models notebook:
https://github.com/vinampud/gsb545/blob/main/kaggle_hws/hw2_EDA_and_models.ipynb

(^ didn't realize models had to go in their own notebooks, hope this is okay)


## Discussion

### What I tried

Two modeling approaches were explored: **Random Forest** (bagging) and **XGBoost** (boosting), using 5-fold stratified cross-validation, with balanced accuracy as the evaluation metric to match the Kaggle competition.

Feature engineering was applied to both models, including binary threshold flags (`soil_lt_25`, `rain_lt_300`, `temp_gt_30`, `wind_gt_10`), a composite `magic_score`, ET proxy, heat stress, dryness index, and soil health index, all motivated by the EDA findings.

### Performance Summary

| Model | CV OOF Balanced Accuracy | CV Std | Kaggle LB Score |
|---|---|---|---|
| Random Forest (bagging) | 0.96453 | 0.00061 | 0.96192 |
| XGBoost (boosting) | 0.97099 | 0.00088 | 0.96874 |

### What worked well

- **XGBoost > Random Forest** on both CV and the Kaggle leaderboard. The gap was ~0.006 in CV and ~0.007 on the leaderboard --> meaningful but not super dramatic.
- **Feature engineering** The domain threshold flags and `magic_score` encode near-deterministic rules about the irrigation need target, giving both models a strong signal to work with.
- **Class weighting** (`class_weight='balanced'` for RF, `compute_sample_weight('balanced')` for XGBoost) was essential given the severe imbalance in the High class.

### What didn't work well / limitations

- **Random Forest was significantly slower than XGBoost** on the full 630,000-row dataset — RF took ~55 minutes while XGBoost completed in ~20 minutes, despite XGBoost using more estimators. This is because XGBoost's histogram-based splitting (`tree_method='hist'`) is much more efficient than RF's exact splitting on large datasets.
- **Random Forest was also less accurate** than XGBoost. The sequential error-correction of boosting is better suited to learning the sharp threshold-based decision boundaries in this dataset.
- **No hyperparameter tuning** was done beyond the initial configuration. More tuning (especially `max_depth`, `learning_rate`, and `n_estimators` for XGBoost) would likely improve the LB score.
- **The reference notebooks achieve 0.979–0.981** by also blending in the original source dataset and using multi-model ensembles — both of which I didn't do here.

### Conclusion

XGBoost is the stronger model here, achieving a Kaggle LB score of **0.96874** vs Random Forest's **0.96192**. The improvement from bagging to boosting of ~0.007 is still meaningful in the context of this competition. XGBoost was also nearly 3x faster on the full dataset, making it the clearly better choice on both accuracy and efficiency. Further gains would likely come from blending in the original source dataset, adding LightGBM and CatBoost to an ensemble, and more aggressive hyperparameter tuning.

## Phase 2 Plan

The biggest gains would come from blending in the original source dataset to nearly double the training data, adding LightGBM and CatBoost to create an ensemble, and running Optuna to tune XGBoost hyperparameters. I'd also add the logit score features from the reference notebook and apply threshold optimization on the predicted probabilities to better handle the rare High class.

# HOMEWORK 3
## gradient boosting notebook
https://github.com/vinampud/gsb545/blob/main/kaggle_hws/hw3_gradient_boosting.ipynb

## Discussion

### What I tried

For this assignment, I implemented **XGBoost** and **LightGBM**. The three iterations of hyperparamter tuning allowed me to explore how parameters like `learning_rate`, `max_depth`, `n_estimators`, `subsample`, and even the terms unique to the model (eg. `gamma`, `reg_alpha`, `reg_lambda` for XGBoost and `num_leaves`, `min_child_samples` for LightGBM) affect performance.

### Performance Summary

| Model | Hyperparameter Set | CV Mean Accuracy | CV Std Accuracy | Kaggle LB Score |
|-------|---------------------|------------------|-----------------|-----------------|
| XGBoost | Config 1 (n_estimators=200, max_depth=5, learning_rate=0.10) | 0.97031 | 0.00067 | 0.96745 |
| XGBoost | Config 2 (n_estimators=300, max_depth=7, learning_rate=0.05) | 0.96888 | 0.00077 | - |
| XGBoost | Config 3 (n_estimators=150, max_depth=3, learning_rate=0.20) | 0.96883 | 0.00106 | - |
| LightGBM | Config 1 (n_estimators=300, learning_rate=0.05, num_leaves=31) | 0.96988 | 0.00086 | 0.9669 |
| LightGBM | Config 2 (n_estimators=250, learning_rate=0.10, num_leaves=50) | 0.96776 | 0.00104 | - |
| LightGBM | Config 3 (n_estimators=400, learning_rate=0.03, num_leaves=16) | 0.96852 | 0.00107 | - |

### What worked well

- **Hyperparameter exploration showed meaningful differences**: Though it took a while, the three configurations for each model produced varying CV accuracies, demonstrating how tuning parameters impacts results. For example, XGBoost's first run (higher learning rate, moderate depth) achieved the highest CV accuracy (0.97031), while the third run (shallow trees, high learning rate) was slightly lower (0.96883).
- **Both models performed strongly on Kaggle**: XGBoost's best submission scored 0.96745, and LightGBM's scored 0.9669, both significantly competitive. The CV accuracies were good predictors after switching to accuracy scoring.
- **Feature engineering carried over effectively**: Using the threshold flags and composite features from Homework 2 provided a solid foundation, allowing the boosting models to learn effectively without overfitting.

### What didn't work well / limitations

- **Initial metric mismatch**: I started with balanced accuracy overestimated performance (should have double checked AI's recommendations) compared to Kaggle's plain accuracy, leading to a gap between CV scores and my results for my Kaggle submissions. I then switched back to `accuracy_score` which seemed to "fix" the score.
- **Some hyperparameter sets underperformed**: XGBoost's second config (deeper trees, lower learning rate) and LightGBM's second config (more leaves, higher learning rate) had lower CV accuracies, showing that not all parameter combinations improve results. This underscores the need for careful tuning.
- **Small improvements between models**: The difference between XGBoost (0.96745) and LightGBM (0.9669) on the leaderboard was only about 0.00055, indicating that while XGBoost edged out slightly, the gains from switching algorithms were modest rather than dramatic.

### Model Comparison

XGBoost and LightGBM are both gradient boosting methods but differ in implementation: 
- XGBoost uses histogram-based approximations for speed
- LightGBM employs gradient-based one-side sampling and exclusive feature bundling for efficiency on large datasets.

In this assignment, XGBoost slightly outperformed LightGBM, suggesting XGBoost was better suited to the dataset's features and thresholds. Though the difference was small, this could mean a whole lot in a professional work / real-world setting. However, we can still say that LightGBM's performance was excellent, making it a strong alternative. Overall, the boosting approach was effective, with meaningful improvements over the bagging Random Forest from Homework 2, but further improvements would require ensemble methods or additional data as seen in top Kaggle notebooks.

# HOMEWORK 4
## notebook
https://github.com/vinampud/gsb545/blob/main/kaggle_hws/hw4_feature_engineering.ipynb

## Feature Engineering, Model Diversity, and Ensembling
### What I tried

For this assignment, I expanded the workflows from previous assignments by introducing **additional model diversity, new feature engineering, and ensembling**.

In addition to the boosting models from Homework 3 (**XGBoost and LightGBM**), I added a **Logistic Regression** model to provide a fundamentally different approach (linear). This allowed me to see that my models differed meaningfully in terms of assumptions and representation.

I also extended my feature engineering by adding interaction-based features such as:
- `temp_x_humidity`
- `moisture_per_rain`

These were designed to capture relationships between environmental conditions rather than relying only on the given individual variables or threshold flags.

### Feature evaluation

To evaluate feature usefulness, I examined the built in **feature importance from the XGBoost model**.

The most important features were:
- The engineered threshold flags (eg. `soil_lt_25`, `rain_lt_300`)
- The composite `magic_score`
- Key environmental variables like temperature and rainfall

This confirms that the feature engineering from previous assignments remains highly effective. The threshold-based features in particular appear to build strong signals related to irrigation need.

The newer interaction features contributed less than the main engineered features, which tells us that while they added some information, the majority of predictive power still comes from the domain-informed threshold features.

### Model performance

| Model | Validation Accuracy | Kaggle Public LB Score |
|------|-------------------|------------------------|
| XGBoost | 0.985881 | 0.95923 |
| LightGBM | 0.985825 | 0.95987 |
| Logistic Regression | 0.867659 | 0.70948 |
| Ensemble (XGB + LGB) | 0.985889 | **0.95949** |

The ensemble combined XGBoost and LightGBM using **probability averaging**, which helped reduce variance and leverage the strengths of both models.

### What worked well

- **Boosting models remained the strongest performers**, with both XGBoost and LightGBM getting very similar Kaggle scores.
- **LightGBM slightly outperformed XGBoost on the leaderboard**, even though their validation accuracies were almost identical.
- **Feature engineering continued to drive performance**, especially the threshold-based and composite features.
- **Ensembling provided a small improvement**, increasing performance slightly over individual models.
- Adding Logistic Regression increased model diversity, which is useful for ensembling, even though its standalone performance was much lower.

### What didn’t work as well

- **Improvements from ensembling were very small**, showing us that XGBoost and LightGBM are already capturing similar patterns in the data.
- The **interaction features did not significantly outperform existing engineered features**.
- **Logistic Regression performed much worse (~0.71 LB score)**, showing that linear models (as expected) struggle to capture the nonlinear relationships in this dataset.

### Interpretation of results

There is a noticeable difference between validation accuracy and Kaggle leaderboard scores, which highlights the importance of generalization.

While the models perform extremely well on given validation data, performance on unseen data is lower. This suggests that:

- These models may be slightly overfitting to patterns in the training data
- Validation accuracy may be optimistic due to the simplicity of the train-test split
- Additional improvements would likely require more robust validation (like cross-validation) or more diverse models

Additionally, the very small differences between XGBoost, LightGBM, and the ensemble indicate that the models are converging toward similar performance limits given the current features.

### What I will continue to use

- XGBoost as the primary model
- Domain-informed feature engineering (thresholds + composite features)
- Ensembling as a final step to improve performance
