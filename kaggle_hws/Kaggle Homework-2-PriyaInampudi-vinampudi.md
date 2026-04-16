## 3 notebooks I found insightful/helpful:
1. https://www.kaggle.com/code/aliafzal9323/s6e4-0-970-stacked-lgb-xgb-cat-feature-engine
2. https://www.kaggle.com/code/mohit78241/s6e4-ensemble-voting-transfer-0-981-lb
3. https://www.kaggle.com/code/manasi197/s6e4-xgboost

s6e4-xgboost: Introduced the domain threshold flags and magic_score composite feature that I directly used. 

s6e4-stacked-lgb-xgb-cat: Showed the value of blending the original source dataset with competition data and using a LGB+XGB+CatBoost stack with a meta-learner. I'm currently using only XGBoost, so adding more base models is a major gain still available to me.

s6e4-ensemble-voting-transfer: Highest scorer, showed that combining multiple strong submissions via conditional voting (consensus when models agree, stronger model as fallback when they don't) outperforms any single model. Once I have a few solid models, ensembling predictions is worth doing over picking one winner.

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