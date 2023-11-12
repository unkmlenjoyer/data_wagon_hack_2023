"""Script to modelling on processed data"""

# %%
import warnings

import catboost
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.config import ModelingConfig
from src.utils import optimize_df_memory

warnings.filterwarnings("ignore")


# %%
conf = ModelingConfig()

USE_OPTUNA = False

# %%
pd.set_option("display.max_columns", conf.pd_max_cols)
pd.set_option("display.max_rows", conf.pd_max_rows)

# %%
sns.set_style("darkgrid")
sns.set({"figure.figsize": conf.sns_fig_size})

# %%
TASK_TYPE = "10_day_predict"

# %%
data = optimize_df_memory(
    pd.read_feather(conf.path_processed_folder + "data_train.feather")
)

# %%
drop_features = [
    "wagnum",
    "month_x",
    "month_y",
    "month_to_predict",
    "target_day",
    "target_month",
]

target_feature = {
    "month_predict": "target_month",
    "10_day_predict": "target_day",
}

# %%
X, y = (
    data.drop(columns=drop_features),
    data[target_feature[TASK_TYPE]],
)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=conf.random_seed, stratify=y
)

# %%
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.2, random_state=conf.random_seed, stratify=y_test
)


# %%
def objective(trial, return_models=False):
    trial_id = trial.number

    if TASK_TYPE == "month_predict":
        param_space = {
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 0.8),
            "depth": trial.suggest_int("depth", 6, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
            "bootstrap_type": "Bayesian",
            "bagging_temperature": trial.suggest_float("bagging_temperature", 1, 6),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 50),
            "auto_class_weights": "SqrtBalanced",
        }
    else:
        param_space = {
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
            "depth": trial.suggest_int("depth", 6, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
            "bootstrap_type": "MVS",
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 10, 25),
            "auto_class_weights": "SqrtBalanced",
        }

    splitter = StratifiedKFold(5, shuffle=True, random_state=conf.random_seed)

    scores_valid, scores_train, models = [], [], []

    for i, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
        train_x, val_x = X_train.iloc[train_idx, :], X_train.iloc[val_idx, :]
        train_y, val_y = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_pool = catboost.Pool(train_x, train_y)
        valid_pool = catboost.Pool(val_x, val_y)

        clf = catboost.CatBoostClassifier(
            **param_space,
            iterations=500,
            eval_metric="PRAUC:use_weights=False",
            random_seed=conf.random_seed,
            early_stopping_rounds=5,
        )

        pruning_callback = optuna.integration.CatBoostPruningCallback(
            trial, "PRAUC:use_weights=false"
        )

        clf.fit(
            train_pool,
            eval_set=valid_pool,
            verbose=0,
            use_best_model=True,
            callbacks=[pruning_callback] if i == 0 else None,
        )

        if i == 0:
            pruning_callback.check_pruned()

        y_train_pred = clf.predict_proba(train_pool)[:, 1]
        y_valid_pred = clf.predict_proba(valid_pool)[:, 1]

        train_score = average_precision_score(train_y, y_train_pred)
        valid_score = average_precision_score(val_y, y_valid_pred)

        print(
            f"Trial # {trial_id}: fold # {i}, score train: {train_score}, score valid: {valid_score}"
        )

        scores_valid.append(valid_score)
        scores_train.append(train_score)
        models.append(clf)

    mean_cv_score_valid, std_cv_score_valid = np.mean(scores_valid), np.std(
        scores_valid
    )

    mean_cv_score_train, std_cv_score_train = np.mean(scores_train), np.std(
        scores_train
    )

    final_score = mean_cv_score_valid - std_cv_score_valid

    print(
        f"Trial # {trial_id}: \
        score train: {mean_cv_score_train} +/- {std_cv_score_train} \
        score valid: {mean_cv_score_valid} +/- {std_cv_score_valid}"
    )

    return (final_score, models) if return_models else final_score


if USE_OPTUNA:
    study = optuna.create_study(
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=500),
        direction="maximize",
    )
    study.optimize(objective, n_trials=500, n_jobs=-1, show_progress_bar=True)

else:
    sel_clf = catboost.CatBoostClassifier(random_seed=conf.random_seed).load_model(
        "../models/" + TASK_TYPE + "_clf_v1"
    )

# %%
if USE_OPTUNA:
    scores, models = objective(
        optuna.trial.FixedTrial(study.best_params), return_models=True
    )

    sel_clf = models[0]
    sel_clf.save_model("../models/" + TASK_TYPE + "_clf_v2")


# %%
# ----------------------------------
# -------------v1-------------------

# MONTH CLF
# Trial # 0: fold # 0, score train: 0.6786055593129637, score valid: 0.6611890236264374
# Trial # 0: fold # 1, score train: 0.6649752539096638, score valid: 0.6560282935326784
# Trial # 0: fold # 2, score train: 0.6678261552467547, score valid: 0.6524689050501199
# Trial # 0: fold # 3, score train: 0.6695647904479617, score valid: 0.6540674805360456
# Trial # 0: fold # 4, score train: 0.6715767810891389, score valid: 0.6515391718369485
# score train: 0.6705097080012965 +/- 0.004591425924412159
# score valid: 0.655058574916446 +/- 0.0034233787901398766

# 0.653496438840056

# threshold	precision	recall	    f1_score
# 0.69697	0.624046	0.609695	0.616787

# 	0.701702	0.628939	0.607831	0.618205 - 1000

#               precision   recall    f1-score    support

#            0       0.98      0.98      0.98      7742
#            1       0.59      0.58      0.59       402

#     accuracy                           0.96      8144
#    macro avg       0.78      0.78      0.78      8144
# weighted avg       0.96      0.96      0.96      8144

# 0.6573261188596414

# -------------------

# DAY CLF
# Trial # 0: fold # 0, score train: 0.44943027131861524, score valid: 0.4680614420527784
# Trial # 0: fold # 1, score train: 0.47129864734847443, score valid: 0.40628828202207007
# Trial # 0: fold # 2, score train: 0.4191263856212902, score valid: 0.43898500042159744
# Trial # 0: fold # 3, score train: 0.4207350068035721, score valid: 0.42466838044398514
# Trial # 0: fold # 4, score train: 0.46642999928377316, score valid: 0.4246820795342589
# score train: 0.44540406207514505 +/- 0.02203591845445296
# score valid: 0.432537036894938 +/- 0.020572665541035966

# 0.45949210652112715

# threshold	precision   recall	f1_score
# 0.656566	0.444613	0.55283	0.492851

# 	threshold	precision	recall	f1_score
# 655	0.655656	0.444109	0.554717	0.493289 - 1000

# 0.42159736190809927

# precision    recall  f1-score   support

#            0       0.99      0.99      0.99      8011
#            1       0.44      0.50      0.47       133

#     accuracy                           0.98      8144
#    macro avg       0.72      0.75      0.73      8144
# weighted avg       0.98      0.98      0.98      8144

# ----------------------------------
# -------------v2-------------------

# %%
# Trial # 0: fold # 0, score train: 0.66752801611983, score valid: 0.667384234623492
# Trial # 0: fold # 1, score train: 0.6669530166198945, score valid: 0.6609049695757615
# Trial # 0: fold # 2, score train: 0.6676902947402807, score valid: 0.6552319994099485
# Trial # 0: fold # 3, score train: 0.6679885555575408, score valid: 0.6566206338540155
# Trial # 0: fold # 4, score train: 0.6670124276852608, score valid: 0.6570360769075257
# score train: 0.6674344621445614 +/- 0.0003977750750580386
# score valid: 0.6594355828741486 +/- 0.004397541193019565

# 0.6608842132890751

# threshold	precision	recall	    f1_score
# 0.646465	0.582778	0.651958	0.61543

# 0.6645195962615575

#                    precision recall   f1-score   support

#            0       0.98      0.97      0.98      7742
#            1       0.56      0.66      0.61       402

#     accuracy                           0.96      8144
#    macro avg       0.77      0.82      0.79      8144
# weighted avg       0.96      0.96      0.96      8144

# %%
# Trial # 0: fold # 0, score train: 0.45069270063125605, score valid: 0.47183660348756573
# Trial # 0: fold # 1, score train: 0.4832869790839452, score valid: 0.4162309596763879
# Trial # 0: fold # 2, score train: 0.4585004920933774, score valid: 0.47043223392371364
# Trial # 0: fold # 3, score train: 0.43139471800531387, score valid: 0.44295008718107914
# Trial # 0: fold # 4, score train: 0.4550734718039219, score valid: 0.42303412627622305
# score train: 0.45578967232356293 +/- 0.0166436551436573
# score valid: 0.44489680210899396 +/- 0.023157322782357572

# 0.46801270241287196

# threshold	precision	recall	f1_score
# 0.717172	0.511022	0.481132	0.495627


# %%
scores_threshold = []
y_pred_val_score = sel_clf.predict_proba(X_val)[:, 1]

# %%
pr_auc_val = average_precision_score(y_val, y_pred_val_score)

# %%
for t in np.linspace(0, 1, 1000):
    y_pred_val = np.int8(y_pred_val_score > t)

    pr_score = precision_score(y_val, y_pred_val)
    rec_score = recall_score(y_val, y_pred_val)
    f_score = f1_score(y_val, y_pred_val)
    scores_threshold.append((t, pr_score, rec_score, f_score))

# %%
data_scores = pd.DataFrame(
    data=scores_threshold, columns=["threshold", "precision", "recall", "f1_score"]
)
# %%
sns.scatterplot(data_scores, x="threshold", y="precision", label="precision")
sns.scatterplot(data_scores, x="threshold", y="recall", label="recall")
sns.scatterplot(data_scores, x="threshold", y="f1_score", label="f1_score")
# %%
data_scores[data_scores.f1_score == data_scores.f1_score.max()]

# %%
y_pred_test_score = sel_clf.predict_proba(X_test)[:, 1]
pr_auc_test = average_precision_score(y_test, y_pred_test_score)

# %%
print(classification_report(y_test, y_pred_test_score > 0.655656))

# %%
pr_auc_test

# %%
explainer = shap.TreeExplainer(sel_clf)

# %%
val_dataset = catboost.Pool(data=X_test, label=y_test)
shap_values = explainer.shap_values(val_dataset)
shap.summary_plot(shap_values, X_test, max_display=15, plot_size=(20, 15))
