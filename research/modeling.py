"""Script to modelling on processed data"""

# %%
import catboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from src.config import ModelingConfig
from src.utils import optimize_df_memory

# %%
conf = ModelingConfig()

# %%
data = optimize_df_memory(
    pd.read_feather(conf.data_folder + "/processed/features_v1.feather")
)

# %%
X, y = data.drop(columns=["score", "id"]), data.score

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=conf.RANDOM_SEED
)

# %%
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.2, random_state=conf.RANDOM_SEED
)


# %%
def fit_catboost(trial, train, val):
    params = {
        "depth": trial.suggest_int("depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.8),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 50),
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0, 10
        )
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    X_train_part, y_train_part = train
    X_val_part, y_val_part = val

    model = catboost.CatBoostRegressor(
        **params,
        iterations=600,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=conf.RANDOM_SEED,
    )

    pruning_callback = optuna.integration.CatBoostPruningCallback(trial, "RMSE")

    model.fit(
        X_train_part,
        y_train_part,
        eval_set=(X_val_part, y_val_part),
        verbose=0,
        early_stopping_rounds=10,
        use_best_model=True,
        callbacks=[pruning_callback],
    )

    pruning_callback.check_pruned()

    y_pred = model.predict(X_val_part)

    return y_pred


# %%
def objective(trial):
    splitter = KFold(5, shuffle=True, random_state=conf.RANDOM_SEED)

    scores = []
    for train_idx, val_idx in splitter.split(X_train, y_train):
        train_x, val_x = X_train.iloc[train_idx, :], X_train.iloc[val_idx, :]
        train_y, val_y = y_train.iloc[train_idx], y_train.iloc[val_idx]

        y_pred_val = fit_catboost(trial, (train_x, train_y), (val_x, val_y))

        score = mean_squared_error(val_y, y_pred_val, squared=False)
        scores.append(score)

    mean_cv_score = np.mean(scores)
    std_cv_score = np.std(scores)

    return mean_cv_score - std_cv_score


# %%
study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%
model = catboost.CatBoostRegressor(
    **trial.params,
    iterations=600,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=conf.RANDOM_SEED,
)

# %%
model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=10,
    use_best_model=True,
    plot=True,
)
