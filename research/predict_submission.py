"""Скрипт формирования сабмита для скора на платформе"""
# %%
# Импорт библиотек
import catboost
import pandas as pd
from src.config import ModelingConfig
from src.utils import optimize_df_memory

# %%
# Импорт и определение констант
conf = ModelingConfig()

# Пороги моделей
MONTH_THRESHOLD = 0.701702
DAY_THRESHOLD = 0.655656

# %%
# Подгрузка моделей
month_clf, day_clf = (
    catboost.CatBoostClassifier().load_model("../models/" + "month_predict_clf_v1"),
    catboost.CatBoostClassifier().load_model("../models/" + "10_day_predict_clf_v1"),
)

# %%
# Данные для предсказания
data = optimize_df_memory(
    pd.read_feather(conf.path_processed_folder + "data_predict.feather")
)

# %%
# Для каких вагонов надо предсказать
subm_data = pd.read_csv("../data/raw/submission/y_test.csv")

# %%
merged_sub = pd.merge(
    subm_data,
    data,
    how="left",
    left_on=["wagnum", "month"],
    right_on=["wagnum", "month_x"],
)

# %%
# Удаление ненужных признако (чисто технических)
drop_features = [
    "wagnum",
    "month",
    "month_x",
    "month_y",
    "month_to_predict",
]

X_predict = merged_sub.drop(columns=drop_features)

# %%
merged_sub["target_month"] = (
    month_clf.predict_proba(X_predict)[:, 1] > MONTH_THRESHOLD
).astype(int)
merged_sub["target_day"] = (
    day_clf.predict_proba(X_predict)[:, 1] > DAY_THRESHOLD
).astype(int)

# %%
# Генерация посылки
merged_sub[["wagnum", "month", "target_month", "target_day"]].to_csv(
    "../data/final/submission_v4_1000.csv", index=False
)
