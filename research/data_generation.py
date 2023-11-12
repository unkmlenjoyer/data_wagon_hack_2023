# %% [markdown]
"""Скрипт генерации данных"""

# %%
from functools import reduce

import numpy as np
import pandas as pd

# %%
pd.set_option("display.max_columns", 300)
pd.set_option("display.max_rows", 500)

# %%
# Константы
# В данном скрипте константа рандомизации не нужна
PATH_TRAIN_FISRT = r"../data/raw/train_1"
PATH_TRAIN_SECOND = r"../data/raw/train_2"

# %% [markdown]
# ## Загрузка данных

# %%
# Cписок вагонов с остаточным пробегом на момент прогноза
wag_prob = pd.concat(
    map(
        lambda x: pd.read_parquet(x + "/wagons_probeg_ownersip.parquet"),
        [PATH_TRAIN_FISRT, PATH_TRAIN_SECOND],
    )
).reset_index(drop=True)

# Параметры вагонов
wag_param = pd.concat(
    map(
        lambda x: pd.read_parquet(x + "/wag_params.parquet"),
        [PATH_TRAIN_FISRT, PATH_TRAIN_SECOND],
    )
).reset_index(drop=True)

# Данные по плановым ремонтам
pr_rem = pd.concat(
    map(
        lambda x: pd.read_parquet(x + "/pr_rems.parquet"),
        [PATH_TRAIN_FISRT, PATH_TRAIN_SECOND],
    )
).reset_index(drop=True)

# Текущие ремонты вагонов
tr_rem = pd.concat(
    map(
        lambda x: pd.read_parquet(x + "/tr_rems.parquet"),
        [PATH_TRAIN_FISRT, PATH_TRAIN_SECOND],
    )
).reset_index(drop=True)

# Данные по дислокации
dislok = pd.concat(
    map(
        lambda x: pd.read_parquet(x + "/dislok_wagons.parquet"),
        [PATH_TRAIN_FISRT, PATH_TRAIN_SECOND],
    )
).reset_index(drop=True)

# Данные грузов
freight = pd.concat(
    map(
        lambda x: pd.read_parquet(x + "/freight_info.parquet"),
        [PATH_TRAIN_FISRT, PATH_TRAIN_SECOND],
    )
).reset_index(drop=True)

# Данные колеии
kti_data = pd.concat(
    map(
        lambda x: pd.read_parquet(x + "/kti_izm.parquet"),
        [PATH_TRAIN_FISRT, PATH_TRAIN_SECOND],
    )
).reset_index(drop=True)

# Таргет по прогноза выбытия вагонов в плановый ремонт на месяц и на 10 дней
target = pd.concat(
    map(
        lambda x: pd.read_csv(x + "/target/y_train.csv"),
        [PATH_TRAIN_FISRT, PATH_TRAIN_SECOND],
    )
).reset_index(drop=True)

# Данные для предикта
y_to_predict = pd.read_csv("../data/raw/submission" + "/y_test.csv")

# %% [markdown]
# ## Подготовка данных
# 7203124 - целевое число записей
# %%
# ------------- ОТЧЕТНЫЕ ДАННЫЕ ------------------
# Генерация всех дат отчетов и вагонов
rep_dates = pd.DataFrame(data={"repdate": pd.date_range("2022-08-01", "2023-02-28")})
wagons_full = pd.DataFrame(data={"wagnum": wag_prob.wagnum.unique()})

# Конкатенация путем кросс-джойна для полного отчета по дням
df_core = pd.merge(rep_dates, wagons_full, how="cross")
df_core["month"] = df_core["repdate"].dt.month.astype(np.int64)

df_core["month_observ"] = df_core["repdate"].dt.strftime("%Y-%m-01")

# Соединение с данными реального отчета для полного отчета по дням
df_core = pd.merge(df_core, wag_prob, how="left", on=["repdate", "wagnum", "month"])
df_core = df_core.sort_values(by=["wagnum", "repdate"]).reset_index(drop=True)

# %%
# ------------- ПЛАНОВЫЙ РЕМОНТ ------------------
# В плановом ремонте есть ремонты вагонов по дням - дубликаты, там есть разница
#  только в дистанции, поэтому оставляем для каждой даты ремонта вагона последнюю запись
pr_rem = pr_rem.drop_duplicates(subset=["wagnum", "rem_month"], keep="last")
pr_rem["repdate"] = pr_rem["rem_month"]

# Станция отгрузки == стнация предприятия ремонта вагонов ??
pr_rem["st_send_prem_eq"] = pr_rem["st_id_send"] == pr_rem["st_id_rem"]

pr_rem = pr_rem.drop(columns=["distance"])

# %%
# ------------- ПАРАМЕТРЫ ВАГОНА ------------------
# У вагонов могут меняться параметры, поэтому номер дублируется.
wag_param = wag_param.drop_duplicates(subset="wagnum", keep="last")

# %%
# ------------- ТЕКУЩИЙ РЕМОНТ ---------------------
# Посчитаем текущие ремонты за период по вагонам
# Присоединять в конце
tr_rem = tr_rem.drop_duplicates()
tr_rem = tr_rem.groupby(["rem_month", "wagnum"], as_index=False).agg(
    curr_repair_count=("wagnum", "count")
)
tr_rem["month_observ"] = tr_rem["rem_month"].astype(str)
tr_rem = tr_rem.drop("rem_month", axis=1)

# %%
# ------------- Дислокация на дату ---------------------
dislok_freight = pd.merge(dislok, freight, on="fr_id", how="left")

# Есть дубликат - убираем
dislok_freight = dislok_freight.drop_duplicates()

# %%
dislok_freight_clear = (
    dislok_freight[
        [
            "plan_date",
            "wagnum",
            "isload",
            "fr_id",
            "last_fr_id",
            "fr_class",
            "skoroport",
            "naval",
            "nasip",
        ]
    ]
    .rename(columns={"plan_date": "repdate"})
    .drop_duplicates()
)


# %%
#
df_core = pd.merge(df_core, pr_rem, on=["wagnum", "repdate", "month"], how="left")
df_core = pd.merge(df_core, wag_param, on="wagnum", how="left")
df_core = pd.merge(df_core, dislok_freight_clear, on=["wagnum", "repdate"], how="left")

# %%
# Заполняем остаточный пробег либо back fill, либо forward fill
df_core["ost_prob"] = df_core.groupby(["wagnum"], as_index=False).ost_prob.transform(
    "bfill"
)
df_core["ost_prob"] = df_core.groupby(["wagnum"], as_index=False).ost_prob.transform(
    "ffill"
)

# %%
# Заполняем остаточный пробег огромным числом, если вагон на пробеге
# либо берем саму норму межремонтную, либо пробег оставляем.
df_core["ost_prob"] = df_core.apply(
    lambda x: 1e6
    if (pd.isna(x["ost_prob"]) and x["norma_km"] == 0)
    else (x["norma_km"] if pd.isna(x["ost_prob"]) else x["ost_prob"]),
    axis=1,
)

# %%
# Разница в пробеге - дистанция (не всегда корректно но хоть что-то)
df_core["distance"] = (
    df_core.groupby(["wagnum"]).ost_prob.transform("diff").fillna(0)
).apply(lambda x: abs(x) if x < 0 else 0)

# %%
# Средняя скорость за сутки в км/ч
df_core["avg_speed"] = df_core["distance"].apply(lambda x: x / 24)

# %%
# Остаток пробега на конец месяца
df_core["month_ost_prob"] = df_core.groupby(
    ["wagnum", "month_observ"], as_index=False
).ost_prob.transform("last")

# %%
# Генерация агрегатов - статистик для вагонов на каждый месяц

# Скорость и расстояние (примерное)
distance_speed = (
    df_core[df_core.avg_speed > 0]
    .groupby(["wagnum", "month_observ"], as_index=False)
    .agg(
        days_speed_non_zero=("repdate", "count"),
        speed_day_max=("avg_speed", "max"),
        speed_day_min=("avg_speed", "min"),
        speed_day_avg=("avg_speed", "mean"),
        speed_day_std=("avg_speed", "std"),
        distance_sum=("distance", "sum"),
        distance_day_min=("distance", "min"),
        distance_day_max=("distance", "max"),
        distance_day_avg=("distance", "mean"),
        distance_day_std=("distance", "std"),
    )
)

# %%
# Сколько дней скорость == 0
zero_state = (
    df_core[df_core.avg_speed == 0]
    .groupby(["wagnum", "month_observ"], as_index=False)
    .agg(days_zero_speed=("repdate", "count"))
)

# %%
# Сколько ремонтов было за месяц у вагона
plan_repair_count = df_core.groupby(["wagnum", "month_observ"], as_index=False).agg(
    plan_repair_count_month=("rem_month", "count")
)

# %%
# Сколько дней вагон был в нагрузке в движении и сколько уникальных ID грузов перевез
payload = (
    df_core[df_core.avg_speed > 0]
    .groupby(["wagnum", "month_observ"], as_index=False)
    .agg(
        days_payload=("isload", "sum"),
        uniq_payload=("fr_id", "nunique"),
    )
)

# %%
# Остаток на конец месяца по пробегу у конкретного вагона
month_ost_prob = df_core.groupby(["wagnum", "month_observ"], as_index=False).agg(
    max_ost_prob=("month_ost_prob", "max")
)

# %%
# Генерация данных для финального датасета
rep_dates = pd.DataFrame(
    data={
        "month_observ": pd.date_range("2022-08-01", "2023-02-28").strftime("%Y-%m-01")
    }
).drop_duplicates()

final = (
    pd.merge(rep_dates, wagons_full, how="cross")
    .sort_values(by=["wagnum", "month_observ"])
    .reset_index(drop=True)
)

# %%
# Конкатенация всех статистик
to_concat = [
    final,
    zero_state,
    distance_speed,
    month_ost_prob,
    plan_repair_count,
    payload,
    tr_rem,
]

result = reduce(
    lambda x, y: pd.merge(x, y, how="left", on=["wagnum", "month_observ"]), to_concat
).fillna(0)

# %%
# Данные вагона
wag_param_sample = wag_param[
    [
        "wagnum",
        "rod_id",
        "gruz",
        "cnsi_volumek",
        "tara",
        "cnsi_probeg_dr",
        "cnsi_probeg_kr",
        "ownertype",
        "norma_km",
        "date_build",
        "srok_sl",
    ]
]

# На пробеге или нет
wag_param_sample["is_probeg"] = (wag_param_sample["norma_km"] == 0).astype(int)
# %%
result = pd.merge(result, wag_param_sample, on="wagnum", how="left")

# %%
# отношение остатка к норме
result["ost_percent"] = result.apply(
    lambda x: 1 if x["is_probeg"] == 1 else x["max_ost_prob"] / x["norma_km"], axis=1
)

# %%
# Сколько еще жить вагону до истечения срока службы
result["last_age"] = (
    pd.to_datetime(result["srok_sl"]) - pd.to_datetime(result["month_observ"])
).dt.days / 365

# %%
# Возраст вагона
result["age"] = (
    pd.to_datetime(result["month_observ"]) - pd.to_datetime(result["date_build"])
).dt.days / 365

# %%
# Работает ли вагон выше своего срока службы
result["over_age"] = (result["last_age"] < 0).astype(int)

# %%
result = result.drop(columns=["srok_sl", "date_build", "norma_km"])
result = result.rename(columns={"month_observ": "month"})

# %%
# Сдвиг таргета для предсказания
result["month_to_predict"] = (
    pd.to_datetime(result["month"]) + pd.DateOffset(months=1)
).astype(str)

# %%
# Данные для предикта
data_predict = pd.merge(
    y_to_predict,
    result[result["month_to_predict"] == "2023-03-01"],
    how="left",
    left_on=["wagnum", "month"],
    right_on=["wagnum", "month_to_predict"],
)

# %%
# Данные для обучения
data_train = pd.merge(
    target,
    result,
    how="left",
    left_on=["wagnum", "month"],
    right_on=["wagnum", "month_to_predict"],
)

# %%
data_predict.to_feather("../data/processed/data_predict.feather")
data_train[~data_train.month_y.isna()].reset_index(drop=True).to_feather(
    "../data/processed/data_train.feather"
)
