"""Useful utils for research"""

from typing import List

import numpy as np
import pandas as pd


def optimize_df_memory(data: pd.DataFrame, cat_cols: List[str] = []) -> pd.DataFrame:
    """Function to reduce memory usage.

    Iterates over columns in order to compare max/min values and datatypes.
    If max/min values are in range of low-memort datatype, that column converts to
    low-memory datatype.

    In order to process categorical features, just pass them to cat_cols.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to reduce
    cat_cols : List[str], optional
        Categorical features, by default []

    Returns
    -------
    pd.DataFrame
        Low-memory dataframe
    """

    start_mem = data.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in data.columns:
        if cat_cols and col in cat_cols:
            data[col] = data[col].astype("category")
        else:
            col_type = data[col].dtype.name
            if col_type not in ["object", "category", "datetime64[ns, UTC]"]:
                c_min = data[col].min()
                c_max = data[col].max()
                if col_type.startswith("int"):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        data[col] = data[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        data[col] = data[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        data[col] = data[col].astype(np.int64)
                elif col_type.startswith("float"):
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        data[col] = data[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        data[col] = data[col].astype(np.float32)
                    else:
                        data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return data
