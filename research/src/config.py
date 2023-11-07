"""Configuration for research scripts"""

from abc import ABC
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BaseResearchConfig(ABC):
    """Base configuration for all researches

    Attributes
    ----------
    pd_max_cols : int
        Pandas max number of columns to show

    pd_max_rows : int
        Pandas max number of rows to show

    sns_style : str
        Seaborn style

    sns_figsize : Tuple[int, int]
        Seaborn size of graphs

    random_seed : int
        Random constant. Fixed to reproduce.
    """

    pd_max_cols: int = 100
    pd_max_rows: int = 200
    sns_style: str = "darkgrid"
    sns_fig_size: Tuple[int, int] = (30, 20)
    random_seed: int = 4242


@dataclass
class EdaConfig(BaseResearchConfig):
    """Configuration for EDA

    Attributes
    ----------
    path_raw_folder : str
        Path to folder with raw data
    """

    path_raw_folder: str = "../data/raw/"


@dataclass
class ModelingConfig(BaseResearchConfig):
    """Configuration for modeling

    Attributes
    ----------
    path_processed_folder : str
        Path to folder with processed data

    test_size : float
        Percentage of sample data for the test.
        Must be in [0, 1]. By default 0.2
    """

    path_processed_folder: str = "../data/processed/"
    test_size: float = 0.2
