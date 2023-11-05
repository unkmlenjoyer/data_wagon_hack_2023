"""Configuration for research scripts"""

from abc import ABC
from dataclasses import dataclass


@dataclass
class BaseResearchConfig(ABC):
    pass


@dataclass
class EdaConfig(BaseResearchConfig):
    pass


@dataclass
class ModelingConfig(BaseResearchConfig):
    pass
