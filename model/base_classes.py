"""
abstract base model for classifier and segmentation inference
"""

from abc import ABC, abstractmethod
from utils.config import Config

class BaseClassifier(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

class BaseSemanticMask(ABC):
    """Abstract class to create semantic mask model"""

    def __init__(self, classifier):
        self.classifier=classifier # trained classifier model

    @abstractmethod
    def create_mask(self, image):
        pass