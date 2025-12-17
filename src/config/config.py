from dataclasses import dataclass,field
from typing import Tuple

@dataclass
class DatasetConfig:
    #Configuration for dataset file paths and parameters
    root: str = "./data"
    train_images: str = "./data/GTSRB/Final_Training/Images"
    test_images: str = "./data/GTSRB 2/Final_Test/Images"
    concept_csv: str = "./data/concepts_per_class.csv"
    num_classes: int = 43
    image_size: Tuple[int, int] = (32, 32)
    val_split: float = 0.2
    batch_size: int = 64
    num_workers: int = 4

@dataclass
class ModelConfig:
    ##Configuration for model architecture parameters
    backbone: str = "efficientnet_v2_s"
    num_concepts: int = 15
    dropout: float = 0.3

@dataclass
class TrainingConfig:
    ##Configuration for training hyperparameters and environment
    lr: float = 1e-3
    epochs: int = 60 #trying to find optimal epoch
    patience: int = 10
    device: str = "cuda"
    checkpoint_dir: str = "./experiments/checkpoints" ##for saving model weights


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
