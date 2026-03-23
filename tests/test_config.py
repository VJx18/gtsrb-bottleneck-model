from src.config.config import DatasetConfig, TrainingConfig, ModelConfig

def test_datasetconfig_defaults():
    dataset = DatasetConfig()
    assert dataset.num_classes == 43
    assert dataset.val_split == 0.2
    assert dataset.batch_size == 64
    assert dataset.seed == 42
    assert "./data" in dataset.root
    assert "./data/GTSRB/Final_Training/Images" in dataset.train_images
    assert "./data/GTSRB 2/Final_Test/Images" in dataset.test_images
    assert "./data/concepts_per_class.csv" in dataset.concept_csv
    assert "./data/GT-final_test.csv" in dataset.class_id_test_csv

def test_trainingconfig_defaults():
    training = TrainingConfig()
    assert training.lr == 1e-4
    assert training.epochs == 60
    assert training.patience == 6
    assert "cuda" in training.device
    assert "./experiments/checkpoints" in training.checkpoint_dir

def test_modelconfig_defaults():
    model = ModelConfig()
    assert model.num_concepts == 43
    assert "efficientnet_v2_s" in model.backbone

