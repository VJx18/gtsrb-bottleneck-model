import pytest
import torch
from unittest.mock import patch, mock_open
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

from src.data.dataset import GTSRBDataset, get_dataloaders


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir_with_images():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / "00000").mkdir()
        (root / "00001").mkdir()
        (root / "00002").mkdir()
        (root / "00003").mkdir()
        (root / "00004").mkdir()

        # 5 Fake-Bilder
        img1 = Image.new('RGB', (64, 64), color='red')
        img2 = Image.new('RGB', (64, 64), color='blue')
        img3 = Image.new('RGB', (64, 64), color='brown')
        img4 = Image.new('RGB', (64, 64), color='white')
        img5 = Image.new('RGB', (64, 64), color='black')

        img1.save(root / "00000" / "00000_00000.ppm")
        img2.save(root / "00001" / "00001_00000.ppm")
        img3.save(root / "00002" / "00002_00000.ppm")
        img1.save(root / "00003" / "00003_00000.ppm")
        img1.save(root / "00004" / "00004_00000.ppm")

        yield str(root)

@pytest.fixture
def temp_dir_test():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / "Test").mkdir()

        img1 = Image.new('RGB', (64, 64), color='white')
        img2 = Image.new('RGB', (64, 64), color='black')

        img1.save(root / "Test" / "test_001.ppm")
        img2.save(root / "Test" / "test_002.ppm")

        yield(str(root))


@pytest.fixture
def fake_concept_csv(tmp_path):
    df = pd.DataFrame({
        "class_id": [0, 1, 2, 3, 4],
        "class_name": ["stop", "yield", "construction", "ice", "sharp curve to left"],
        "concept_a": [1, 0, 0, 1, 0],
        "concept_b": [0, 1, 1, 0, 0],
        "concept_c": [1, 1, 0, 0, 1],
    })
    path = tmp_path / "concepts.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fake_test_class_csv(tmp_path):
    content = """filename,class_id
test_001.ppm,5
test_002.ppm,17"""
    path = tmp_path / "test_gt.csv"
    path.write_text(content)
    return str(path)


# ────────────────────────────────────────────────────────────────
# GTSRBDataset – Basis-Tests
# ────────────────────────────────────────────────────────────────

def test_init_sets_attributes(temp_dir_with_images, fake_concept_csv):
    ds = GTSRBDataset(
        img_dir=temp_dir_with_images,
        concept_csv_path=fake_concept_csv,
        split="train",
        val_split=0.2,
        seed=42
    )

    assert ds.img_dir == temp_dir_with_images
    assert ds.split == "train"
    assert ds.val_split == 0.2
    assert ds.seed == 42
    assert ds.num_concepts == 3 
    assert len(ds.image_paths) > 0


def test_train_val_split_is_reproducible(temp_dir_with_images):
    ds1 = GTSRBDataset(temp_dir_with_images, "", split="train", seed=123)
    ds2 = GTSRBDataset(temp_dir_with_images, "", split="train", seed=123)

    assert ds1.image_paths == ds2.image_paths
    assert ds1.labels == ds2.labels


def test_val_split_is_complementary_to_train(temp_dir_with_images):
    train_ds = GTSRBDataset(temp_dir_with_images, "", split="train", val_split=0.2, seed=7)
    val_ds   = GTSRBDataset(temp_dir_with_images, "", split="val",   val_split=0.2, seed=7)

    train_set = set(train_ds.image_paths)
    val_set   = set(val_ds.image_paths)

    assert len(train_set & val_set) == 0
    assert len(train_set | val_set) == 5   # 2 Bilder auf train und val verteilt
    assert len(train_set) == 4
    assert len(val_set) == 1



# ────────────────────────────────────────────────────────────────
# __getitem__ Tests
# ────────────────────────────────────────────────────────────────

def test_getitem_returns_correct_shapes(temp_dir_with_images, fake_concept_csv):
    ds = GTSRBDataset(
        img_dir=temp_dir_with_images,
        concept_csv_path=fake_concept_csv,
        split="train",
        transform=None
    )

    img, (concepts, label) = ds[0]

    assert img.shape == (3, 32, 32)           # Default transform
    assert concepts.shape == (3,)
    assert label in [0, 1, 2, 3, 4]


def test_getitem_uses_correct_concept_vector(temp_dir_with_images, fake_concept_csv):
    ds = GTSRBDataset(temp_dir_with_images, fake_concept_csv, split="train")

    index_0 = None
    for i in range(len(ds)):
        _, (_, label) = ds[i]
        if label == 0:
            index_0 = i
            break
    
    assert index_0 is not None

    _, (concepts, label) = ds[index_0]
    assert label == 0
    expected = torch.tensor([1., 0., 1.], dtype=torch.float32)
    assert torch.allclose(concepts, expected, atol=1e-6)



# ────────────────────────────────────────────────────────────────
# get_dataloaders
# ────────────────────────────────────────────────────────────────

class FakeConfig:
    class dataset:
        train_images = ""
        test_images = ""
        concept_csv = ""
        class_id_test_csv = ""
        val_split = 0.2
        seed = 42
        batch_size = 4
        num_workers = 0
        image_size = (32, 32)


@patch.multiple("src.data.dataset.GTSRBDataset",
                __len__=lambda s: 8,
                __getitem__=lambda s, i: (torch.rand(3,32,32), (torch.zeros(3), 1)))
def test_dataloaders_return_correct_batch_shapes(temp_dir_with_images, fake_concept_csv, monkeypatch):
    cfg = FakeConfig()
    cfg.dataset.train_images = temp_dir_with_images
    cfg.dataset.test_images = temp_dir_with_images
    cfg.dataset.concept_csv = fake_concept_csv
    cfg.dataset.class_id_test_csv = ""

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    for loader in [train_loader, val_loader, test_loader]:
        batch = next(iter(loader))
        images, (concepts, labels) = batch

        assert images.shape == (4, 3, 32, 32)   # batch_size=4
        assert concepts.shape == (4, 3)
        assert labels.shape == (4,)