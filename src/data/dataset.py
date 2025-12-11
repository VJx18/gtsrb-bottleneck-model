import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class GTSRBDataset(Dataset):
    def __init__(self, img_dir, concept_csv_path, split='train', val_split=0.2, transform=None, seed=42):
      
        self.img_dir = img_dir
        self.split = split
        self.val_split = val_split
        self.seed = seed
        self.transform = transform

        # load concepts
        if os.path.exists(concept_csv_path):
            self.concept_df = pd.read_csv(concept_csv_path)
            self.num_concepts = len(self.concept_df.columns) - 1
        else:
            self.concept_df = None
            self.num_concepts = 15

        self.image_paths = []
        self.labels = []
        self._load_images()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def _load_images(self):
        if not os.path.exists(self.img_dir):
            return

        all_image_paths = []
        all_labels = []

        for class_folder in os.listdir(self.img_dir):
            class_path = os.path.join(self.img_dir, class_folder)

            if os.path.isdir(class_path):
                class_id = int(class_folder)

                for img_file in os.listdir(class_path):
                    if img_file.endswith('.ppm') or img_file.endswith('.png'):
                        img_path = os.path.join(class_path, img_file)
                        all_image_paths.append(img_path)
                        all_labels.append(class_id)

        # for test split, use all data
        if self.split == 'test':
            # load test images
            for img_file in os.listdir(self.img_dir):
                    if img_file.endswith('.ppm') or img_file.endswith('.png'):
                        img_path = os.path.join(self.img_dir, img_file)
                        all_image_paths.append(img_path)
            self.image_paths = all_image_paths
            #self.labels = all_labels
            print(f"Loaded {len(self.image_paths)} test images")
            return

        # for train/val, split the data
        total_size = len(all_image_paths)
        indices = np.arange(total_size)

        # shuffle with seed
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        val_size = int(total_size * self.val_split)

        if self.split == 'val':
            split_indices = indices[:val_size]
        else:  # train
            split_indices = indices[val_size:]

        self.image_paths = [all_image_paths[i] for i in split_indices]
        self.labels = [all_labels[i] for i in split_indices]

        print(f"Loaded {len(self.image_paths)} {self.split} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label = self.labels[index]

        if self.concept_df is not None:
            concept_row = self.concept_df[self.concept_df['class_id'] == label]
            concept_vector = torch.tensor(concept_row.iloc[:, 2:].values.flatten(), dtype=torch.float32) ##get concept vector values i.e read everyrow but skip 1st and 2nd column (because they are class_id and class_name)
        else:
            concept_vector = torch.zeros(self.num_concepts, dtype=torch.float32)

        return image, (concept_vector, label)


def get_dataloaders(config):
    # train transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(config.dataset.image_size),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # test transforms
    test_transform = transforms.Compose([
        transforms.Resize(config.dataset.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # create train dataset
    train_dataset = GTSRBDataset(
        config.dataset.train_images,
        config.dataset.concept_csv,
        split='train',
        val_split=config.dataset.val_split,
        transform=train_transform
    )

    # create val dataset
    val_dataset = GTSRBDataset(
        config.dataset.train_images,
        config.dataset.concept_csv,
        split='val',
        val_split=config.dataset.val_split,
        transform=test_transform
    )

    # create test dataset
    test_dataset = GTSRBDataset(
        config.dataset.test_images,
        config.dataset.concept_csv,
        split='test',
        transform=test_transform
    )

    # create loaders
    train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size,
                            shuffle=True, num_workers=config.dataset.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size,
                          shuffle=False, num_workers=config.dataset.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.dataset.batch_size,
                           shuffle=False, num_workers=config.dataset.num_workers)

    return train_loader, val_loader, test_loader
