import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class GTSRBDataset(Dataset): ##manual iteration over batches don't offer shuffling which increases risks of overfitting so we use Dataset and dataloader

      def __init__(self, img_dir, concept_csv_path, 
  transform=None):  ##how and waht data to load 
          self.img_dir = img_dir
          self.transform = transform

          if os.path.exists(concept_csv_path):
              self.concept_df = pd.read_csv(concept_csv_path)
              self.num_concepts = len(self.concept_df.columns) - 1 #calculate how many concept features are there except columndID 
          else:
              self.concept_df = None
              self.num_concepts = 15

          self.image_paths = []  ##populating to be done later
        
          self.labels = []
          self._load_images()

          if transform is None:   ##resizing to 32x32
              self.transform = transforms.Compose([
                  transforms.Resize((32, 32)),
                  transforms.ToTensor(),  ##convert to tensor
                  ##transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
              ])

      def _load_images(self):
          if not os.path.exists(self.img_dir):
              return

          for class_folder in os.listdir(self.img_dir):
              class_path = os.path.join(self.img_dir, class_folder)

              if os.path.isdir(class_path):
                  class_id = int(class_folder)

                  for img_file in os.listdir(class_path):
                      if img_file.endswith('.ppm') or img_file.endswith('.png'):
                          img_path = os.path.join(class_path, img_file)
                          self.image_paths.append(img_path)
                          self.labels.append(class_id)

      def __len__(self): ## total number of samples 
          return len(self.image_paths)

      def __getitem__(self, index): ## returns data and (label) at given index 
          img_path = self.image_paths[index]
          image = Image.open(img_path).convert('RGB')  ##to make sure every image has 3 channels so input shape is always consisten ,for eg : [3,32,32]
          image = self.transform(image)

          label = self.labels[index]

          if self.concept_df is not None:
              concept_row =self.concept_df[self.concept_df['ClassId'] == label]
              concept_vector = torch.tensor(concept_row.iloc[:,1:].values.flatten(), dtype=torch.float32)  ##get concept vector values i.e read everyrow but skip 1st column(as it's the columnId )
          else:
              concept_vector = torch.zeros(self.num_concepts,dtype=torch.float32) ##fill with zeroes if no csv 

          return image, (concept_vector, label) 
