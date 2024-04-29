import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class RealFakeDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.classes = {'nature': 0, 'ai': 1}
        self.image_files = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for class_name, class_label in self.classes.items():
            class_folder = os.path.join(self.data_folder, class_name)
            for image_file in os.listdir(class_folder):
                self.image_files.append(os.path.join(class_folder, image_file))
                self.labels.append(class_label)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        image = self.transform(image)

        return image, self.labels[idx]
