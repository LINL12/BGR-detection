import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import yaml
from model import ResNet18

# 读取.yaml
def read_yaml_config(config_file):
    with open(config_file, "r") as file:
        data = yaml.safe_load(file)

    data_root = data['path']
    train_path = os.path.join(data_root, data['train'])
    label_path = os.path.join(data_root, data['label'])
    class_names = data['names']

    return data_root, train_path, label_path, class_names

#数据集标准化
def load_dataset(data_root, train_path, label_path, class_names, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #自定义数据集
    class CustomDataset(Dataset):
        def __init__(self, data_dir, label_dir, class_names, transform=None):
            self.data_dir = data_dir
            self.label_dir = label_dir
            self.class_names = class_names
            self.transform = transform
            self.images = os.listdir(data_dir)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image_name = self.images[idx]

            image_path = os.path.join(self.data_dir, image_name)
            label_path = os.path.join(self.label_dir, f"{os.path.splitext(image_name)[0]}.txt")

            image = Image.open(image_path).convert('RGB')

            with open(label_path, 'r') as label_file:
                label = int(label_file.read().strip())

            if self.transform:
                image = self.transform(image)

            return image, label

    #数据加载器
    train_dataset = CustomDataset(data_dir=train_path, label_dir=label_path, class_names=class_names, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader



