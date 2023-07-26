import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from model.ResNet18 import ResNet18
from utils.dataLoader import load_dataset
from utils.dataLoader import read_yaml_config
from utils.trainresnet import train_model




if __name__ == "__main__":

    learning_rate = 0.01
    num_epochs = 5
    batch_size = 32

    config_file = "/data0/linhao/BGR-detection/bgr-detection/data/bgr-data.yaml"
    data_root, train_path, label_path, class_names = read_yaml_config(config_file)

    train_loader_d = load_dataset(data_root, train_path, label_path, class_names, batch_size)

    """2分类任务，3通道图像"""
    net_d = ResNet18(num_classes=2, in_channels=3)

    criterion_d = nn.CrossEntropyLoss()
    optimizer_d = torch.optim.SGD(net_d.parameters(), lr=learning_rate, momentum=0.9)
    scheduler_d = StepLR(optimizer_d, step_size=3, gamma=0.1)

    #训练
    train_model(data_root, train_path, label_path, class_names, lr=learning_rate, epochs=num_epochs, 
    batch_size=batch_size, net=net_d, train_loader=train_loader_d, criterion=criterion_d, optimizer=optimizer_d, scheduler=scheduler_d)
