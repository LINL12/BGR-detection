import os
import shutil
from tqdm import tqdm
import torch
import torch.nn.init as init
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import ResNet18




#显卡信息
def GPUinfo():
    ng = torch.cuda.device_count()
    infos = [torch.cuda.get_device_properties(i) for i in range(ng)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print("Devices:%d" %ng)
    print(infos)


#模型参数初始化
def initialize_model_params(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


#训练
def train_model(data_root, train_path, label_path, class_names,lr, epochs, batch_size, net, train_loader, criterion, optimizer, scheduler):
    
    GPUinfo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    initialize_model_params(net)

    # 保存模型和结果曲线图到 ./runs/路径下
    if not os.path.exists("/data0/linhao/BGR-detection/bgr-detection/runs"):
        os.makedirs("/data0/linhao/BGR-detection/bgr-detection/runs")

    exp_num = 1
    while os.path.exists(f"/data0/linhao/BGR-detection/bgr-detection/runs/exp{exp_num}"):
        exp_num += 1
    os.makedirs(f"/data0/linhao/BGR-detection/bgr-detection/runs/exp{exp_num}")


    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    best_acc = 0.0
    for epoch in range(epochs):

        net.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}, Training")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(loss=total_loss / (len(train_loader) + 1), accuracy=100. * correct_train / total_train)


        train_loss = total_loss / len(train_loader)
        train_accuracy = 100. * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 学习率调整
        scheduler.step()


        net.eval()
        total_test = 0
        correct_test = 0
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}, Testing"):
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        test_loss = total_loss / len(train_loader)
        test_accuracy = 100. * correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # 保存最新模型和最好模型
        torch.save(net.state_dict(), f"/data0/linhao/BGR-detection/bgr-detection/runs/exp{exp_num}/latest_model.pth")
        #成功率覆盖
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            shutil.copyfile(f"/data0/linhao/BGR-detection/bgr-detection/runs/exp{exp_num}/latest_model.pth",
                            f"/data0/linhao/BGR-detection/bgr-detection/runs/exp{exp_num}/best_model.pth")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs') 
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.savefig(f"/data0/linhao/BGR-detection/bgr-detection/runs/exp{exp_num}/training_plot.png")


