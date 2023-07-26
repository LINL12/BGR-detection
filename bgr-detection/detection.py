import os
import shutil
from tqdm import tqdm  
from PIL import Image
import torch
import torchvision.transforms as transforms

from model.ResNet18 import ResNet18
from utils.trainresnet import GPUinfo


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#模型加载
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=2, in_channels=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device

#将测试结果保存到 ./test/result 路径下
def create_result_dir(result_path):
    result_dirs = [d for d in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, d))]
    max_num = max([int(d.split("result")[-1]) for d in result_dirs if d.startswith("result")] + [0])

    new_result_dir = os.path.join(result_path, f"result{max_num + 1}")
    os.makedirs(new_result_dir, exist_ok=True)

    return new_result_dir

#逐张检测，将检测结果与标签对照后分类
def batch_detect(model, device, data_path, result_path):
    new_result_dir = create_result_dir(result_path)
    result_1_dir = os.path.join(new_result_dir, "result_1")
    result_2_dir = os.path.join(new_result_dir, "result_2")

    os.makedirs(result_1_dir, exist_ok=True)
    os.makedirs(result_2_dir, exist_ok=True)

    image_list = os.listdir(data_path)
    for image_file in tqdm(image_list, desc="Processing"):  
        image_path = os.path.join(data_path, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = output.max(1)

        if predicted.item() == 0:
            shutil.copy(image_path, os.path.join(result_1_dir, image_file))
        else:
            shutil.copy(image_path, os.path.join(result_2_dir, image_file))


if __name__ == "__main__":
    
    GPUinfo()
    model_path = "/data0/linhao/BGR-detection/bgr-detection/runs/exp10/best_model.pth"
    data_path = "/data0/linhao/BGR-detection/dataset/image/test"
    result_path = "/data0/linhao/BGR-detection/bgr-detection/test/result"

    model, device = load_model(model_path)
    batch_detect(model, device, data_path, result_path)
