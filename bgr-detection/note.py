# parent
# ├── BGR-detection
# └── dataset
#     └── image/train
#     └── label/train



path: /data0/linhao/BGR-detection/dataset # dataset root dir
train: image/train
val: image/train
test:  # test images (optional)
label: label/train

# Classes
names:
  0: BGR
  1: RGB


  """
├── BGR-detection
 └── dataset
     └── image
       └── train
       └── test
     └── label/train
 └── bgr-detection
  └── data/bgr-data.yaml
  └── model/ResNet18.py
  └── runs/exp...
  └── test/result/
  └── utils
    └── dataLoader.py
    └── trainresnet.py
"""








class DogCatDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.label_name = {"BGR": 0, "RGB": 1}
        self.root_path = root_path
        self.transform = transform
        self.get_train_img_info()



    def __getitem__(self, index):
        self.img = cv2.imread(os.path.join(self.root_path, self.train_img_name[index]))
        if self.transform is not None:
            self.img = self.transform(self.img)
        self.label = self.train_img_label[index]
        return self.img, self.label

    def __len__(self):
        return len(self.train_img_name)

    def get_train_img_info(self):
        self.train_img_name = os.listdir(self.root_path)
        self.train_img_label = [0 if 'bgr' in imgname else 1 for imgname in self.train_img_name]


