# data.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import cv2

class CustomCocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transforms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ])
        
        # COCO 어노테이션 로드
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        #print(f"{img_id} , {image.size}")
        
        #print(f'{img_path}, {idx}')
        # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # #print(f"image shape :{image.shape} ")
        # image = cv2.resize(image, (224, 224))


        # image = np.array(image)
        # image = np.moveaxis(image, -1, 0)  # (H, W, C) -> (C, H, W)

        # # Tensor로 변환
        # #image = torch.tensor(image, dtype=torch.float32)
        # image  = torch.from_numpy(image).float()

        # 어노테이션 정보 로드
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            image = self.transforms(image)
            print(f"image shape :{image.shape} ")
        
        return image, target
    
def collate_fn(batch):
# batch는 (image, label)의 튜플들이 묶인 리스트입니다
    images, labels = zip(*batch)
    
    # 이미지를 배치 차원으로 합칩니다. (batch_size, C, H, W)
    images = torch.stack(images, dim=0)
    print(f"STACK : {images.shape}")
    
    return images, labels
    

def get_data_loader(root_dir, annotation_file, batch_size=4, transforms=None):
    
    try : 
        dataset = CustomCocoDataset(root_dir, annotation_file)
    
    except Exception as e:
        # 에러가 발생하면 에러 메시지 출력
        print(f"에러 발생: {e}")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
