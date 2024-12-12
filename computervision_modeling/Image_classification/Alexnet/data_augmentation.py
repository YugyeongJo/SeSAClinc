import os
import cv2
import imgaug.augmenters as iaa
from imgaug import parameters as iap


base_dir = "./deep_dataset"
folders = ["dry", "oily", "normal"]


color_augmenter = iaa.Sequential([
    iaa.MultiplyHueAndSaturation((0.5, 1.5)), 
    iaa.AddToBrightness((-30, 30)),          
])


for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    save_folder = os.path.join(base_dir, f"{folder}_augmented")
    os.makedirs(save_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        
 
        image = cv2.imread(img_path)
        if image is None:
            continue

        augmented_image = color_augmenter(image=image)
        

        save_path = os.path.join(save_folder, f"aug_{filename}")
        cv2.imwrite(save_path, augmented_image)

print("색상 증강 완료!")
