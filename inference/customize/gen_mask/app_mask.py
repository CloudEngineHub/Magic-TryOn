from PIL import Image
import torch
import os
import numpy as np
from preprocess.utils_mask import get_mask_location
from torchvision import transforms
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


parsing_model = Parsing(0)
openpose_model = OpenPose(0)


tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )


def get_mask(img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed, im_height, im_width,):
    
    openpose_model.preprocessor.body_estimation.model.to(device)


    garm_img= garm_img.convert("RGB").resize((im_width,im_height))
    human_img_orig = img.convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((im_width,im_height))
    else:
        human_img = human_img_orig.resize((im_width,im_height))



    keypoints = openpose_model(human_img.resize((384,512)))
    model_parse, _ = parsing_model(human_img.resize((384,512)))
    mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
    #### if lower_body:
    # mask, _ = get_mask_location('dc', "lower_body", model_parse, keypoints)
    #### if dresses:
    # mask, _ = get_mask_location('dc', "dresses", model_parse, keypoints)
    mask = mask.resize((im_width,im_height))
    return mask
    

# 路径设置
image_dir = "datasets/person/customize/video/00001/images"
mask_dir = "datasets/person/customize/video/00001/mask"

# 创建输出目录
os.makedirs(mask_dir, exist_ok=True)

# 处理所有图像
for filename in sorted(os.listdir(image_dir)):
    if filename.endswith(".png"):
        img_path = os.path.join(image_dir, filename)
        imgs = Image.open(img_path).convert("RGB")  # 保证是RGB格式

        # 获取图像的宽和高
        width, height = imgs.size

        # 获取mask
        mask = get_mask(imgs, imgs, "Model", False, False, 30, 42, height, width,).convert("L")  # 转换为灰度图

        # 保存mask
        mask_filename = filename.replace(".png", "_mask.png")
        mask.save(os.path.join(mask_dir, mask_filename))


        print(f"Processed and saved: {mask_filename}")