import os
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T

def apply_inverse_mask_torch(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    to_tensor = T.ToTensor()
    normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    denormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化回 [0,1]
    to_pil = T.ToPILImage()

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, filename)
        prefix = filename.replace('.png', '')
        mask_name = prefix + '_mask.png'  # 假设 mask 是 .png 格式
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {filename}")
            continue

        # 读取图像并转为张量
        img = Image.open(image_path).convert('RGB')
        img_tensor = to_tensor(img)  # [C, H, W], [0,1]
        img_tensor = normalize(img_tensor)  # [-1,1]

        # 读取mask并转为张量
        mask = Image.open(mask_path).convert("L")
        mask_tensor = to_tensor(mask)  # [1, H, W], [0,1]
        inv_mask = 1 - mask_tensor  # 遮挡区域为0 → 变成1（保留），其余为0

        # 扩展mask到3通道
        inv_mask_3ch = inv_mask.expand_as(img_tensor)

        # 相乘
        masked_tensor = img_tensor * inv_mask_3ch

        # 反归一化 → [0, 1]
        masked_tensor = denormalize(masked_tensor).clamp(0, 1)

        # 保存为图片
        masked_image = to_pil(masked_tensor)
        output_path = os.path.join(output_dir, filename.replace('.jpg', '.png'))
        masked_image.save(output_path)

# 示例使用
if __name__ == "__main__":
    image_folder = "datasets/person/customize/video/00001/images"
    mask_folder = "datasets/person/customize/video/00001/mask"
    output_folder = "datasets/person/customize/video/00001/agnostic"
    apply_inverse_mask_torch(image_folder, mask_folder, output_folder)
