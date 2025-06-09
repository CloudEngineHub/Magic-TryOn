import os
import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 设置图像文件夹路径
image_folder = "datasets/garment/vivo/vivo_garment"  # ← 修改为你的图像文件夹路径
output_folder = "datasets/garment/vivo/"

# 设置图像文件夹路径
output_path = os.path.join(output_folder, "vivo_caption_qwen.json")

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)


# 加载预处理器
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# 获取并排序图像路径列表
supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
image_paths = sorted(
    [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(supported_exts)],
    key=lambda x: os.path.basename(x)
)

# 加载已有结果（如果有）
if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        existing_names = set(entry['name'] for entry in results)
else:
    results = []
    existing_names = set()

# 遍历图像并处理
batch_results = []
for idx, image_path in enumerate(image_paths, 1):
    image_name = os.path.basename(image_path)
    if image_name in existing_names:
        continue  # 跳过已处理图像

    print(f"[{idx}] Processing: {image_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "This is an image of a garment. Describe the features of the garment in detail, starting the description with 'a' or 'an'. Avoid phrases like 'an image' or 'a photo' and focus directly on the appearance, such as color, pattern, style, and any visible text or graphics."}
            ],
        }
    ]

    # 文本处理与输入准备
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    description = output_text[0]
    print("Output:", description)

    batch_results.append({'name': image_name, 'describe': description})
    existing_names.add(image_name)

    # 每1000张保存一次
    if idx % 1000 == 0 or idx == len(image_paths):
        print(f"Saving intermediate results... ({len(results) + len(batch_results)} total)")
        results.extend(batch_results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        batch_results = []

print(f"\n✅ 所有图像已处理并保存至：{output_path}")
