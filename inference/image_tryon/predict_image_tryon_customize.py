import os
import sys
import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
from pathlib import Path
from einops import rearrange
import torchvision
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from magic_tryon.dist import set_multi_gpus_devices
from magic_tryon.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel)
from magic_tryon.models.wan_transformer3d_tryon import WanTransformer3DModel
from magic_tryon.models.cache_utils import get_teacache_coefficients
from magic_tryon.pipeline.pipeline_wan_tryon import WanFunInpaintPipeline
from magic_tryon.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from magic_tryon.utils.lora_utils import merge_lora, unmerge_lora
from magic_tryon.utils.utils import (filter_kwargs,
                                    get_video_to_video_latent_tryon_full,
                                    get_image_latent_tryon_full,
                                    save_videos_grid)
import json
import math

# Inference Config
GPU_memory_mode     = "sequential_cpu_offload"
ulysses_degree      = 1
ring_degree         = 1
enable_teacache     = True
teacache_threshold  = 0.10
num_skip_start_steps = 5
teacache_offload    = False

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "weights/MagicTryOn_14B_V1"

# Choose the sampler in "Flow"
sampler_name        = "Flow"

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    return width, height, frame_count, fps

def get_image_size(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return width, height

def repaint(
    person: torch.Tensor,
    mask: torch.Tensor,
    result: torch.Tensor,
    kernal_size: int=None,
    ):
    if kernal_size is None:
        h = person.size(-1)
        kernal_size = h // 50
        if kernal_size % 2 == 0:
            kernal_size += 1
    # Apply 2D average pooling on mask video
    # (B, C, F, H, W) -> (B*F, C, H, W)
    mask = rearrange(mask, 'b c f h w -> (b f) c h w')
    mask = torch.nn.functional.avg_pool2d(mask, kernal_size, stride=1, padding=kernal_size // 2)
    mask = rearrange(mask, '(b f) c h w -> b c f h w', b=person.size(0))
    # Use mask video to repaint result video
    result = person * (1 - mask) + result * mask
    return result

def get_description_from_json(cloth_image, json_path):
    """
    从 JSON 文件中读取指定图像名的描述信息。

    参数:
        cloth_image (str): 图像文件名，如 '000123_1.jpg'
        json_path (str): JSON 文件路径，如 '/path/to/caption_qwen.json'

    返回:
        str 或 None: 对应图像的描述文本，若未找到返回 None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if item['name'] == cloth_image:
                return item['describe']
        return None  # 若未找到
    except Exception as e:
        print(f"读取 JSON 文件失败: {e}")
        return None
    
def adjust_size(value):
    # Round up to divisible by 8 first
    if value % 8 != 0:
        value = math.ceil(value / 8) * 8
    # Then adjust until (value // 8) % 10 == 4 or 8
    while True:
        last_digit = (value // 8) % 10
        if last_digit in [4, 8]:
            return value
        value += 8  # Increment by 8 until it meets the condition
# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16


to_video = False
use_repaint = True


image_id = "00001"

org_video_path    = f'datasets/person/customize/image/{image_id}/images/0000.png'
masked_video_path = f'datasets/person/customize/image/{image_id}/agnostic/0000.png'
mask_video_path   = f'datasets/person/customize/image/{image_id}/mask/0000_mask.png'
pose_video_path   = f'datasets/person/customize/image/{image_id}/image-densepose/0000.png'

cloth_id        = "00002.png" 
cloth_image_path        = os.path.join("datasets/garment/vivo/vivo_garment/", cloth_id)
cloth_line_image_path   = os.path.join("datasets/garment/vivo/vivo_garment_anilines/", cloth_id)
cloth_text = get_description_from_json(cloth_id, "datasets/garment/vivo/vivo_caption_qwen.json") 

width, height, frame_count, v_fps = get_video_properties(org_video_path)
width = adjust_size(width)
height = adjust_size(height)
# Other params
if to_video:
    print("set video")
    sample_size         = [height, width]
    video_length        = 16
    fps                 = 16
else:
    sample_size         = [height, width]
    video_length        = 1
    fps                 = 1


prompt = "The image depicts an anime-style character sitting on a grassy field under a bright, sunny sky with fluffy clouds. The character has long dark hair adorned with a blue ribbon and a small flower accessory. She is wearing a " + cloth_text
print(prompt)
negative_prompt     = "衣服不真实的，没有穿着感的，衣服质量差的，纽扣不整齐的，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


guidance_scale          = 3.0
seed                    = 43
num_inference_steps     = 20
lora_weight             = 0.55
save_path               = "samples/image-cus"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
).to(weight_dtype)
text_encoder = text_encoder.eval()

# Get Clip Image Encoder
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype)
clip_image_encoder = clip_image_encoder.eval()

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanFunInpaintPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder
)
if ulysses_degree > 1 or ring_degree > 1:
    transformer.enable_multi_gpus_inference()

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    input_video, masked_video, mask_video, pose_video, clip_image, cloth_image, cloth_line_image = get_image_latent_tryon_full(
                                        org_video_path, masked_video_path, mask_video_path, pose_video_path, 
                                        video_length=video_length, sample_size=sample_size, fps=fps, ref_image=cloth_image_path, line_image_path = cloth_line_image_path)

    if to_video:
        input_video = input_video.repeat(1, 1, video_length, 1, 1)
        masked_video = masked_video.repeat(1, 1, video_length, 1, 1)
        mask_video = mask_video.repeat(1, 1, video_length, 1, 1)
        pose_video = pose_video.repeat(1, 1, video_length, 1, 1)

    sample = pipeline(
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        ### condition ###
        video = input_video,
        masked_video = masked_video,
        mask_video = mask_video,
        pose_video = pose_video,
        cloth_image = cloth_image,
        cloth_line_image = cloth_line_image,
        clip_image = clip_image,
    ).videos

    if use_repaint:
        results = repaint(input_video, mask_video, sample)
    else:
        results = sample

    # 打印最小值和最大值
    print("Sample min value:", sample.min().item())
    print("Sample max value:", sample.max().item())

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)

os.makedirs(os.path.join(save_path, "sanity_check"), exist_ok=True)
for idx, (pixel_value, masked_video_value, control_pixel_value, mask_value, cloth_pixel_value, cloth_line_pixel_value) in enumerate(zip(
    input_video, masked_video, pose_video, mask_video, cloth_image, cloth_line_image )):
    pixel_value = pixel_value[None, ...]
    control_pixel_value = control_pixel_value[None, ...]
    mask_value = mask_value[None, ...]
    cloth_pixel_value = cloth_pixel_value[None, ...]
    cloth_line_pixel_value = cloth_line_pixel_value[None, ...]
    masked_video_value = masked_video_value[None, ...]

    gif_name = 'input'
    save_videos_grid(pixel_value, f"{save_path}/sanity_check/{gif_name[:10]}.mp4", fps=fps, rescale=False)
    save_videos_grid(masked_video_value, f"{save_path}/sanity_check/{gif_name[:10]}_masked_video.mp4", fps=fps, rescale=False)
    save_videos_grid(cloth_pixel_value, f"{save_path}/sanity_check/{gif_name[:10]}_cloth.mp4", fps=fps, rescale=False)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = results[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(results, video_path, fps=fps, rescale=False)

if to_video:
    results = results[:, :, 0:1, :, :]
    video_length = 1
if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()