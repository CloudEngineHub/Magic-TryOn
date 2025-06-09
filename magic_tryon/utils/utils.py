import os
import gc
import imageio
import inspect
import numpy as np
import torch
import torchvision
import cv2
from einops import rearrange
from PIL import Image
from torchvision import transforms

def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider

def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst

def save_flows_as_gif(forward_flow_values, output_path, fps=12):
    """
    保存光流为GIF视频。

    参数:
    - forward_flow_values: 光流数据，形状为 (batch_size, channels, num_frames, height, width)。
    - output_path: 输出GIF文件路径。
    - fps: GIF的帧率。
    """
    batch_size, channels, num_frames, height, width = forward_flow_values.shape

    for b in range(batch_size):
        frames = []
        for f in range(num_frames):
            # 提取光流的水平和垂直分量
            flow_x = forward_flow_values[b, 0, f]
            flow_y = forward_flow_values[b, 1, f]

            if isinstance(flow_x, torch.Tensor):
                flow_x = flow_x.detach().cpu().numpy()
            if isinstance(flow_y, torch.Tensor):
                flow_y = flow_y.detach().cpu().numpy()

            # 计算光流的角度和幅度
            magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

            # 创建一个HSV图像用于可视化
            hsv = np.zeros((height, width, 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2  # 角度转换为色调
            hsv[..., 1] = 255  # 饱和度设为最大值
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 幅度转换为亮度

            # 转换HSV图像为BGR格式
            bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 转换BGR为RGB格式，因为imageio需要RGB格式
            rgb_flow = cv2.cvtColor(bgr_flow, cv2.COLOR_BGR2RGB)

            # 添加到帧列表
            frames.append(rgb_flow)

        # 保存为GIF
        imageio.mimsave(output_path, frames, fps=fps)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=True, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)

def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], 
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video
            
            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None
        
        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image

def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = input_video_path

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        if validation_video_mask is not None:
            validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
            input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, input_video_mask, ref_image, clip_image


def get_video_to_video_latent_tryon_baseline(org_video_path, masked_video_path, mask_video_path, pose_video_path, 
                                    video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if org_video_path is not None:
        if isinstance(org_video_path, str):
            cap = cv2.VideoCapture(org_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = org_video_path

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
        # input_video = input_video * 2 - 1

        if isinstance(masked_video_path, str):
            cap = cv2.VideoCapture(masked_video_path)
            masked_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    masked_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            masked_video = masked_video_path

        masked_video = torch.from_numpy(np.array(masked_video))[:video_length]
        masked_video = masked_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
        # masked_video = masked_video * 2 - 1

        if isinstance(mask_video_path, str):
            cap = cv2.VideoCapture(mask_video_path)
            mask_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    mask_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            mask_video = mask_video_path

        mask_video = torch.from_numpy(np.array(mask_video))[:video_length]
        mask_video = mask_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        mask_video[mask_video < 0.5] = 0
        mask_video[mask_video >= 0.5] = 1

        is_binary = torch.all(torch.logical_or(mask_video == 0, mask_video == 1))
        if is_binary:
            print("mask_latents contains only 0 or 1.")
        else:
            print("mask_latents contains values other than 0 or 1.")
    

        if isinstance(pose_video_path, str):
            cap = cv2.VideoCapture(pose_video_path)
            pose_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    pose_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            pose_video = pose_video_path

        pose_video = torch.from_numpy(np.array(pose_video))[:video_length]
        pose_video = pose_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
        # pose_video = pose_video * 2 - 1

    
        if validation_video_mask is not None:
            validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
            input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")

            cloth_image = Image.open(ref_image).convert("RGB")
            cloth_image = np.array(cloth_image)  # 转换为NumPy数组
            cloth_image = cv2.resize(cloth_image, (sample_size[1], sample_size[0]))
            cloth_image = torch.from_numpy(cloth_image)  # 转换为PyTorch张量
            cloth_image = cloth_image.unsqueeze(0) #1,h,w,c
            cloth_image = cloth_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w
            # cloth_image = cloth_image * 2 - 1
            

        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))

    else:
        clip_image = None


    return input_video, masked_video, mask_video, pose_video, input_video_mask, clip_image, cloth_image


def get_video_to_video_latent_tryon_full(org_video_path, masked_video_path, mask_video_path, pose_video_path, 
                                    video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None, line_image_path=None):
    if org_video_path is not None:
        if isinstance(org_video_path, str):
            cap = cv2.VideoCapture(org_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = org_video_path

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
        # input_video = input_video * 2 - 1

        if isinstance(masked_video_path, str):
            cap = cv2.VideoCapture(masked_video_path)
            masked_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    masked_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            masked_video = masked_video_path

        masked_video = torch.from_numpy(np.array(masked_video))[:video_length]
        masked_video = masked_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
        # masked_video = masked_video * 2 - 1

        if isinstance(mask_video_path, str):
            cap = cv2.VideoCapture(mask_video_path)
            mask_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    mask_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            mask_video = mask_video_path

        mask_video = torch.from_numpy(np.array(mask_video))[:video_length]
        mask_video = mask_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        mask_video[mask_video < 0.5] = 0
        mask_video[mask_video >= 0.5] = 1

        is_binary = torch.all(torch.logical_or(mask_video == 0, mask_video == 1))
        if is_binary:
            print("mask_latents contains only 0 or 1.")
        else:
            print("mask_latents contains values other than 0 or 1.")
    

        if isinstance(pose_video_path, str):
            cap = cv2.VideoCapture(pose_video_path)
            pose_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    pose_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            pose_video = pose_video_path

        pose_video = torch.from_numpy(np.array(pose_video))[:video_length]
        pose_video = pose_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
        # pose_video = pose_video * 2 - 1

    
        if validation_video_mask is not None:
            validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
            input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")

            cloth_image = Image.open(ref_image).convert("RGB")
            cloth_image = np.array(cloth_image)  # 转换为NumPy数组
            cloth_image = cv2.resize(cloth_image, (sample_size[1], sample_size[0]))
            cloth_image = torch.from_numpy(cloth_image)  # 转换为PyTorch张量
            cloth_image = cloth_image.unsqueeze(0) #1,h,w,c
            cloth_image = cloth_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w
            # cloth_image = cloth_image * 2 - 1

            line_image = Image.open(line_image_path).convert("RGB")
            line_image = np.array(line_image)  # 转换为NumPy数组
            line_image = cv2.resize(line_image, (sample_size[1], sample_size[0]))
            line_image = torch.from_numpy(line_image)  # 转换为PyTorch张量
            line_image = line_image.unsqueeze(0) #1,h,w,c
            line_image = line_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w
            # line_image = line_image * 2 - 1
            

        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))

    else:
        clip_image = None


    return input_video, masked_video, mask_video, pose_video, input_video_mask, clip_image, cloth_image, line_image


def get_image_latent_tryon_full(org_video_path, masked_video_path, mask_video_path, pose_video_path, 
                                    video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None, line_image_path=None):

    # org image 
    org_image = Image.open(org_video_path).convert("RGB")
    org_image = np.array(org_image)  # 转换为NumPy数组
    org_image = cv2.resize(org_image, (sample_size[1], sample_size[0]))
    org_image = torch.from_numpy(org_image)  # 转换为PyTorch张量
    org_image = org_image.unsqueeze(0) #1,h,w,c
    org_image = org_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

    # masked image
    masked_image = Image.open(masked_video_path).convert("RGB")
    masked_image = np.array(masked_image)  # 转换为NumPy数组
    masked_image = cv2.resize(masked_image, (sample_size[1], sample_size[0]))
    masked_image = torch.from_numpy(masked_image)  # 转换为PyTorch张量
    masked_image = masked_image.unsqueeze(0) #1,h,w,c
    masked_image = masked_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

    # mask image
    mask_image = Image.open(mask_video_path).convert("RGB")
    mask_image = np.array(mask_image)  # 转换为NumPy数组
    mask_image = cv2.resize(mask_image, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_NEAREST)
    mask_image = torch.from_numpy(mask_image)  # 转换为PyTorch张量
    mask_image = mask_image.unsqueeze(0) #1,h,w,c
    mask_image = mask_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w
    mask_image[mask_image < 0.5] = 0
    mask_image[mask_image >= 0.5] = 1

     # pose image 
    pose_image = Image.open(pose_video_path).convert("RGB")
    pose_image = np.array(pose_image)  # 转换为NumPy数组
    pose_image = cv2.resize(pose_image, (sample_size[1], sample_size[0]))
    pose_image = torch.from_numpy(pose_image)  # 转换为PyTorch张量
    pose_image = pose_image.unsqueeze(0) #1,h,w,c
    pose_image = pose_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

        
    # cloth image 
    cloth_image = Image.open(ref_image).convert("RGB")
    cloth_image = np.array(cloth_image)  # 转换为NumPy数组
    cloth_image = cv2.resize(cloth_image, (sample_size[1], sample_size[0]))
    cloth_image = torch.from_numpy(cloth_image)  # 转换为PyTorch张量
    cloth_image = cloth_image.unsqueeze(0) #1,h,w,c
    cloth_image = cloth_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

    # line image
    line_image = Image.open(line_image_path).convert("RGB")
    line_image = np.array(line_image)  # 转换为NumPy数组
    line_image = cv2.resize(line_image, (sample_size[1], sample_size[0]))
    line_image = torch.from_numpy(line_image)  # 转换为PyTorch张量
    line_image = line_image.unsqueeze(0) #1,h,w,c
    line_image = line_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

    clip_image = Image.open(ref_image).convert("RGB")


    return org_image, masked_image, mask_image, pose_image, clip_image, cloth_image, line_image


def get_vtion_image_latent_tryon_full(org_video_path, masked_video_path, mask_video_path, pose_video_path, 
                                    video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None, line_image_path=None):

    # org image 
    org_image = Image.open(org_video_path).convert("RGB")
    org_image = np.array(org_image)  # 转换为NumPy数组
    org_image = cv2.resize(org_image, (sample_size[1], sample_size[0]))
    org_image = torch.from_numpy(org_image)  # 转换为PyTorch张量
    org_image = org_image.unsqueeze(0) #1,h,w,c
    org_image = org_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

    # masked image
    masked_image = Image.open(masked_video_path).convert("RGB")
    masked_image = np.array(masked_image)  # 转换为NumPy数组
    masked_image = cv2.resize(masked_image, (sample_size[1], sample_size[0]))
    masked_image = torch.from_numpy(masked_image)  # 转换为PyTorch张量
    masked_image = masked_image.unsqueeze(0) #1,h,w,c
    masked_image = masked_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

    # mask image
    mask_image = Image.open(mask_video_path).convert("RGB")
    mask_image = np.array(mask_image)  # 转换为NumPy数组
    mask_image = cv2.resize(mask_image, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_NEAREST)
    mask_image = torch.from_numpy(mask_image)  # 转换为PyTorch张量
    mask_image = mask_image.unsqueeze(0) #1,h,w,c
    mask_image = mask_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w
    mask_image[mask_image < 0.5] = 0
    mask_image[mask_image >= 0.5] = 1

     # pose image 
    pose_image = Image.open(pose_video_path).convert("RGB")
    pose_image = np.array(pose_image)  # 转换为NumPy数组
    pose_image = cv2.resize(pose_image, (sample_size[1], sample_size[0]))
    pose_image = torch.from_numpy(pose_image)  # 转换为PyTorch张量
    pose_image = pose_image.unsqueeze(0) #1,h,w,c
    pose_image = pose_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

        
    # cloth image 
    cloth_image = Image.open(ref_image).convert("RGB")
    cloth_image = np.array(cloth_image)  # 转换为NumPy数组
    cloth_image = cv2.resize(cloth_image, (sample_size[1], sample_size[0]))
    cloth_image = torch.from_numpy(cloth_image)  # 转换为PyTorch张量
    cloth_image = cloth_image.unsqueeze(0) #1,h,w,c
    cloth_image = cloth_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

    #cloth-mask image
    # 使用replace方法进行替换
    cloth_mask_path = ref_image.replace("/cloth/", "/cloth-mask/")
    cloth_mask = Image.open(cloth_mask_path).convert("RGB")
    cloth_mask = np.array(cloth_mask)  # 转换为NumPy数组
    cloth_mask = cv2.resize(cloth_mask, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_NEAREST)
    cloth_mask = torch.from_numpy(cloth_mask)  # 转换为PyTorch张量
    cloth_mask = cloth_mask.unsqueeze(0) #1,h,w,c
    cloth_mask = cloth_mask.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w
    cloth_mask[cloth_mask < 0.5] = 0
    cloth_mask[cloth_mask >= 0.5] = 1

    # line image
    line_image = Image.open(line_image_path).convert("RGB")
    line_image = np.array(line_image)  # 转换为NumPy数组
    line_image = cv2.resize(line_image, (sample_size[1], sample_size[0]))
    line_image = torch.from_numpy(line_image)  # 转换为PyTorch张量
    line_image = line_image.unsqueeze(0) #1,h,w,c
    line_image = line_image.permute([3, 0, 1, 2]).unsqueeze(0) / 255 #1,c,1,h,w

    clip_image = Image.open(ref_image).convert("RGB")


    return org_image, masked_image, mask_image, pose_image, clip_image, cloth_image, line_image, cloth_mask