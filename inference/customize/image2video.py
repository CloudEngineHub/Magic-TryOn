import cv2
import os

def images_to_video(image_folder, output_video_path, fps=30, i=None):
    # 获取文件夹中的所有图像文件并排序
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])

    # 检查文件夹是否为空
    if not image_files:
        print("文件夹中没有图像文件")
        return

    # 只保留最后 i 张图像
    if i is not None:
        image_files = image_files[-i:]

    # 获取第一张图像的大小
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 视频写入器初始化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 读取图像并写入视频
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"无法读取图像: {image_path}")
            continue

        out.write(img)

    out.release()
    print(f"视频保存为 {output_video_path}")

# 示例调用
if __name__ == "__main__":
    seq_name = '00001'
    base_path = f'datasets/person/customize/video/{seq_name}'
    last_i = None  # 只处理最后i张图像

    images_to_video(f'{base_path}/images', f'{base_path}/video.mp4', fps=30, i=last_i)
    images_to_video(f'{base_path}/agnostic', f'{base_path}/agnostic.mp4', fps=30, i=last_i)
    images_to_video(f'{base_path}/mask', f'{base_path}/mask.mp4', fps=30, i=last_i)
    images_to_video(f'{base_path}/image-densepose', f'{base_path}/densepose.mp4', fps=30, i=last_i)
