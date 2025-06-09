import cv2
import os

def video_to_images(video_path, output_dir):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 构建帧文件名
        frame_filename = os.path.join(output_dir, f"{frame_idx:04d}.png")
        
        # 保存图像
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"视频处理完成，共保存了 {frame_idx} 帧图像到 {output_dir}")

# 示例调用
if __name__ == "__main__": 
    video_to_images("datasets/person/customize/video/00001/video.mp4", 
                    "datasets/person/customize/video/00001/images")
    