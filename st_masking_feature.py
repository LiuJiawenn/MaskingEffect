# 本脚本将VideoSet 220个无损视频 转化为 11×11×9 的一维特征向量
# 时间空间特征各 输出一个特征向量
import os
import ffmpeg
import cv2

def h264_to_avi(path_264, path_avi):
    # print("s_264  ", path_264)
    # print("s_avi: ", path_avi)
    if os.path.exists(path_avi):
        return

    stream = ffmpeg.input(path_264)
    stream = ffmpeg.output(stream, path_avi, vcodec='rawvideo')
    ffmpeg.run(stream)


def spatial_masking_effect(image,block):
    pass


def temporal_masking_effect(image, patch_h, patch_w, patch_d):
    pass


# path
raw_video_dir = ''  # 存放所有原视频.264文件的文件夹
feature_dir = ''  # 用于保存所有特征的文件夹

# 读取视频文件名
video_list = os.listdir(raw_video_dir)
# 对于每一个原视频，计算时空掩蔽特征并保存
for video in video_list:
    print('指望在计算视频'+str(video))
    # video 是视频名字

    # step 1： 获取解压后的avi视频
    s_264 = ''  # 264文件路径
    s_avi = ''  # 转换为avi视频后的存储路径
    h264_to_avi(s_264, s_avi)

    # step 2: 拆分成帧，保存在列表中，每个帧计算空间掩蔽相应
    frames = []  # 用于保存拆分出来的帧
    spatial_masking_frames = []  # 用于保存空间掩蔽响应

    cap = cv2.VideoCapture(s_avi)
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        spatial_masking_frames.append(spatial_masking_effect(gray, block))
    cap.release()

    # step 3: 使用frames 矩阵计算时间掩蔽效应矩阵
    temporal_masking_frames = temporal_masking_effect(frames,patch_h, patch_w, patch_d)

    # step 4: 将时空特征响应矩阵分成180*320*30的,每个patch计算均值，形成11×11×9 的一维特征向量

    # step 5：保存向量







