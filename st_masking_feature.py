# 本脚本将VideoSet 220个无损视频 转化为 11×11×9 的一维特征向量
# 时间空间特征各 输出一个特征向量
import os
import ffmpeg
import cv2
from spatial_masking import spatial_masking_effect


def h264_to_avi(path_264, path_avi):
    # print("s_264  ", path_264)
    # print("s_avi: ", path_avi)
    if os.path.exists(path_avi):
        return

    stream = ffmpeg.input(path_264)
    stream = ffmpeg.output(stream, path_avi, vcodec='rawvideo')
    ffmpeg.run(stream)

def extract_patches(arr, patch_shape=(180, 320, 30), extraction_step=(90, 160, 15)):
    # input (1080,1920,3)
    # 对应维度+1需要跳过的字节数
    patch_strides = arr.strides
    # 四个维度，每个维度创建一个切片
    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    # patch_indices_shape = (10,33,60,1)
    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    patches = patches.reshape((-1, patch_shape[0]* patch_shape[1]))
    return patches


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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        spatial_masking_frames.append(spatial_masking_effect(gray, block))
    cap.release()

    # step 3: 使用frames 矩阵计算时间掩蔽效应矩阵
    temporal_masking_frames = temporal_masking_effect(frames,patch_h, patch_w, patch_d)

    # step 4: 将时空特征响应矩阵分成180*320*30的,每个patch计算均值，形成11×11×9 的一维特征向量
    s_patches = extract_patches(np.array(spatial_masking_frames))
    t_patches = extract_patches(np.array(temporal_masking_frames))

    s_feature = np.mean(np.array(s_patches), axis=(1, 2, 3)) # 11×11×9
    t_feature = np.mean(np.array(t_patches), axis=(1, 2, 3))

    # step 5：保存向量
    feature_path = feature_dir+ video+'/'+video+'.npy'
    # if os.path.exist:
    # mkdir('')
    # 跑一个存一个，这样如果中间断了，之前跑过的视频还能留着
    np.save(feature_path, [s_feature, t_feature])







