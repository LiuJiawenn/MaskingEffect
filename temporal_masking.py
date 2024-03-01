import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
import matplotlib.pyplot as plt

def extract_patches(arr, patch_shape=(32, 32, 3), extraction_step=(32, 32, 3)):
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

def readY(videoPath,patch_H, patch_w):
    # 读取视频前30帧返回矩阵
    cap = cv2.VideoCapture(videoPath)
    Y = []

    for i in range(31):
        ret1, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patches = extract_patches(gray.reshape(1080, 1920, 1), (patch_H, patch_w, 1), (patch_H, patch_w, 1))
        Y.append(patches)

    cap.release()
    # 36patch, 每个已经整理成了（180×320）列向量，共30个列向量
    return np.transpose(np.array(Y[:30]),(1, 2, 0)), Y[30]




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    patch_H = 90
    patch_w = 160
    # patch_H = 9
    # patch_w = 16
    Y_mat,y_mat = readY('C:/Users/Jiawen/Desktop/runningFolder/videoSRC060_1920x1080_302562_qp_00.avi', patch_H, patch_w)
    # plt.imshow(y_mat[22,:].reshape(patch_H,patch_w))
    # plt.show()
    patch_nb= Y_mat.shape[0]
    vector_len = Y_mat.shape[1]
    frame_nb = Y_mat.shape[2]
    temporal_map_patch = []
    temporal_map = np.zeros((1080,1920))

    # 只取最大的10个奇异值
    s_nb = 30
    for p in range(patch_nb):
        print(p)
        Y = Y_mat[p]
        y = y_mat[p]
        U, S, Vh = np.linalg.svd(Y, full_matrices=True, compute_uv=True)
        U_n = U[:, :s_nb]
        S_n = np.diag(S[:s_nb])
        Vh_n = Vh[:s_nb, :]

        xkl = np.dot(S_n, Vh_n)
        xl = xkl[:, -1]
        x_kplus1_l = xkl[:, 1:]
        x_k_lminus1 = xkl[:, 0:-1]

        al=x_kplus1_l@np.linalg.pinv(x_k_lminus1)
        cl = U_n

        RT_l_plus_1 = np.abs(y-cl@al@xl)
        RT_l_plus_1 = RT_l_plus_1.reshape(patch_H, patch_w)
        temporal_map_patch.append(RT_l_plus_1)

    rows = []
    r = 1080//patch_H
    c = 1920//patch_w
    for i in range(r):
        # 每一行有 6 个图像块，水平堆叠
        row = np.hstack(temporal_map_patch[i * c:(i + 1) * c])
        rows.append(row)

    # 将所有行垂直堆叠起来形成最终的图像
    final_image = np.vstack(rows)
    np.save('testData/final_image180.npy', final_image)
    plt.imshow(final_image)
    plt.show()


def block_masking(Y,y):
    U, S, Vh = np.linalg.svd(Y, full_matrices=True, compute_uv=True)
    U_n = U[:, :s_nb]
    S_n = np.diag(S[:s_nb])
    Vh_n = Vh[:s_nb, :]

    xkl = np.dot(S_n, Vh_n)
    xl = xkl[:, -1]
    x_kplus1_l = xkl[:, 1:]
    x_k_lminus1 = xkl[:, 0:-1]

    al = x_kplus1_l @ np.linalg.pinv(x_k_lminus1)
    cl = U_n

    RT_l_plus_1 = np.abs(y - cl @ al @ xl)
    RT_l_plus_1 = RT_l_plus_1.reshape(patch_H, patch_w)
    return RT_l_plus_1


def temporal_masking_effect(gray_video_matrics, patch_h, patch_w, patch_d):
    frame_nb = len(gray_video_matrics)
    # 先按照patch_d在矩阵前面补充帧
    first_d_frames = gray_video_matrics[:, :, 0:patch_d]
    first_d_frames = first_d_frames[:, :, -1] # 倒序
    gray_video_matrics = first_d_framse + gray_video_matrics


    w = len(gray_video_matrics[0])
    h = len(gray_video_matrics[0][0])

    restored_masks = np.zeros((150, 1080, 1920))
    # 设置patch大小的窗口，按照行，列, 深度移动
    for w_d in range(0, frame_nb):
        for w_h in range(0, h, patch_h):
            for w_w in range(0, w, patch_w):
                # 窗口内整理成Y， 窗口后面的一帧整理成y
                Y = np.array(gray_video_matrics[w_d:w_d+patch_d, w_h:w_h+patch_h, w_w:w_w+patch_w]).reshape(patch_w*patch_h,patch_d)
                y = np.array(gray_video_matrics[w_d+patch_d, w_h:w_h+patch_h, w_w:w_w+patch_w]).reshape(patch_w*patch_h)
                # 根据Y，y计算时间掩蔽
                block_mask = block_masking(Y, y)
                # 暂存掩蔽响应
                masks[w_d, w_h:w_h+patch_h, w_w:w_w+patch_w] = block_mask

    # 返回响应矩阵
    return restored_masks



