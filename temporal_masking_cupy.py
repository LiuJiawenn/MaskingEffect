import cupy as cp
from numpy.lib.stride_tricks import as_strided
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


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


if __name__ == '__main__':
    start_time = time.time()
    patch_H = 90
    patch_w = 160
    s_nb = 30
    Y_mat, y_mat = readY('C:/Users/Jiawen/Desktop/runningFolder/videoSRC023_1920x1080_30_qp_00.avi', patch_H, patch_w)
    Y_mat = cp.array(Y_mat)
    y_mat = cp.array(y_mat)
    patch_nb = Y_mat.shape[0]
    vector_len = Y_mat.shape[1]
    frame_nb = Y_mat.shape[2]
    temporal_map_patch = []
    temporal_map = cp.zeros((1080, 1920))

    for p in range(patch_nb):
        # print(p)
        Y = Y_mat[p]
        y = y_mat[p]
        U, S, Vh = cp.linalg.svd(Y, full_matrices=True, compute_uv=True)
        U_n = U[:, :s_nb]
        S_n = cp.diag(S[:s_nb])
        Vh_n = Vh[:s_nb, :]

        xkl = cp.dot(S_n, Vh_n)
        xl = xkl[:, -1]
        x_kplus1_l = xkl[:, 1:]
        x_k_lminus1 = xkl[:, 0:-1]

        al = cp.dot(x_kplus1_l, cp.linalg.pinv(x_k_lminus1))
        cl = U_n

        RT_l_plus_1 = np.abs(y - cl @ al @ xl)
        RT_l_plus_1 = RT_l_plus_1.reshape(patch_H, patch_w)
        temporal_map_patch.append(RT_l_plus_1)

    rows = []
    r = 1080 // patch_H -1
    c = 1920 // patch_w
    for i in range(r):
        # 每一行有 6 个图像块，水平堆叠
        row = np.hstack(temporal_map_patch[i * c:(i + 1) * c])
        rows.append(row)

    final_image = cp.vstack(rows)
    end_time = time.time()
    print(f"运行时间：{end_time - start_time}秒")

    # 由于matplotlib无法直接显示CuPy数组，需要先转换为NumPy数组
    final_image_np = cp.asnumpy(final_image)
    np.save('testData/temporalMasking.npy', final_image_np)
    cm2 = plt.cm.get_cmap('jet')
    plt.imshow(final_image_np, cmap=cm2)
    plt.show()