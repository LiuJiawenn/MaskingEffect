import cupy as cp
from numpy.lib.stride_tricks import as_strided
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2


def neighbour_list(block=9):
    if block % 2 == 0:
        print("block size should be single!")
        return
    res = []
    order = (block-1)//2
    for i in range(1, order):
        if i % 2 == 1:
            res += [(-i, 0), (0, i), (i, 0), (0, -i)]
        else:
            res += [(-i, -i), (-i, i), (i, i), (i, -i)]
    return res


def find_neighbours(img, cx, cy, neighbourList):
    r = img.shape[0] - 1
    c = img.shape[1] - 1
    res = []
    for a, b in neighbourList:
        ix = cx + a
        iy = cy + b
        # 下面处理边界情况，越界像素使用图片镜像进行补充
        if ix < 0 or ix > r or iy < 0 or iy > c:
            res.append(0)
        else:
            res.append(img[ix][iy])
    return res


def neighbour_matricx(img, ix, iy, neighbourList,block = 9):
    resX = []
    resY = []
    for i in range(block):
        cx = ix+i
        for j in range(block):
            cy = iy+j
            resY.append(img[cx][cy])
            resX.append(find_neighbours(img,cx,cy,neighbourList))
    X = np.array(resX)
    Y = np.array(resY).reshape(block*block,1)
    return X, Y


def bolck_masking(X,Y):
    N = X.shape[0]-1
    S = cp.zeros(N+1)

    RXP = cp.linalg.pinv(X.T@X / N)
    RYX = Y.T@X / N

    # print(temp.shape)
    for i in range(N+1):
        S[i] = cp.abs(Y[i]-RYX@RXP@X[i].T)

    l = int(cp.sqrt(N+1))
    S = S.reshape(l,l)
    return S


# 返回blocks的起始坐标
def blockindex(r, c, block):
    res = []
    for i in range(0, r, block):
        for j in range(0, c, block):
            if i + block < r and j + block < c:
                res.append([i,j])
    return res


if __name__ == '__main__':
    start_time = time.time()
    block = 17
    img = cv2.imread('C:/Users/Jiawen/Desktop/aa/frames/31.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    r = gray.shape[0]
    c = gray.shape[1]

    neighbourList = neighbour_list(block)
    blocks = blockindex(r, c, block)
    masks = []
    count = 0
    for x, y in blocks:
        print(count)
        count+=1
        # 开始计算block的masking
        X, Y = neighbour_matricx(gray, x, y, neighbourList, block)
        S = bolck_masking(cp.array(X),cp.array(Y))
        masks.append(S)

    rows = []
    r = 1080 // block -1
    c = 1920 // block
    for i in range(r):
        # 每一行有 6 个图像块，水平堆叠
        row = np.hstack(masks[i * c:(i + 1) * c])
        rows.append(row)

    # 将所有行垂直堆叠起来形成最终的图像
    final_image = np.vstack(rows)
    end_time = time.time()
    print(f"运行时间：{end_time - start_time}秒")

    # 由于matplotlib无法直接显示CuPy数组，需要先转换为NumPy数组
    final_image_np = cp.asnumpy(final_image)
    np.save('spatialMasking_gpu.npy', final_image_np)
    plt.imshow(final_image_np, cmap='gray')
    plt.show()





