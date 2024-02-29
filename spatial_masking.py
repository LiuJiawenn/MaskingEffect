import matplotlib.pyplot as plt
import numpy as np
import time
import cv2


def neighbour_list(block=9):
    # 返回邻域像素与中心像素的相对位置
    # 返回的是圆形扩张的稀疏的邻域像素
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


def bolck_masking(X, Y):
    N = X.shape[0]-1
    S = np.zeros(N+1)

    RXP = np.linalg.pinv(X.T@X / N)
    RYX = Y.T@X / N

    # print(temp.shape)
    for i in range(N+1):
        S[i] = np.abs(Y[i]-RYX@RXP@X[i].T)

    l = int(np.sqrt(N+1))
    S = S.reshape(l,l)
    return S
#

# 返回blocks的起始坐标
def blockindex(r, c, block):
    res = []
    for i in range(0, r, block):
        for j in range(0, c, block):
            if i + block <= r and j + block <= c:
                res.append([i, j])
    return res


def get_neighborhood(image, neighbour_offsets):
    rows, cols = image.shape
    pad_width = max(max(abs(dx), abs(dy)) for dx, dy in neighbour_offsets)
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    neighborhood = np.empty((rows, cols, len(neighbour_offsets)), dtype=image.dtype)

    for i, (dx, dy) in enumerate(neighbour_offsets):
        neighborhood[:, :, i] = padded_image[pad_width + dx:pad_width + dx + rows,
                                             pad_width + dy:pad_width + dy + cols]

    return neighborhood

if __name__ == '__main__':
    start_time = time.time()
    block = 9
    neighbourList = neighbour_list(block)
    img = cv2.imread('C:/Users/Jiawen/Desktop/aa/frames/31.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    neighborhood_pixels = get_neighborhood(gray, neighbourList)

    r = gray.shape[0]
    c = gray.shape[1]
    blocks = blockindex(r, c, block)
    masks = []
    count = 0
    for x, y in blocks:
        print(count)
        count+=1

        Y = gray[x:x+block,y:y+block].reshape(-1, 1)
        X = neighborhood_pixels[x:x+block,y:y+block].reshape(-1, neighborhood_pixels.shape[2])

        S = bolck_masking(X, Y)
        masks.append(S)
    #
    rows = []

    r = 1080 // block
    c = 1920 // block
    for i in range(r):
        # 每一行有 6 个图像块，水平堆叠
        row = np.hstack(masks[i * c:(i + 1) * c])
        rows.append(row)

    # 将所有行垂直堆叠起来形成最终的图像
    final_image = np.vstack(rows)
    end_time = time.time()
    print(f"运行时间：{end_time - start_time}秒")

    np.save('testData/spatialMasking9.npy', final_image)
    plt.imshow(final_image)
    plt.show()




