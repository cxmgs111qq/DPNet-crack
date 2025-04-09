'''结合边缘和PW，用边缘代替公式前面的1'''
import torch
import torch.nn.functional as F
import math
import numpy as np
import time
import matplotlib.pyplot as plt
def mainEdgePW(input=torch.randint(0, 3, (1, 400, 400)), num_classes=3, perc=1/100,device='cpu'):
    b, H, W = input.size()
    zipp = 4 if W < 512 else 8  # 压缩等级
    t1 = time.time()
    input = input.unsqueeze(1).float()  # b*h*w->b*1*h*w,缩放必须.必须float
    input = F.interpolate(input, size=(H // zipp, W // zipp), mode='nearest')  # 先压缩尺寸
    input = input.squeeze(1).long()  # b*1*h*w->b*h*w
    one_hot_tensor = F.one_hot(input, num_classes=num_classes).permute(0, 3, 1, 2).float().to(device)
    KSIZE = max(5, 2*int((input.shape[1] + input.shape[2]) * perc)+1)  # 经验尺寸
    kernel = torch.ones(num_classes, 1, KSIZE, KSIZE, dtype=torch.float32).to(device)  # 全是1的卷积核
    count_tensor = F.conv2d(one_hot_tensor, kernel, padding=KSIZE // 2, groups=num_classes)  # 统计矩阵 b*cls*h*w

    # 矢量化操作
    edge=laplace_edge_detection(input)
    center = input  # 原矩阵b*h*w
    k = torch.count_nonzero(count_tensor, dim=1)  # 计算count内非0数量，b*h*w
    a = torch.sum(count_tensor[:, 1:], dim=1)  # 计算除背景外总和,0是背景类，b*h*w
    d = KSIZE
    n = count_tensor.gather(1, center.unsqueeze(1)).squeeze(1)  # center类标签数量，b*h*w
    # weight = edge + 1.5 ** torch.log(k) * a / (d ** 2 + n)#实验版
    weight = edge + 1.5 * a / (d ** 2 + n) #paper后续测试版
    PW = weight

    PW = PW.unsqueeze(1).float()
    PW = F.interpolate(PW, size=(H, W), mode='bilinear')  # 还原尺寸
    PW = PW.squeeze(1)
    t2 = time.time()
    print(t2 - t1)
    show(edge[0].numpy(),PW[0].numpy())
    return PW
def show(edge,pw):
    plt.subplot(2, 1, 1)  # nama,file
    plt.imshow(edge,cmap='YlOrBr', interpolation='nearest')
    plt.xticks([])  # 去x坐标刻度
    plt.yticks([])
    plt.axis('off')
    plt.subplot(2, 1, 2)  # nama,file
    plt.imshow(pw,cmap='YlOrBr', interpolation='nearest')
    plt.xticks([])  # 去x坐标刻度
    plt.yticks([])
    plt.axis('off')
    plt.show()
    # 显示图像


def laplace_edge_detection(ground_truth):
    # 定义拉普拉斯算子
    laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.long).unsqueeze(0).unsqueeze(0)

    # 将拉普拉斯算子应用于每个通道
    edges = F.conv2d(ground_truth.unsqueeze(1), laplace_kernel, padding=1)

    # 将边缘检测结果转换为二值图像
    edge_map = (edges != 0).float()

    return edge_map.squeeze(1)

def test():
    from PIL import Image
    img = r"W:\PythonWork\mmseg\myProj\crack500\train\labels\20160328_153645_1_361.png"
    label = Image.open(img)
    label = np.array(label, dtype=np.uint8)
    input = torch.tensor(label, dtype=torch.int64).unsqueeze(0)
    input = torch.where(input == 255, torch.tensor(0), input)
    result = mainEdgePW(input, num_classes=8)
if __name__ == "__main__":
    test()