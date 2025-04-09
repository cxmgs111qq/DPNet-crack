'''多类语义分割的few shot功能库'''
import torch
import torch.nn.functional as F
import torch.nn as nn
def masked_average_pooling_with_background2(input_tensor, mask):
    """
    实现 masked average pooling，并将 mask=0 的位置视为背景类
    :param input_tensor: 输入张量，形状为 (N, C, H, W)
    :param mask: 掩码张量，形状为 (N, 1, H, W)
    :return: 池化后的张量，形状为 (2, C, 1, 1)
    """
    # 确保掩码的形状与输入张量匹配
    input_tensor = F.interpolate(input_tensor, size=mask.shape[-2:], mode='bilinear')
    assert input_tensor.shape[0] == mask.shape[0]
    assert input_tensor.shape[2:] == mask.shape[2:]

    # 将掩码中值为 255 的部分替换为 2
    mask = mask.squeeze(1).long()  # 去掉通道维度并转换为 long 类型
    mask[mask == 255] = 2  # 将 255 替换为 2

    # 将掩码进行 one-hot 编码，扩展到 (N, 3, H, W)
    one_hot_mask = F.one_hot(mask, num_classes=3).permute(0, 3, 1, 2).float()  # 进行 one-hot 编码并调整维度

    # 前景掩码和背景掩码
    foreground_mask = one_hot_mask[:, 1:2, :, :]
    background_mask = one_hot_mask[:, 0:1, :, :]
    padding_mask= one_hot_mask[:, 2, :, :]
    # 计算前景的 masked average pooling
    masked_input_foreground = input_tensor * foreground_mask
    mask_sum_foreground = foreground_mask.sum(dim=(2, 3), keepdim=True)
    pooled_foreground = masked_input_foreground.sum(dim=(2, 3), keepdim=True) / (mask_sum_foreground+ 1e-5)

    # 计算背景的 masked average pooling
    masked_input_background = input_tensor * background_mask
    mask_sum_background = background_mask.sum(dim=(2, 3), keepdim=True)
    pooled_background = masked_input_background.sum(dim=(2, 3), keepdim=True) / (mask_sum_background+ 1e-5)

    #计算padding的255
    masked_input_padding = input_tensor * padding_mask
    mask_sum_padding = padding_mask.sum(dim=(1, 2), keepdim=True)
    pooled_padding = masked_input_padding.sum(dim=(2, 3), keepdim=True) / (mask_sum_padding+ 1e-5)
    # 合并前景和背景的池化结果
    '''bug 顺序错了'''
    # pooled_output = torch.cat((pooled_foreground, pooled_background,pooled_padding), dim=0)
    pooled_output = torch.cat((pooled_background, pooled_foreground,pooled_padding), dim=0)

    return pooled_output #3*512
def masked_average_pooling_km(input_tensor, mask, num_classes): #1*100*32*44 ,1*1*240*320, 8
    # 确保掩码的形状与输入张量匹配
    input_tensor = F.interpolate(input_tensor, size=mask.shape[-2:], mode='bilinear')
    assert input_tensor.shape[0] == mask.shape[0]
    assert input_tensor.shape[2:] == mask.shape[2:]

    # 将掩码中值为 255 的部分替换为 num_classes
    mask = mask.squeeze(1).long()  # 去掉通道维度并转换为 long 类型
    mask[mask == 255] = num_classes  # 将 255 替换为 num_classes,最后一维

    # 将掩码进行 one-hot 编码，扩展到 (N, num_classes+1, H, W)
    one_hot_mask = F.one_hot(mask, num_classes=num_classes+1).permute(0, 3, 1, 2).float()  # 进行 one-hot 编码并调整维度

    # 初始化一个列表来存储每个类别的池化结果
    pooled_outputs = []

    for i in range(num_classes):
        # 当前类别的掩码
        current_mask = one_hot_mask[:, i:i+1, :, :]

        # 计算当前类别的 masked average pooling
        masked_input = input_tensor * current_mask
        mask_sum = current_mask.sum(dim=(2, 3), keepdim=True)
        pooled = masked_input.sum(dim=(2, 3), keepdim=True) / (mask_sum + 1e-5)

        pooled_outputs.append(pooled)

    # 合并所有类别的池化结果
    pooled_output = torch.cat(pooled_outputs, dim=0)
    return pooled_output

class GateSum(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.linear_layer=nn.Conv2d(input, output, kernel_size=1)

    def forward(self, class_embeds, support_propty):
        gate_input = torch.cat([class_embeds, support_propty], dim=1)
        gate = torch.sigmoid(self.linear_layer(gate_input)) #b*2*h*w  和propotype尺寸一样，每个元素都有概率
        updated_prototype = gate * class_embeds + (1 - gate) * support_propty
        return updated_prototype

#直接相加的方式
def getSupPred(support_f,support_gt,mask_pred,num_classes=2):
    scaler = 20
    support_propty = masked_average_pooling_km(support_f, support_gt, num_classes=num_classes)  # 3*100*1*1
    support_propty = support_propty # 去除padding的原型 2*100*1*1
    dist = F.cosine_similarity(support_propty, mask_pred, dim=1) * scaler  # 根据原型和特征的相似度预测，相似度即概率
    return dist  # cls*h*w
def sample():
    # 示例用法
    input_tensor = torch.randn(1,2, 4, 4)  # 示例输入张量
    mask = torch.randint(0, 2, (1, 1, 4, 4))  # 示例掩码张量

    pooled_output2 = masked_average_pooling_km(input_tensor, mask,num_classes=2)
    pooled_output= masked_average_pooling_with_background2(input_tensor, mask)
    print(pooled_output,pooled_output2)
    print(pooled_output.size())
if __name__ == '__main__':
    # # 输入特征图 [2, 512, 16, 16]，2个样本
    # features = torch.randn(1, 512, 50, 50)
    #
    # # 对应掩码 [2, 1, 32, 32]，5个类别
    # masks = torch.randint(0, 5, (1, 1, 50, 50))
    # masks[:,:,2,2]=255
    #
    # # 生成原型
    # prototypes = masked_average_pooling_km(features, masks, num_classes=5)
    #
    # # 输出形状 [2*5, 512] = [10, 512]
    # print(prototypes.shape)
    sample()