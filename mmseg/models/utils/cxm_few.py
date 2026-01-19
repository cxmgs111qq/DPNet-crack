import torch
import torch.nn.functional as F
import torch.nn as nn
from .cxm_allSegFew import masked_average_pooling_km
'''crack 2分类的 fewshot 功能库'''


def sample():
    # 示例用法
    input_tensor = torch.randn(1, 3, 4, 4)  # 示例输入张量
    mask = torch.randint(0, 2, (1, 1, 4, 4))  # 示例掩码张量

    pooled_output = masked_average_pooling_with_background2(input_tensor, mask)
    print(pooled_output)
    print(pooled_output.size())

class WeightedSum(nn.Module):
    def __init__(self,wl=None,ws=None):
        super().__init__()
        # 初始化两个可训练的权重参数
        if wl is None:
            self.wl = nn.Parameter(torch.tensor(0.5))
        else:
            self.wl=wl

        if ws is None:
            self.ws = nn.Parameter(torch.tensor(0.5))
        else:
            self.ws = ws

    def forward(self, class_embeds, support_propty):
        # 计算加权和
        weighted_sum = self.wl * class_embeds +self.ws* support_propty
        # weighted_sum = 0.5 * class_embeds + 0.5 * support_propty
        return weighted_sum
class AttentionFusion(nn.Module):
    def __init__(self, input_dim=512):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, class_embeds, support_propty):
        combined = torch.cat((class_embeds, support_propty), dim=1)
        attention_weights = self.attention(combined)
        fused_tensor = attention_weights * class_embeds + (1 - attention_weights) * support_propty
        return fused_tensor
def qkv(tensor_k,tensor_q):#qk为正 kq为反
    attention_scores = torch.matmul(tensor_q, tensor_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(512.0))

    # 应用 softmax 函数
    attention_weights = F.softmax(attention_scores, dim=-1)

    # 计算输出
    output_tensor = torch.matmul(attention_weights, tensor_k)
    return output_tensor
def masked_average_pooling_with_background(input_tensor, mask):
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

    # 将掩码进行 one-hot 编码，扩展到 (N, 2, H, W)
    mask = mask.squeeze(1).long()  # 去掉通道维度并转换为 long 类型
    one_hot_mask = F.one_hot(mask, num_classes=2).permute(0, 3, 1, 2).float()  # 进行 one-hot 编码并调整维度

    # 前景掩码和背景掩码
    foreground_mask = one_hot_mask[:, 1:2, :, :]
    background_mask = one_hot_mask[:, 0:1, :, :]

    # 计算前景的 masked average pooling
    masked_input_foreground = input_tensor * foreground_mask
    mask_sum_foreground = foreground_mask.sum(dim=(2, 3), keepdim=True)
    pooled_foreground = masked_input_foreground.sum(dim=(2, 3), keepdim=True) / mask_sum_foreground

    # 计算背景的 masked average pooling
    masked_input_background = input_tensor * background_mask
    mask_sum_background = background_mask.sum(dim=(2, 3), keepdim=True)
    pooled_background = masked_input_background.sum(dim=(2, 3), keepdim=True) / mask_sum_background

    # 合并前景和背景的池化结果
    pooled_output = torch.cat((pooled_foreground, pooled_background), dim=0)

    return pooled_output

#多一个padding类
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
def fewV22(support_f,support_gt,mask_pred):
    scaler=20
    # support_propty=masked_average_pooling_with_background2(support_f,support_gt) #3*512
    support_propty=masked_average_pooling_km(support_f,support_gt,num_classes=2) #3*100*1*1
    support_propty=support_propty[:-1]#去除padding的原型 2*100*1*1
    dist = F.cosine_similarity(support_propty,mask_pred, dim=1) * scaler #根据原型和特征的相似度预测，相似度即概率
    return dist #cls*h*w

def fewV21(fts, prototype):
    prototype=prototype.permute(2,1,0)[...,None]
    dist=calDist(fts,prototype)
    return dist
def calDist(fts, prototype, scaler=20):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: N x C x H x W
        prototype: prototype of one semantic class
            expect shape: 1 x C
    """
    dist = F.cosine_similarity(fts, prototype, dim=1) * scaler
    return dist




if __name__ == '__main__':
    # tensor1 = torch.randn(5, 3)  # 形状为 b*c 的张量
    # tensor2 = torch.randn(5, 3)  # 形状为 b*c 的张量
    #
    # model = WeightedSum()
    # output = model(tensor1, tensor2)
    # print(output)
    sample()
