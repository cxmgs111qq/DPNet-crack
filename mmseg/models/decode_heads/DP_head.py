# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.ops import point_sample
from mmengine.dist import all_reduce
from mmengine.model.weight_init import (caffe2_xavier_init, normal_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmseg.models.backbones.vit import TransformerEncoderLayer

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, MatchMasks, SampleList,
                         seg_data_to_instance_data)
from mmseg.utils import (MLP, LayerNorm2d, PatchEmbed, cross_attn_layer,
                         get_uncertain_point_coords_with_randomness, resize)
from .decode_head import BaseDecodeHead
from mmseg.models.utils.cxm_few import WeightedSum
from mmseg.models.utils.cxm_allSegFew import GateSum,masked_average_pooling_km
#dpnet发布版
class EAM(nn.Module):
    def __init__(self, token, pathch):
        super().__init__()
        self.tokenMod= nn.Sequential(
            nn.Linear(token, token, bias=False),  # 从 c -> c/r
            nn.ReLU())
        self.patchMod= nn.Sequential(
            nn.Linear(2, 1, bias=False),  # 从 c -> c/r
            nn.ReLU())
        self.tokenMod2= nn.Sequential(
            nn.Linear(1408, 1408, bias=False),  # 从 c -> c/r
            nn.ReLU()
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):        #b*tok*patch

        PatchAvg=torch.mean(x,dim=1,keepdim=True) #patch均值
        PatchMax,index=torch.max(x,dim=1,keepdim=True) #patch均值
        out = torch.cat((PatchAvg, PatchMax), dim=1)
        out=out.permute(0, 2, 1)
        patchatt=self.sigmoid(self.patchMod(out))#patch注意力
        patchatt = patchatt.permute(0, 2, 1)
        out=patchatt*x
        return out
class MLPMaskDecoder(nn.Module):
    """Module for decoding query and visual features with MLP layers to
    generate the attention biases and the mask proposals."""

    def __init__(
        self,
        *,
        in_channels: int,
        total_heads: int = 1,
        total_layers: int = 1,
        embed_channels: int = 256,
        mlp_channels: int = 256,
        mlp_num_layers: int = 3,
        rescale_attn_bias: bool = False,
    ):
        super().__init__()
        self.total_heads = total_heads
        self.total_layers = total_layers

        dense_affine_func = partial(nn.Conv2d, kernel_size=1)
        # Query Branch
        self.query_mlp = MLP(in_channels, mlp_channels, embed_channels,
                             mlp_num_layers)
        # Pixel Branch
        self.pix_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )
        # Attention Bias Branch
        self.attn_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels * self.total_heads * self.total_layers,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )
        if rescale_attn_bias:
            self.bias_scaling = nn.Linear(1, 1)
        else:
            self.bias_scaling = nn.Identity()

    def forward(self, query: torch.Tensor,
                x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward function.
        Args:
            query (Tensor): Query Tokens [B,N,C].
            x (Tensor): Visual features [B,C,H,W]

        Return:
            mask_preds (Tensor): Mask proposals.
            attn_bias (List[Tensor]): List of attention bias.
        """
        query = self.query_mlp(query)
        pix = self.pix_mlp(x)
        b, c, h, w = pix.shape
        # preidict mask
        mask_preds = torch.einsum('bqc,bchw->bqhw', query, pix)
        # generate attn bias
        attn = self.attn_mlp(x)
        attn = attn.reshape(b, self.total_layers, self.total_heads, c, h, w)
        attn_bias = torch.einsum('bqc,blnchw->blnqhw', query, attn)
        attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1)
        attn_bias = attn_bias.chunk(self.total_layers, dim=1)
        attn_bias = [attn.squeeze(1) for attn in attn_bias]
        return mask_preds, attn_bias


#SAN完整结构
class SideAdapterNetwork(nn.Module):
    """Side Adapter Network for predicting mask proposals and attention bias.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        clip_channels (int): Number of channels of visual features.
            Default: 768.
        embed_dims (int): embedding dimension. Default: 240.
        patch_size (int): The patch size. Default: 16.
        patch_bias (bool): Whether use bias in patch embedding.
            Default: True.
        num_queries (int): Number of queries for mask proposals.
            Default: 100.
        fusion_index (List[int]): The layer number of the encode
            transformer to fuse with the CLIP feature.
            Default: [0, 1, 2, 3].
        cfg_encoder (ConfigType): Configs for the encode layers.
        cfg_decoder (ConfigType): Configs for the decode layers.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
    """

    def __init__(
            self,
            in_channels: int = 3,
            clip_channels: int = 768,
            embed_dims: int = 240,
            patch_size: int = 16,
            patch_bias: bool = True,
            num_queries: int = 100,
            fusion_index: list = [0, 1, 2, 3],
            cfg_encoder: ConfigType = ...,
            cfg_decoder: ConfigType = ...,
            norm_cfg: dict = dict(type='LN'),
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(  #就是trans里的patchembed，这里的作用是？
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            input_size=(640, 640),
            bias=patch_bias,
            norm_cfg=None,
            init_cfg=None,
        )
        ori_h, ori_w = self.patch_embed.init_out_size #640/patchsize
        num_patches = ori_h * ori_w #图像分成的path数
        self.pos_embed = nn.Parameter( #1*1600*240
            torch.randn(1, num_patches, embed_dims) * .02)
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, num_queries, embed_dims))
        self.query_embed = nn.Parameter(
            torch.zeros(1, num_queries, embed_dims))
        encode_layers = []
        for i in range(cfg_encoder.num_encode_layer): #encode 层数，每一层都是一个transformer，参考san图3，默认8层比图中多
            encode_layers.append( 
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=cfg_encoder.num_heads, #默认6头
                    feedforward_channels=cfg_encoder.mlp_ratio * embed_dims,
                    norm_cfg=norm_cfg))
        self.encode_layers = nn.ModuleList(encode_layers)
        #新增注意力,即 PAM
        self.EAM=EAM(1856,240)

        conv_clips = []
        for i in range(len(fusion_index)): #跟clip结合的层数？
            conv_clips.append( #4个从768到240的1*1卷积，用于接受clip信息？图里确实是4个
                nn.Sequential(
                    LayerNorm2d(clip_channels),
                    ConvModule(
                        clip_channels,
                        embed_dims,
                        kernel_size=1,
                        norm_cfg=None,
                        act_cfg=None)))
        self.conv_clips = nn.ModuleList(conv_clips)
        self.fusion_index = fusion_index
        self.mask_decoder = MLPMaskDecoder(
            in_channels=embed_dims,
            total_heads=cfg_decoder.num_heads,
            total_layers=cfg_decoder.num_layers,
            embed_channels=cfg_decoder.embed_channels,
            mlp_channels=cfg_decoder.mlp_channels,
            mlp_num_layers=cfg_decoder.num_mlp,
            rescale_attn_bias=cfg_decoder.rescale)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        for i in range(len(self.conv_clips)):
            caffe2_xavier_init(self.conv_clips[i][1].conv)

    def fuse_clip(self, fused_index: int, x: torch.Tensor,
                  clip_feature: torch.Tensor, hwshape: Tuple[int,
                                                             int], L: int):
        """Fuse CLIP feature and visual tokens."""
        fused_clip = (resize(
            self.conv_clips[fused_index](clip_feature.contiguous()),
            size=hwshape,
            mode='bilinear',
            align_corners=False)).permute(0, 2, 3, 1).reshape(x[:, -L:,
                                                                ...].shape)
        x = torch.cat([x[:, :-L, ...], x[:, -L:, ...] + fused_clip], dim=1)
        return x

    def encode_feature(self, image: torch.Tensor,
                       clip_features: List[torch.Tensor],
                       deep_supervision_idxs: List[int]) -> List[List]:
        """Encode images by a lightweight vision transformer."""
        assert len(self.fusion_index) == len(clip_features)
        x, hwshape = self.patch_embed(image) #划分成patch token格式 b4*p1600*ed240
        # print(x.size())
        ori_h, ori_w = self.patch_embed.init_out_size
        pos_embed = self.pos_embed #1*1600*240
        if self.pos_embed.shape[1] != x.shape[1]:
            # resize the position embedding
            pos_embed = (
                resize(
                    self.pos_embed.reshape(1, ori_h, ori_w,
                                           -1).permute(0, 3, 1, 2),
                    size=hwshape,
                    mode='bicubic',
                    align_corners=False,
                ).flatten(2).permute(0, 2, 1))
        pos_embed = torch.cat([
            self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed
        ],
                              dim=1) #1*100*240+1*1600*240 = 1*1700*240
        x = torch.cat([self.query_embed.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + pos_embed
        L = hwshape[0] * hwshape[1]
        fused_index = 0
        if self.fusion_index[fused_index] == 0:
            x = self.fuse_clip(fused_index, x, clip_features[0][0], hwshape, L)
            fused_index += 1
        outs = []
        for index, block in enumerate(self.encode_layers, start=1):
            x = block(x)
            # print(x[:, -L:, ...].size())
            if index < len(self.fusion_index
                           ) and index == self.fusion_index[fused_index]:
                x = self.fuse_clip(fused_index, x,
                                   clip_features[fused_index][0], hwshape, L)
                fused_index += 1

            if index in deep_supervision_idxs or index == len(
                    self.encode_layers):
                x_query = x[:, :-L, ...]  #

                x_feat2 = self.EAM(x[:, -L:, ...]).permute(0, 2, 1) \
                    .reshape(x.shape[0], x.shape[-1], hwshape[0], hwshape[1])
                outs.append({'query': x_query, 'x': x_feat2})

            if index < len(self.encode_layers):
                x = x + pos_embed
        return outs

    def decode_feature(self, features):
        mask_embeds = []
        attn_biases = []
        for feature in features:
            mask_embed, attn_bias = self.mask_decoder(**feature)
            mask_embeds.append(mask_embed)
            attn_biases.append(attn_bias)
        return mask_embeds, attn_biases

    def forward(
        self, image: torch.Tensor, clip_features: List[torch.Tensor],
        deep_supervision_idxs: List[int]
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward function."""
        features = self.encode_feature(image, clip_features,
                                       deep_supervision_idxs) #一个轻量级的vit
        mask_embeds, attn_biases = self.decode_feature(features)
        return mask_embeds, attn_biases


class RecWithAttnbias(nn.Module):
    """Mask recognition module by applying the attention biases to rest deeper
    CLIP layers.

    Args:
        sos_token_format (str): The format of sos token. It should be
            chosen from  ["cls_token", "learnable_token", "pos_embedding"].
            Default: 'cls_token'.
        sos_token_num (int): Number of sos token. It should be equal to
            the number of quries. Default: 100.
        num_layers (int): Number of rest CLIP layers for mask recognition.
            Default: 3.
        cross_attn (bool): Whether use cross attention to update sos token.
            Default: False.
        embed_dims (int): The feature dimension of CLIP layers.
            Default: 768.
        num_heads (int): Parallel attention heads of CLIP layers.
            Default: 768.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Whether to use bias in multihead-attention.
            Default: True.
        out_dims (int): Number of channels of the output mask proposals.
            It should be equal to the out_dims of text_encoder.
            Default: 512.
        final_norm (True): Whether use norm layer for sos token.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        frozen_exclude (List): List of parameters that are not to be frozen.
    """

    def __init__(self,
                 sos_token_format: str = 'cls_token',
                 sos_token_num: int = 100,
                 num_layers: int = 3,
                 cross_attn: bool = False,
                 embed_dims: int = 768,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 out_dims: int = 512,
                 final_norm: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 frozen_exclude: List = []):
        super().__init__()

        assert sos_token_format in [
            'cls_token', 'learnable_token', 'pos_embedding'
        ]
        self.sos_token_format = sos_token_format
        self.sos_token_num = sos_token_num
        self.frozen_exclude = frozen_exclude
        self.cross_attn = cross_attn
        self.num_layers = num_layers
        self.num_heads = num_heads
        if sos_token_format in ['learnable_token', 'pos_embedding']: #cls不走这里
            self.sos_token = nn.Parameter(
                torch.randn(sos_token_num, 1, self.proj.shape[0]))
            self.frozen.append('sos_token')

        layers = []
        for i in range(num_layers): # 3层transformer？
            layers.append(
                BaseTransformerLayer(
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=embed_dims, #768
                        num_heads=num_heads, #12
                        batch_first=False,
                        bias=qkv_bias),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=mlp_ratio * embed_dims,
                        act_cfg=act_cfg),
                    operation_order=('norm', 'self_attn', 'norm', 'ffn')))
        self.layers = nn.ModuleList(layers)

        self.ln_post = build_norm_layer(norm_cfg, embed_dims)[1]
        self.proj = nn.Linear(embed_dims, out_dims, bias=False) #768到512

        self.final_norm = final_norm
        self._freeze()

    def init_weights(self, rec_state_dict):
        if hasattr(self, 'sos_token'):
            normal_init(self.sos_token, std=0.02)
        if rec_state_dict is not None:
            load_state_dict(self, rec_state_dict, strict=False, logger=None)
        else:
            super().init_weights()

    def _freeze(self):
        if 'all' in self.frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in self.frozen_exclude]):
                param.requires_grad = False

    def _build_attn_biases(self, attn_biases, target_shape):
        formatted_attn_biases = []
        for attn_bias in attn_biases:
            # convert it to proper format: N*num_head,L,L
            # attn_bias: [N, num_head/1, num_sos,H,W]
            n, num_head, num_sos, h, w = attn_bias.shape
            # reshape and downsample
            attn_bias = F.adaptive_max_pool2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),
                output_size=target_shape)
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)

            true_num_head = self.num_heads
            assert (num_head == 1 or num_head
                    == true_num_head), f'num_head={num_head} is not supported.'
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)
            L = attn_bias.shape[-1]
            if self.cross_attn:
                # [n*num_head, num_sos, L]
                formatted_attn_biases.append(attn_bias)
            else:
                # [n*num_head, num_sos+1+L, num_sos+1+L]
                new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L,
                                                    num_sos + 1 + L)
                new_attn_bias[:, :num_sos] = -100
                new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0
                new_attn_bias[:num_sos, num_sos] = -100
                new_attn_bias = (
                    new_attn_bias[None, ...].expand(n * true_num_head, -1,
                                                    -1).clone())
                new_attn_bias[..., :num_sos, -L:] = attn_bias
                formatted_attn_biases.append(new_attn_bias)

        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [
                formatted_attn_biases[0] for _ in range(self.num_layers)
            ]
        return formatted_attn_biases

    def forward(self, bias: List[Tensor], feature: List[Tensor]):
        """Forward function to recognize the category of masks
        Args:
            bias (List[Tensor]): Attention bias for transformer layers
            feature (List[Tensor]): Output of the image encoder,
            including cls_token and img_feature.
        """
        cls_token = feature[1].unsqueeze(0)
        img_feature = feature[0]
        b, c, h, w = img_feature.shape
        # construct clip shadow features
        x = torch.cat(
            [cls_token,
             img_feature.reshape(b, c, -1).permute(2, 0, 1)])

        # construct sos token
        if self.sos_token_format == 'cls_token':
            sos_token = cls_token.repeat(self.sos_token_num, 1, 1)
        elif self.sos_token_format == 'learnable_token':
            sos_token = self.sos_token.expand(-1, b, -1)
        elif self.sos_token_format == 'pos_embedding':
            sos_token = self.sos_token.expand(-1, b, -1) + cls_token

        # construct attn bias
        attn_biases = self._build_attn_biases(bias, target_shape=(h, w))

        if self.cross_attn:
            for i, block in enumerate(self.layers):
                if self.cross_attn:
                    sos_token = cross_attn_layer(
                        block,
                        sos_token,
                        x[1:, ],
                        attn_biases[i],
                    )
                    if i < len(self.layers) - 1:
                        x = block(x)
        else:
            x = torch.cat([sos_token, x], dim=0)
            for i, block in enumerate(self.layers):
                x = block(x, attn_masks=[attn_biases[i]])
            sos_token = x[:self.sos_token_num]

        sos_token = sos_token.permute(1, 0, 2)  # LND -> NLD
        sos_token = self.ln_post(sos_token)
        sos_token = self.proj(sos_token)
        if self.final_norm:
            sos_token = F.normalize(sos_token, dim=-1)
        return sos_token


@MODELS.register_module()
class DPHead(BaseDecodeHead):
    """Side Adapter Network (SAN) for open-vocabulary semantic segmentation
    with pre-trained vision-language model.

    This decode head is the implementation of `Side Adapter Network
    for Open-Vocabulary Semantic Segmentation`
    <https://arxiv.org/abs/2302.12242>.
    Modified from https://github.com/MendelXu/SAN/blob/main/san/model/side_adapter/side_adapter.py # noqa:E501
    Copyright (c) 2023 MendelXu.
    Licensed under the MIT License

    Args:
        num_classes (int): the number of classes.
        san_cfg (ConfigType): Configs for SideAdapterNetwork module
        maskgen_cfg (ConfigType): Configs for RecWithAttnbias module
    """

    def __init__(self, num_classes: int, san_cfg: ConfigType,
                 maskgen_cfg: ConfigType, deep_supervision_idxs: List[int],
                 train_cfg: ConfigType,
                 ws=None,
                 wl=None,
                 **kwargs):
        super().__init__(
            in_channels=san_cfg.in_channels,
            channels=san_cfg.embed_dims,
            num_classes=num_classes,
            **kwargs)
        assert san_cfg.num_queries == maskgen_cfg.sos_token_num, \
            'num_queries in san_cfg should be equal to sos_token_num ' \
            'in maskgen_cfg'
        del self.conv_seg
        self.side_adapter_network = SideAdapterNetwork(**san_cfg) #san主要结构
        self.rec_with_attnbias = RecWithAttnbias(**maskgen_cfg) #掩码生成器模块。
        self.deep_supervision_idxs = deep_supervision_idxs
        self.train_cfg = train_cfg
        #原型融合放法
        self.WeightedSum = WeightedSum(wl,ws)
        self.GateSum = GateSum(2*num_classes,num_classes)
        if train_cfg:
            self.match_masks = MatchMasks(
                num_points=train_cfg.num_points,
                num_queries=san_cfg.num_queries,
                num_classes=num_classes,
                assigner=train_cfg.assigner)
        self.sup_propoty=torch.zeros(num_classes,100,1,1).to(device='cuda') #技术测试
        self.count=0 #测试计数

    def init_weights(self):

        rec_state_dict = None
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') == 'Pretrained_Part':
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            rec_state_dict = checkpoint.copy()
            para_prefix = 'decode_head.rec_with_attnbias'
            prefix_len = len(para_prefix) + 1
            for k, v in checkpoint.items():
                rec_state_dict.pop(k)
                if para_prefix in k:
                    rec_state_dict[k[prefix_len:]] = v

        self.side_adapter_network.init_weights()
        self.rec_with_attnbias.init_weights(rec_state_dict)


    '''train不需要传gt，
    test 要传，因为要分开支持集'''
    def forward(self, inputs: Tuple[Tensor],
                deep_supervision_idxs,
                support_gt=None) -> Tuple[List]:
        """Forward function.

        Args:
            inputs (Tuple[Tensor]): A triplet including images,
            list of multi-level visual features from image encoder and
            class embeddings from text_encoder.

        Returns:
            mask_props (List[Tensor]): Mask proposals predicted by SAN.
            mask_logits (List[Tensor]): Class logits of mask proposals.
        """
        imgs, clip_feature, class_embeds = inputs #image原始input2*3*640*640，vit output：4*list2（2*768*20*20），clipoutput：3*512
        # predict mask proposals and attention bias
        mask_props, attn_biases = self.side_adapter_network( #input，vit ouput，传入两部分特征，与文章相符。但是vitout和文章clipout一致。[7]应该是某个模块的7部分
            imgs, clip_feature, deep_supervision_idxs)#输出mask建议：list2（2*100*40*40）  和 注意力偏差 list2（2*12*100*40*40）：
        if support_gt is not None: #train
            support_f=mask_props[-1][-1:] #最后一份当作支持集，这样查询集会重复
            mask_props=[mask_prop[:-1] for mask_prop in mask_props]
        else:
            support_f=None
        # mask recognition with attention bias
        mask_embeds = [
            self.rec_with_attnbias(att_bias, clip_feature[-1]) #vit最后一个stage输出和att输入，对应原文clip最后粉色块
            for att_bias in attn_biases
        ]
        if support_gt is not None:#train
            mask_embeds=[ mask_embed[:-1] for mask_embed in mask_embeds] #去除支持集
        # Obtain class prediction of masks by comparing the similarity
        # between the image token and the text embedding of class names.
        mask_logits = [
            torch.einsum('bqc,nc->bqn', mask_embed, class_embeds)
            for mask_embed in mask_embeds
        ]
        return mask_props, mask_logits,support_f #这俩是啥？ predction前的mask proposal和proposal logit


    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        support=False #是否分离出支持集
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances = seg_data_to_instance_data(self.ignore_index,
                                                       batch_data_samples)#为什么会存在实例分割的东西？

        # forward-
        all_mask_props, all_mask_logits, _= self.forward(#经过forward方法，输出图2 san结尾的mask logit和clip结尾的mask props，但是在本类中均合并到san
            x, self.deep_supervision_idxs,None
            ) #传入support gt
        if support:
            batch_gt_instances=batch_gt_instances[:-1]#去除support

        if isinstance(batch_data_samples[0].img_shape, torch.Size):
            # slide inference
            size = batch_data_samples[0].img_shape
        elif 'pad_shape' in batch_data_samples[0]:
            size = (batch_data_samples[0].pad_shape)[:2]
        else:
            size = batch_data_samples[0].img_shape

        seg_logits=self.DPmodule(all_mask_props[-1], all_mask_logits[-1],size,all_mask_props[-1][-1:],batch_data_samples[-1].gt_sem_seg.data.unsqueeze(0)) #sup_f,sup_gt
        losses=self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): Images, visual features from image encoder
            and class embedding from text encoder.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        sup_gt=batch_img_metas[0].get('support_gt', None)
        mask_props, mask_logits ,support_f= self.forward(inputs, [],sup_gt)
        if sup_gt==None and mask_props[0].shape[0]>1:
            #input加入了support，但是gt没有的测试
            mask_props=[mask_prop[:-1] for mask_prop in mask_props]
            mask_logits=[mask_logit[:-1] for mask_logit in mask_logits]
        return self.predict_by_feat([mask_props[-1], mask_logits[-1]],
                                    batch_img_metas,
                                    support_f
                                    ) #只包含

    def predict_by_feat(self, seg_logits: List[Tensor],
                        batch_img_metas: List[dict],
                        support_f
                        ) -> Tensor:
        """1. Transform a batch of mask proposals to the input shape.
           2. Generate segmentation map with mask proposals and class logits.
        """
        mask_pred = seg_logits[0]
        cls_score = seg_logits[1] #1*100*3
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        #
        # # upsample mask
        seg_logits=self.DPmodule(mask_pred, cls_score,size) #

        return seg_logits

    def DPmodule(self,mask_pred, cls_score,size,support_f=None,support_gt=None):

        # upsample mask
        mask_pred = F.interpolate(
            mask_pred, size=size, mode='bilinear', align_corners=False) #1*100*512*736

        mask_cls = F.softmax(cls_score, dim=-1)[..., :-1]  #
        mask_pred = mask_pred.sigmoid()
        clip_pred = torch.einsum('bqc,bqhw->bchw', mask_cls, mask_pred)
        # return clip_pred
        if support_f is not None and support_gt is not None:  # 双原型版
            temp_sup = masked_average_pooling_km(support_f.detach().clone(), support_gt.detach().clone(),
                                                 num_classes=self.num_classes)
            self.sup_propoty =temp_sup
        else:
            self.count+=1
            # print('没选到',self.count)

        # 无论是否有support_f，就算没有有可能是分辨率问题
        if mask_pred.shape[0]>1: #训练
            sup_pred = F.cosine_similarity(self.sup_propoty.unsqueeze(0), mask_pred.unsqueeze(1).detach().clone(), dim=2) * 1
        else: #测试
            sup_pred = F.cosine_similarity(self.sup_propoty, mask_pred, dim=1) * 1
            sup_pred=sup_pred.unsqueeze(0)
        #### 结合两种预测方法
        seg_logits = self.WeightedSum(clip_pred, sup_pred)  # 融合
        return seg_logits
