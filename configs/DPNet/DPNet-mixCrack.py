_base_ = [
    '../_base_/models/DP_vit-b16.py', '../_base_/datasets/mixCrack.py',
    '../_base_/ep_runtime.py', '../_base_/schedules/schedule_epoch.py'
]
crop_size = (512, 512)
# By default, models are trained on 8 GPUs with 2 images per GPU

# By default, models are trained on 4 GPUs with 8 images per GPU
train_dataloader = dict(batch_size=4,
                        sampler = dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(batch_size=1
                      # ,dataset=dict(metainfo=metainfo)
                      )
test_dataloader = val_dataloader

data_preprocessor = dict(
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    size_divisor=640,
    test_cfg=dict(size_divisor=32))
num_classes=2
model = dict(
    pretrained="../pretrain/clip_vit-base-patch16-224_3rdparty-d08f8887.pth",
    text_encoder=dict(dataset_name='crack500'),
    decode_head=dict(num_classes=num_classes,
                     loss_decode=[dict(type='CrossEntropyLoss',
                                       loss_name='loss_cls_ce',
                                       loss_weight=2.0,
                                       ),
                                  dict(type='CrossEntropyLoss',
                                       use_sigmoid=True,
                                       loss_name='loss_mask_ce',
                                       loss_weight=5.0),
                                  dict(type='DiceLoss',
                                       ignore_index=None,
                                       naive_dice=True,
                                       eps=1,
                                       loss_name='loss_mask_dice',
                                       loss_weight=5.0)
                                  ],
                     wl=0.8,ws=0.2
                     )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=5),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=100,
        by_epoch=True,
    )
]
