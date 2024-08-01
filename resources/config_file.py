point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=1,
        to_float32=True,
        pad_empty_sweeps=True,
        test_mode=False,
        sweep_range=[3, 27]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.625),
            final_dim=(320, 800),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'intrinsics', 'extrinsics', 'pad_shape', 'scale_factor',
                   'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                   'sample_idx', 'timestamp'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=1,
        to_float32=True,
        pad_empty_sweeps=True,
        sweep_range=[3, 27]),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.625),
            final_dim=(320, 800),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=False),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'intrinsics', 'extrinsics', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'sample_idx',
                           'timestamp'))
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/mmdet3d_nuscenes_30f_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadMultiViewImageFromMultiSweepsFiles',
                sweeps_num=1,
                to_float32=True,
                pad_empty_sweeps=True,
                test_mode=False,
                sweep_range=[3, 27]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=True),
            dict(
                type='GlobalRotScaleTransImage',
                rot_range=[-0.3925, 0.3925],
                translation_std=[0, 0, 0],
                scale_ratio_range=[0.95, 1.05],
                reverse_angle=True,
                training=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D',
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'intrinsics', 'extrinsics', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'sample_idx',
                           'timestamp'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True),
    val=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/mmdet3d_nuscenes_30f_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadMultiViewImageFromMultiSweepsFiles',
                sweeps_num=1,
                to_float32=True,
                pad_empty_sweeps=True,
                sweep_range=[3, 27]),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img'],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'intrinsics', 'extrinsics',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'sample_idx', 'timestamp'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/mmdet3d_nuscenes_30f_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True), 
            dict(
                type='LoadMultiViewImageFromMultiSweepsFiles',
                sweeps_num=1,
                to_float32=True,
                pad_empty_sweeps=True,
                sweep_range=[3, 27]),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img'],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'intrinsics', 'extrinsics',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'sample_idx', 'timestamp'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='LoadMultiViewImageFromMultiSweepsFiles',
            sweeps_num=1,
            to_float32=True,
            pad_empty_sweeps=True,
            sweep_range=[3, 27]),
        dict(
            type='ResizeCropFlipImage',
            data_aug_conf=dict(
                resize_lim=(0.47, 0.625),
                final_dim=(320, 800),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                H=900,
                W=1600,
                rand_flip=True),
            training=False),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=['img'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'intrinsics', 'extrinsics',
                               'pad_shape', 'scale_factor', 'flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'sample_idx', 'timestamp'))
            ])
    ])
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320/'
load_from = 'ckpts/fcos3d_repvgg_h2_epoch_12_remapped.pth'
resume_from = None
workflow = [('train', 1)]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
num_layers = 3
model = dict(
    type='Petr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='RepVGG',
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        out_indices=(2, 3),
        deploy=False,
        pretrained='ckpts/fcos3d_repvgg_h2_epoch_12_remapped.pth'),
    img_neck=None,
    pts_bbox_head=dict(
        type='PETRv2Head',
        position_level=1,
        num_classes=10,
        in_channels=1280,
        num_query=304,
        LID=True,
        with_position=True,
        with_multiview=True,
        with_fpe=True,
        with_time=True,
        with_multi=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        num_pred=3,
        transformer=dict(
            type='PETRTransformer',
            input_norm=True,
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=3,
                post_norm_cfg=None,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    norm_cfg=dict(type='UN'),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))))
db_sampler = dict(
    data_root='data/nuscenes/',
    info_path='data/nuscenes/nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk')))
ida_aug_conf = dict(
    resize_lim=(0.47, 0.625),
    final_dim=(320, 800),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 120
find_unused_parameters = False
runner = dict(type='EpochBasedRunner', max_epochs=120)
gpu_ids = range(0, 4)
