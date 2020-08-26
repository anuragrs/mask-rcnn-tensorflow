NUM_GPU=8
BATCH_SIZE_PER_GPU=4
THROUGHPUT_LOG_FREQ=2000

cd /shared/mask-rcnn-tensorflow

herringrun \
-x LD_LIBRARY_PATH \
-x PATH \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 \
-x NCCL_DEBUG=INFO \
-x TENSORPACK_FP16=1 \
-x TF_CUDNN_USE_AUTOTUNE=0 \
-x TF_ENABLE_NHWC=1 \
-x HOROVOD_CYCLE_TIME=0.5 \
-x HOROVOD_FUSION_THRESHOLD=67108864 \
-x FI_PROVIDER="efa" \
python /shared/mask-rcnn-tensorflow/MaskRCNN/train.py \
--logdir /shared/logs/train_log \
--fp16 \
--throughput_log_freq ${THROUGHPUT_LOG_FREQ} \
--config \
TRAIN.BATCH_SIZE_PER_GPU=4 \
MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=/shared/data/coco/ \
PREPROC.PREDEFINED_PADDING=False \
BACKBONE.WEIGHTS=/shared/data/coco/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=FreezeBN \
TRAIN.WARMUP_INIT_LR=0.000416666666667 \
TRAINER=herring
