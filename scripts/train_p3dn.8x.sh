#!/usr/bin/env bash
NUM_GPU=8
BATCH_SIZE_PER_GPU=4
THROUGHPUT_LOG_FREQ=2000


echo ""
echo "NUM_GPU: ${NUM_GPU}"
echo "BATCH_SIZE_PER_GPU: ${BATCH_SIZE_PER_GPU}"
echo "THROUGHPUT_LOG_FREQ: ${THROUGHPUT_LOG_FREQ}"
echo ""

cd /shared/mask-rcnn-tensorflow

/opt/amazon/openmpi/bin/mpirun \
--allow-run-as-root \
--mca plm_rsh_no_tree_spawn 1 \
-mca btl_tcp_if_exclude lo,docker0 \
--tag-output \
--hostfile /shared/hostfiles/hosts_8x \
-N 8 \
--oversubscribe \
-x LD_LIBRARY_PATH \
-x PATH \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_DEBUG=INFO \
-x TENSORPACK_FP16=1 \
-x FI_PROVIDER="efa" \
-x FI_OFI_RXR_RX_COPY_UNEXP=1 \
-x FI_OFI_RXR_RX_COPY_OOO=1 \
-x FI_EFA_MR_CACHE_ENABLE=1 \
-x TF_CUDNN_USE_AUTOTUNE=0 \
-x TF_ENABLE_NHWC=1 \
-x FI_OFI_RXR_INLINE_MR_ENABLE=1 \
-x NCCL_TREE_THRESHOLD=4294967296 \
-x NCCL_MIN_NRINGS=13 \
-x HOROVOD_CYCLE_TIME=0.5 \
-x HOROVOD_FUSION_THRESHOLD=67108864 \
--output-filename /shared/logs/mpirun_logs \
bash /shared/mask-rcnn-tensorflow/scripts/launcher.sh \
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
TRAIN.GRADIENT_CLIP=1.5 \
TRAINER=horovod
