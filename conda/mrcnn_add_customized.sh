########### DO TENSORPACK CUSTOM STUFF HERE ###############
git clone https://github.com/anuragrs/mask-rcnn-tensorflow.git
cd mask-rcnn-tensorflow
git checkout step_time
# make TF mods
sed -i 's#third_party/gpus/cuda/include/###' /shared/conda/lib/python3.8/site-packages/tensorflow/include/tensorflow/core/util/gpu_device_functions.h
sed -i 's#third_party/gpus/cuda/include/###' /shared/conda/lib/python3.8/site-packages/tensorflow/include/tensorflow/core/util/gpu_kernel_helper.h
ln -s /shared/conda/lib/python3.8/site-packages/tensorflow/libtensorflow_framework.so.2 /shared/conda/lib/python3.8/site-packages/tensorflow/libtensorflow_framework.so
# build custom ops
cd MaskRCNN/model/custom_ops/roi_align
make
###########################################################
