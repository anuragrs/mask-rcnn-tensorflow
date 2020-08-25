TF_INSTALL_DIR=/shared/conda/lib/python3.8/site-packages
TF_INC=$(TF_INSTALL_DIR)/tensorflow/include
TF_LIB=$(TF_INSTALL_DIR)/tensorflow

CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = /shared/conda/pkgs/cuda-toolkit/bin/nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0
GPUCFLAGS = -c
LFLAGS    =  -shared -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -L/shared/conda/pkgs/cuda-toolkit/lib64 -I/shared/conda/pkgs/cuda-toolkit/include
GPULFLAGS = -x cu -shared -Xcompiler -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -I/shared/conda/pkgs/cuda-toolkit/include/thrust/system/cuda/detail
DEBUG = -g -G
GPUDEF    = -D GOOGLE_CUDA=1
CGPUFLAGS = -lcudart


SRC       = roi_align_op.cc
GPUSRC    = roi_align_op.cu.cc
PROD      = roi_align_op.so
GPUPROD = roi_align_op_cu.so

default: gpu

cpu:
	$(CXX) $(CFLAGS) $(SRC) $(LFLAGS) -o $(PROD)

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GPUSRC) $(GPULFLAGS) -o $(GPUPROD) $(GPUDEF) -I/usr/local/ --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED
	$(CXX) $(CFLAGS)  $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) -o $(PROD) $(GPUDEF)

clean:
	rm -f $(PROD) $(GPUPROD)

