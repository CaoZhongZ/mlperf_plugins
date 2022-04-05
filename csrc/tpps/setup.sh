export LD_PRELOAD="${CONDA_PREFIX}/lib/libiomp5.so"
# export OMP_NUM_THREADS=56
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_HW_SUBSET=1s,56c,1t
./xxx.exe $1
