#!/bin/bash
export OMP_NUM_THREADS=24
yhrun -N 9 -n 9 -c 24 ./admm_end /WORK/shu_gzwang_1/dataset/kdd12/%d/data%03d  /WORK/shu_gzwang_1/dataset/kdd12/test 4 4 24 mdcd
