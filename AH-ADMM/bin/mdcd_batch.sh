#!/bin/bash
export OMP_NUM_THREADS=2
yhrun -N 9 -n 9 -c 2 ./admm_end /WORK/shu_gzwang_1/dataset/url/%d/data%03d /WORK/shu_gzwang_1/dataset/url/test 4 4 2 mdcd
