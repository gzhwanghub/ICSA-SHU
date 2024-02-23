#!/bin/sh
#BSUB -e /data/home/xjsjleiyongmei/wdx/asyn_group/adadmm/admm.err
#BSUB -o /data/home/xjsjleiyongmei/wdx/asyn_group/adadmm/admm64.out
#BSUB -n 65
#BSUB -q priority
#BSUB -J adadmm
#BSUB -R "span[ptile=8]"
ncpus=`cat $LSB_DJOB_HOSTFILE | wc -l `

 
 

  
  
 
 

  
mpirun -machine $LSB_DJOB_HOSTFILE -np ${ncpus} /data/home/xjsjleiyongmei/wdx/asyn_group/adadmm/src/admm 0 64 5 
mpirun -machine $LSB_DJOB_HOSTFILE -np ${ncpus} /data/home/xjsjleiyongmei/wdx/asyn_group/adadmm/src/admm 0 32 5
 
$LSB_DJOB_HOSTFILE
