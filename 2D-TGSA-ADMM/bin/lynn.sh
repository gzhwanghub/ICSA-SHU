#!/bin/sh
#BSUB -e /data/home/xjsjleiyongmei/wdx/asyn_group/gcadmm/gcadmm.err
#BSUB -o /data/home/xjsjleiyongmei/wdx/asyn_group/gcadmm/gcadmm32.out
#BSUB -n 32
#BSUB -q priority
#BSUB -J gcADMM
#BSUB -R "span[ptile=4]"
ncpus=`cat $LSB_DJOB_HOSTFILE | wc -l `

 
  
  
mpirun -machine $LSB_DJOB_HOSTFILE -np ${ncpus} /data/home/xjsjleiyongmei/wdx/asyn_group/gcadmm/bin/admm  
 

 
$LSB_DJOB_HOSTFILE
