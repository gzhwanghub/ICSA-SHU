# ICSA-SHU
*Intelligent Computing System and Application Laboratory (ICSA)* belongs to the School of Computer Engineering and Science, Shanghai University, and is located in the 804 Laboratory of the East Computer Building, Baoshan Campus, Shanghai University. 

The laboratory mainly studies the application of distributed ADMM in high performance cluster.

## Quick Start
```shell
$ mkdir build
$ cd build && cmake .. && make
$ mpirun -np 16 -f ./hostfile ./executor
```
## submit
git add *
git commit -m "v"
git push
