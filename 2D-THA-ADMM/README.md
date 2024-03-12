# 2D-THA-ADMM

This repo contains the 2D-THA-ADMM algorithm, proposed by [2D-THA-ADMM: communication efficient distributed ADMM algorithm framework based on two-dimensional torus hierarchical AllReduce](https://gzhwanghub.github.io/publication/2d-tha-admm/).

This study introduces a hierarchical AllReduce algorithm designed on a two-dimensional torus (2D-THA) that leverages a hierarchical structure to synchronize model parameters and optimize bandwidth utilization. Subsequently, the 2D-THA synchronization algorithm is integrated with the alternating direction method of multipliers (ADMM) to form the distributed consensus algorithm known as 2D-THA-ADMM.

## Innovation

* hierarchical AllReduce algorithm designed on a two-dimensional torus (2D-THA);
* the 2D-THA is integrated with the ADMM to form the distributed consensus algorithm;

## Environment Requirements

* Ubuntu
* gcc
* MPI
* NFS

## Usage

```bash
mkdir build
cd build
cmake ..
make
cd ../bin
mpirun -np 16 -f ./hostfile ./admm_collective
```


## Citation
```bibtex
@article{wang20242d,
  title={2D-THA-ADMM: communication efficient distributed ADMM algorithm framework based on two-dimensional torus hierarchical AllReduce},
  author={Wang, Guozheng and Lei, Yongmei and Zhang, Zeyu and Peng, Cunlu},
  journal={International Journal of Machine Learning and Cybernetics},
  volume={15},
  number={2},
  pages={207--226},
  year={2024},
  publisher={Springer}
}
```
 

