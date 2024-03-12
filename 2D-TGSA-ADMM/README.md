# 2D-TGSA-ADMM

This repo contains the 2D-TGSA-ADMM algorithm, proposed by [Communication-efficient ADMM-based distributed algorithms for sparse training](https://gzhwanghub.github.io/publication/2d-tgsa-admm/).

Grouped Sparse AllReduce based on the 2D-Torus topology (2D-TGSA) is a *communication-efficient* synchronization algorithm. Moreover, we integrate the general form consistent ADMM with 2D-TGSA to develop a distributed algorithm (2D-TGSA-ADMM) that exhibits excellent *scalability* and can effectively handle large-scale distributed optimization problems.

## Innovation

* Sparse 2D-TDGA communication model;
* Constructing a distributed training framework based on sparse generalized consistent ADMM;
* TopK sparse computation and dynamic penalty term parameters;

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
mpirun -np 16 -f ./hostfile ./aparse_admm 
```


## Citation
```bibtex
@article{wang2023communication,
  title={Communication-efficient ADMM-based distributed algorithms for sparse training},
  author={Wang, Guozheng and Lei, Yongmei and Qiu, Yongwen and Lou, Lingfei and Li, Yixin},
  journal={Neurocomputing},
  volume={550},
  pages={126456},
  year={2023},
  publisher={Elsevier}
}
```
 

