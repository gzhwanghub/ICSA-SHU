/*************************************************************************
    > File Name: collective.h
    > Description: Collective communication
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2021-06-15
 ************************************************************************/

#ifndef FIADMM_COLLECTIVE_H
#define FIADMM_COLLECTIVE_H


#include <math.h>
#include "properties.h"
#include "conf_util.h"

class Collective {
public:
    Collective();

    ~Collective();

    void RingAllreduce(double *data, int count, MPI_Comm communicator);

    void TorusAllreduce(double *data, int worker_per_group, int group_count, MPI_Comm communicator, int *nbrs);

    void HierarchicalTorus(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM, int *nbrs);

    void HierarchicalAllreduce(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM);

    void CreateTorus(MPI_Comm OLD_COMM, MPI_Comm &TORUS_COMM, int worker_per_group, int main_size, int *nbrs);

private:
    int comm_rank_, comm_size_, dim_;
    MPI_Datatype datatype_;
    int sqrt_procnum_, sqrt_leader_;
    int worker_per_group_;
};

#endif //FIADMM_COLLECTIVE_H
