//
// Created by Guozheng Wang on 2021/6/15.
//

#ifndef RING_ALLREDUCE_COLLECTIVE_H
#define RING_ALLREDUCE_COLLECTIVE_H


#include <math.h>
#include "properties.h"
#include "prob.h"

class Collective{
public:
    Collective(args_t *args, problem *problem);

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
//    MPI_Comm SUBGRP_COMM_;
//    MPI_Comm MAINGRP_COMM_;
};

#endif //RING_ALLREDUCE_COLLECTIVE_H