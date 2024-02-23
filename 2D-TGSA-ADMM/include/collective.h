//
// Created by cluster on 2021/6/15.
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

    void TorusAllreduce(double *data, int sqrt_worker_num, MPI_Comm communicator, int *nbrs);

    void MoshpitAllreduce(double *data, double * sum_data, MPI_Comm &SUBGRP_COMM, MPI_Comm &SUBGRP_COMM_Y);

    void WRHT(double *data, double * sum_data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM);

    void HierarchicalTorus(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM, int *nbrs);

    void SparseHierarchicalTorus(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM, int *nbrs);

    void HierarchicalRingAllreduce(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM);

    void HierarchicalSparseAllreduce(double *data, double *global_model, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM);

    void CreateTorus(MPI_Comm OLD_COMM, MPI_Comm &TORUS_COMM, int sqrt_number, int *nbrs);

    void SparseRingAllreduce(double *data, int count,  MPI_Comm communicator);

    void SparseTorusAllreduce(double *data, int sqrt_worker_num, MPI_Comm communicator, int *nbrs);

private:
    int dim_;
    MPI_Datatype datatype_;
    int sqrt_procnum_, sqrt_leader_;
    int worker_per_group_;
//    MPI_Comm SUBGRP_COMM_;
//    MPI_Comm MAINGRP_COMM_;
};

#endif //RING_ALLREDUCE_COLLECTIVE_H