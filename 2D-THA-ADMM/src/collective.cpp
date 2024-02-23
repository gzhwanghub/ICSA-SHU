//
// Created by guozhengwanzg on 2021/6/15.
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <math.h>
#include "collective.h"
#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3
void ReduceSum(double *dst, double *src, int size){
    for (int i = 0; i < size; ++i) {
        dst[i] += src[i];
    }
}

void ReduceReplace(double *dst, double *src, int size){
    for (int i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

int Next(int rank, int size){
    return ((rank + 1) % size);
}

int Prev (int rank , int size){
    return ((size + rank - 1) % size);
}

Collective::Collective(args_t *args, problem *problem) {
    comm_rank_ = args->myid;
    comm_size_ = args->procnum;
    dim_ = problem->n;
    datatype_ = MPI_DOUBLE;
    sqrt_leader_ = args->sqrt_leader_;
    worker_per_group_ = args->worker_per_group_;
    sqrt_procnum_ = args->sqrt_procnum_;
}

void Collective::RingAllreduce(double *data, int count ,MPI_Comm communicator){
    int comm_rank;
    int comm_size;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    int segment_size = count / comm_size;
    int residual = count % comm_size;
    int *segment_sizes = (int *) malloc(sizeof(int) * comm_size);
    int *segment_start_ptr = (int *) malloc(sizeof(int) * comm_size);
    int segment_ptr = 0;
    for(int i = 0; i < comm_size; i++){
        segment_start_ptr[i] = segment_ptr;
        segment_sizes[i] = segment_size;
        if(i < residual){
            segment_sizes[i]++;
        }
        segment_ptr += segment_sizes[i];
    }
    if(segment_start_ptr[comm_size - 1] + segment_sizes[comm_size - 1] != count){
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);
    }
    MPI_Status recv_status;
    MPI_Request recv_req;
    double *buffer = (double *) malloc(sizeof(double) * segment_sizes[0]);
    for(int iter = 0; iter < comm_size - 1; iter++){
        int recv_chunk = (comm_rank - iter - 1 + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype_,
                  Prev(comm_rank, comm_size), 0, communicator, &recv_req);
        MPI_Send(sending_segment, segment_sizes[send_chunk], datatype_,
                 Next(comm_rank, comm_size), 0, communicator);
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Wait(&recv_req, &recv_status);
        ReduceSum(updating_segment, buffer, segment_sizes[recv_chunk]);
    }
    MPI_Barrier(communicator);
    for(int iter = 0; iter < comm_size - 1; iter++){
        int recv_chunk = (comm_rank - iter + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + 1 + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Sendrecv(sending_segment, segment_sizes[send_chunk], datatype_,
                     Next(comm_rank, comm_size), 0, updating_segment,
                     segment_sizes[recv_chunk], datatype_,
                     Prev(comm_rank, comm_size), 0, communicator, &recv_status);
    }
    free(buffer);
}

void Collective::TorusAllreduce(double *data, int worker_per_group, int group_count, MPI_Comm communicator, int *nbrs){
    int comm_rank;
    int comm_size;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    comm_size /= worker_per_group;
    int segment_size = dim_ / comm_size;
    int residual = dim_ % comm_size;
    int *segment_sizes = (int *) malloc(sizeof(int) * comm_size);
    int *segment_start_ptr = (int *) malloc(sizeof(int) * comm_size);
    int segment_ptr = 0;
    for(int i = 0; i < comm_size; i++){
        segment_start_ptr[i] = segment_ptr;
        segment_sizes[i] = segment_size;
        if(i < residual){
            segment_sizes[i]++;
        }
        segment_ptr += segment_sizes[i];
    }
    if(segment_start_ptr[comm_size - 1] + segment_sizes[comm_size - 1] != dim_){
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);
    }
    MPI_Status recv_status;
    MPI_Request recv_req;
    double *buffer = (double *) malloc(sizeof(double) * segment_sizes[0]);
    //scatter-reduce
    for(int iter = 0; iter < comm_size - 1; iter++){
        int recv_chunk = (comm_rank - iter - 1 + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype_,
                  nbrs[0], 0, communicator, &recv_req);
        MPI_Send(sending_segment, segment_sizes[send_chunk], datatype_,
                 nbrs[1], 0, communicator);
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Wait(&recv_req, &recv_status);
        ReduceSum(updating_segment, buffer, segment_sizes[recv_chunk]);
    }
    MPI_Barrier(communicator);
    //segment ringallreduce
    int color = comm_rank % group_count;
    MPI_Comm subgrp_comm; //intra-group
    MPI_Comm_split(communicator, color, comm_rank, &subgrp_comm);
    int subgrp_rank, subgrp_size;
    MPI_Comm_rank(subgrp_comm, &subgrp_rank);
    MPI_Comm_size(subgrp_comm, &subgrp_size);
    int reduce_chunk = (color + 1) % group_count;
    double *reduce_segment = &(data[segment_start_ptr[reduce_chunk]]);
    RingAllreduce(reduce_segment, segment_sizes[reduce_chunk], subgrp_comm);
//    MPI_Allreduce(reduce_segment, buffer,segment_sizes[reduce_chunk], datatype_, MPI_SUM, subgrp_comm);
//    ReduceReplace(reduce_segment,buffer,segment_sizes[reduce_chunk]);
    //allgather
    for(int iter = 0; iter < comm_size - 1; iter++){
        int recv_chunk = (comm_rank - iter + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + 1 + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Sendrecv(sending_segment, segment_sizes[send_chunk], datatype_,
                     nbrs[1], 0, updating_segment,
                     segment_sizes[recv_chunk], datatype_,
                     nbrs[0], 0, communicator, &recv_status);
    }
    free(buffer);
}

void Collective::HierarchicalTorus(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM, int *nbrs){
    RingAllreduce(data, dim_, SUBGRP_COMM);
    if (MAINGRP_COMM != MPI_COMM_NULL){
        int main_size;
        MPI_Comm_size(MAINGRP_COMM, &main_size);
        MPI_Comm RING_COMM;
        CreateTorus(MAINGRP_COMM, RING_COMM, sqrt(main_size), main_size, nbrs);
        TorusAllreduce(data, sqrt(main_size),sqrt(main_size), RING_COMM, nbrs);
    }
    MPI_Bcast(data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::HierarchicalAllreduce(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM){
//    double* recv_data;
//    recv_data = new double[dim_];
//    MPI_Reduce(data,recv_data,dim_, datatype_, MPI_SUM,0, SUBGRP_COMM);
    RingAllreduce(data, dim_,SUBGRP_COMM);
    if(MAINGRP_COMM != MPI_COMM_NULL){
        RingAllreduce(data, dim_,MAINGRP_COMM);
    }
    MPI_Bcast(data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::CreateTorus(MPI_Comm OLD_COMM, MPI_Comm &TORUS_COMM, int worker_per_group, int main_size, int *nbrs){
    int comm_rank, comm_size;
    int dims[2] = {worker_per_group, main_size / worker_per_group}, periods[2] = {0,1}, reorder = 1, coords[2];
    MPI_Status status;
    MPI_Cart_create(OLD_COMM, 2, dims, periods, 1, &TORUS_COMM);
    MPI_Comm_rank(TORUS_COMM, &comm_rank);
    MPI_Cart_coords(TORUS_COMM, comm_rank,2, coords);
    MPI_Cart_shift(TORUS_COMM, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);
    MPI_Cart_shift(TORUS_COMM, 0, 1, &nbrs[UP], &nbrs[DOWN]);
}
