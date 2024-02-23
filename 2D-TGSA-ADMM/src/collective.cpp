//
// Created by guozhengwang on 2021/6/15.
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <math.h>
#include "collective.h"
#include "p2p_communication.h"
#include "common.h"
#include "reduce_operator.h"
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

void Collective::TorusAllreduce(double *data, int sqrt_number, MPI_Comm communicator, int *nbrs){
    int comm_rank;
    int comm_size;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    comm_size /= sqrt_number;
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
    /*scatter-reduce*/
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
//    MPI_Barrier(communicator);
    /*segment ringallreduce*/
    int worker_per_group = sqrt_number;
    int color = comm_rank % worker_per_group;
    MPI_Comm subgrp_comm; //intra-group
    MPI_Comm_split(communicator, color, comm_rank, &subgrp_comm);
    int subgrp_rank, subgrp_size;
    MPI_Comm_rank(subgrp_comm, &subgrp_rank);
    MPI_Comm_size(subgrp_comm, &subgrp_size);
    int reduce_chunk = (color + 1) % worker_per_group;
    double *reduce_segment = &(data[segment_start_ptr[reduce_chunk]]);
//    RingAllreduce(reduce_segment, segment_sizes[reduce_chunk], subgrp_comm);
    // Don't call function
    MPI_Allreduce(reduce_segment, buffer,segment_sizes[reduce_chunk], datatype_, MPI_SUM, subgrp_comm);
    ReduceReplace(reduce_segment,buffer,segment_sizes[reduce_chunk]);
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

void ::Collective::MoshpitAllreduce(double *data, double* sum_data, MPI_Comm &SUBGRP_COMM, MPI_Comm &SUBGRP_COMM_Y){
    MPI_Allreduce(data, sum_data, dim_, datatype_, MPI_SUM, SUBGRP_COMM);
    MPI_Allreduce(sum_data, data, dim_, datatype_, MPI_SUM, SUBGRP_COMM_Y);
}

void::Collective::WRHT(double* data, double* sum_data, MPI_Comm& MAINGRP_COMM, MPI_Comm& SUBGRP_COMM){
    MPI_Reduce(data, sum_data, dim_, datatype_, MPI_SUM, 0, SUBGRP_COMM);
    if(MAINGRP_COMM != MPI_COMM_NULL){
        MPI_Reduce(sum_data, data, dim_, datatype_, MPI_SUM, 0, MAINGRP_COMM);
        MPI_Bcast(data, dim_, datatype_, 0, MAINGRP_COMM);
    }
    MPI_Bcast(data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::HierarchicalTorus(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM, int *nbrs){
    RingAllreduce(data, dim_, SUBGRP_COMM);
    if (MAINGRP_COMM != MPI_COMM_NULL){
        int main_size;
        MPI_Comm_size(MAINGRP_COMM, &main_size);
        MPI_Comm RING_COMM;
        CreateTorus(MAINGRP_COMM, RING_COMM, sqrt(main_size), nbrs);
        TorusAllreduce(data, sqrt(main_size), RING_COMM, nbrs);
    }
    MPI_Bcast(data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::SparseHierarchicalTorus(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM, int *nbrs){
    SparseRingAllreduce(data, dim_, SUBGRP_COMM);
    if (MAINGRP_COMM != MPI_COMM_NULL){
        int main_size;
        MPI_Comm_size(MAINGRP_COMM, &main_size);
        MPI_Comm RING_COMM;
        CreateTorus(MAINGRP_COMM, RING_COMM, sqrt(main_size), nbrs);
        SparseTorusAllreduce(data, sqrt(main_size), RING_COMM, nbrs);
    }
    MPI_Bcast(data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::HierarchicalRingAllreduce(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM){
    RingAllreduce(data, dim_,SUBGRP_COMM);
    if(MAINGRP_COMM != MPI_COMM_NULL){
        RingAllreduce(data, dim_,MAINGRP_COMM);
    }
    MPI_Bcast(data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::HierarchicalSparseAllreduce(double *data, double* sum_data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM){
    MPI_Reduce(data, sum_data, dim_, datatype_, MPI_SUM, 0, SUBGRP_COMM);
    if(MAINGRP_COMM != MPI_COMM_NULL){
//        MPI_Allreduce(sum_data, data, dim_, datatype_ ,MPI_SUM, MAINGRP_COMM);
        SparseRingAllreduce(sum_data, dim_, MAINGRP_COMM);
    }
    MPI_Bcast(sum_data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::CreateTorus(MPI_Comm OLD_COMM, MPI_Comm &TORUS_COMM, int sqrt_number, int *nbrs){
    int comm_rank, comm_size;
    int dims[2] = {sqrt_number, sqrt_number}, periods[2] = {1,1}, reorder = 1, coords[2];
    MPI_Status status;
    MPI_Cart_create(OLD_COMM, 2, dims, periods, 1, &TORUS_COMM);
    MPI_Comm_rank(TORUS_COMM, &comm_rank);
    MPI_Cart_coords(TORUS_COMM, comm_rank,2, coords);
    MPI_Cart_shift(TORUS_COMM, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);
    MPI_Cart_shift(TORUS_COMM, 0, 1, &nbrs[UP], &nbrs[DOWN]);
}

void Collective::SparseRingAllreduce(double *data, int count, MPI_Comm communicator) {
    int comm_size;
    int comm_rank;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    if (comm_size == 1) return;
    int block_size = count / comm_size;
    int residual = count % comm_size;
    std::map<int, std::pair<int, int>> blocks;
    int segment_start = 0;
    for (int i = 0; i < comm_size; ++i) {
        blocks[i].first = segment_start;
        blocks[i].second = block_size;
        if(i < residual){
            blocks[i].second++;
        }
        segment_start += blocks[i].second;
    }
    if(blocks[comm_size - 1].first + blocks[comm_size - 1].second != count){
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);
    }
    bool need_check = true;
    int isize = sizeof(int);
    int vsize = sizeof(double);
    int *index_buffer = new int[count];
    double *value_buffer = new double[count];
    double *recv_buffer = new double[count];
    int send_block_index = comm_rank;
    int recv_block_index = (send_block_index - 1 + comm_size) % comm_size;
    for (int i = 0; i < comm_size - 1; ++i) {
        MPI_Status statuses[2];
        MPI_Request requests[2];
        double *base = data + blocks[send_block_index].first;
        block_size = blocks[send_block_index].second;
        //如果上一次迭代接收到了稠密的块，那么本次传输就不用检查了，必然稠密
        if (need_check) {
            int nnz = 0;
            for (int j = 0; j < block_size; ++j) {
                if (base[j] != 0) {
                    ++nnz;
                }
            }
            //如果满足稀疏传输条件则采用稀疏传输
            if (nnz * (isize + vsize) < block_size * vsize) {
                int k = 0;
                for (int j = 0; j < block_size; ++j) {
                    if (base[j] != 0) {
                        value_buffer[k] = base[j];
                        index_buffer[k] = j;
                        ++k;
                    }
                }
                //把元素的值和元素的索引放到同一个内存空间上，一同发送，不分两次发送了
                memcpy(value_buffer + nnz, index_buffer, isize * nnz);
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, Next(comm_rank, comm_size),
                          spar::MessageType::kScatterReduce, communicator, &requests[0]);
            } else {
                MPI_Isend(base, block_size * vsize, MPI_CHAR, Next(comm_rank, comm_size), spar::kScatterReduce, communicator,
                          &requests[0]);
            }
        } else {
            MPI_Isend(base, block_size * vsize, MPI_CHAR, Next(comm_rank, comm_size), spar::MessageType::kScatterReduce, communicator,
                      &requests[0]);
        }
        base = data + blocks[recv_block_index].first;
        block_size = blocks[recv_block_index].second;
        MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, Prev(comm_rank, comm_size), spar::kScatterReduce, communicator, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        int nnz = 0;
        MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
        //如果接收到的字节数少于这个块应有的大小，说明左边进程发送的是稀疏数据
        if (nnz < block_size * vsize) {
            nnz = nnz / (isize + vsize);
            //分离值和索引
            memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
            for (int j = 0; j < nnz; ++j) {
                spar::Reduce<spar::SumOperator>(base[index_buffer[j]], recv_buffer[j]);
            }
            need_check = true;
        } else {
            for (int j = 0; j < block_size; ++j) {
                spar::Reduce<spar::SumOperator>(base[j], recv_buffer[j]);
            }
            need_check = false;
        }
        // 下一次循环发送这次循环接收到的segment
        send_block_index = recv_block_index;
        recv_block_index = (recv_block_index - 1 + comm_size) % comm_size;
    }

    for (int i = 0; i < comm_size - 1; ++i) {
        MPI_Status statuses[2];
        MPI_Request requests[2];
        double *base = data + blocks[send_block_index].first;
        block_size = blocks[send_block_index].second;
        if (need_check) {
            int nnz = 0;
            for (int j = 0; j < block_size; ++j) {
                if (base[j] != 0) {
                    ++nnz;
                }
            }
            if (nnz * (isize + vsize) < block_size * vsize) {
                int k = 0;
                for (int j = 0; j < block_size; ++j) {
                    if (base[j] != 0) {
                        value_buffer[k] = base[j];
                        index_buffer[k] = j;
                        ++k;
                    }
                }
                memcpy(value_buffer + nnz, index_buffer, isize * nnz);
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, Next(comm_rank, comm_size),
                          spar::kAllGather, communicator, &requests[0]);
            } else {
                MPI_Isend(base, block_size * vsize, MPI_CHAR, Next(comm_rank, comm_size), spar::kAllGather, communicator,
                          &requests[0]);
            }
        } else {
            MPI_Isend(base, block_size * vsize, MPI_CHAR, Next(comm_rank, comm_size), spar::kAllGather, communicator,
                      &requests[0]);
        }
        base = data + blocks[recv_block_index].first;
        block_size = blocks[recv_block_index].second;
        MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, Prev(comm_rank, comm_size), spar::kAllGather, communicator, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        int nnz = 0;
        MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
        if (nnz < block_size * vsize) {
            nnz = nnz / (isize + vsize);
            memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
            for (int j = 0; j < block_size; ++j) {
                base[j] = 0;
            }
            for (int j = 0; j < nnz; ++j) {
                base[index_buffer[j]] = recv_buffer[j];
            }
            need_check = true;
        } else {
            for (int j = 0; j < block_size; ++j) {
                base[j] = recv_buffer[j];
            }
            need_check = false;
        }
        send_block_index = recv_block_index;
        recv_block_index = (recv_block_index - 1 + comm_size) % comm_size;
    }
    delete[] recv_buffer;
    delete[] value_buffer;
    delete[] index_buffer;
}

void Collective::SparseTorusAllreduce(double *data, int sqrt_number, MPI_Comm communicator, int *nbrs){
    int comm_size;
    int comm_rank;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    if (comm_size == 1)
        return;
    comm_size /= sqrt_number;
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
    bool need_check = true;
    int isize = sizeof(int);
    int vsize = sizeof(double);
    int *index_buffer = new int[dim_];
    double *value_buffer = new double[dim_];
    double *recv_buffer = new double[dim_];
    MPI_Status statuses[2];
    MPI_Request requests[2];
    double *buffer = (double *) malloc(sizeof(double) * segment_sizes[0]);
    for (int iter = 0; iter < comm_size - 1; ++iter) {
        int recv_chunk = (comm_rank - iter - 1 + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        //如果上一次迭代接收到了稠密的块，那么本次传输就不用检查了，必然稠密
        if (need_check) {
            int nnz = 0;
            for (int j = 0; j < segment_sizes[send_chunk]; ++j) {
                if (sending_segment[j] != 0) {
                    ++nnz;
                }
            }
            //如果满足稀疏传输条件则采用稀疏传输
            if (nnz * (isize + vsize) < segment_sizes[send_chunk] * vsize) {
                int k = 0;
                for (int j = 0; j < segment_sizes[send_chunk]; ++j) {
                    if (sending_segment[j] != 0) {
                        value_buffer[k] = sending_segment[j];
                        index_buffer[k] = j;
                        ++k;
                    }
                }
                //把元素的值和元素的索引放到同一个内存空间上，一同发送，不分两次发送了
                memcpy(value_buffer + nnz, index_buffer, isize * nnz);
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR,
                         nbrs[1], 0, communicator, &requests[0]);
            } else {
                MPI_Isend(sending_segment, segment_sizes[send_chunk] * vsize, MPI_CHAR,
                         nbrs[1], 0, communicator, &requests[0]);
            }
        } else {
            MPI_Isend(sending_segment, segment_sizes[send_chunk] * vsize, MPI_CHAR,
                     nbrs[1], 0, communicator, &requests[0]);
        }

        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Irecv(recv_buffer, segment_sizes[send_chunk] * vsize, MPI_CHAR, nbrs[0], 0, communicator, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        int nnz = 0;
        MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
        //如果接收到的字节数少于这个块应有的大小，说明左边进程发送的是稀疏数据
        if (nnz < segment_sizes[recv_chunk] * vsize) {
            nnz = nnz / (isize + vsize);
            //分离值和索引
            memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
            for (int j = 0; j < nnz; ++j) {
                spar::Reduce<spar::SumOperator>(updating_segment[index_buffer[j]], recv_buffer[j]);
            }
            need_check = true;
        } else {
            for (int j = 0; j < segment_sizes[recv_chunk]; ++j) {
                spar::Reduce<spar::SumOperator>(updating_segment[j], recv_buffer[j]);
            }
            need_check = false;
        }
    }
    int worker_per_group = sqrt_number;
    int color = comm_rank % worker_per_group;
    MPI_Comm subgrp_comm;
    MPI_Comm_split(communicator, color, comm_rank, &subgrp_comm);
    int subgrp_rank , subgrp_size;
    MPI_Comm_rank(subgrp_comm, &subgrp_rank);
    MPI_Comm_size(subgrp_comm, &subgrp_size);
    int reduce_chunk = (color + 1) % worker_per_group;
    double* reduce_segment = &(data[segment_start_ptr[reduce_chunk]]);
    SparseRingAllreduce(reduce_segment, segment_sizes[reduce_chunk],
                        subgrp_comm);
    // allgather
    for (int iter = 0; iter < comm_size - 1; ++iter) {
        int recv_chunk = (comm_rank - iter + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + 1 + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        if (need_check) {
            int nnz = 0;
            for (int j = 0; j < segment_sizes[send_chunk]; ++j) {
                if (sending_segment[j] != 0) {
                    ++nnz;
                }
            }
            if (nnz * (isize + vsize) < segment_sizes[send_chunk] * vsize) {
                int k = 0;
                for (int j = 0; j < segment_sizes[send_chunk]; ++j) {
                    if (sending_segment[j] != 0) {
                        value_buffer[k] = sending_segment[j];
                        index_buffer[k] = j;
                        ++k;
                    }
                }
                memcpy(value_buffer + nnz, index_buffer, isize * nnz);
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, nbrs[1],
                          spar::MessageType::kAllGather, communicator, &requests[0]);
            } else {
                MPI_Isend(sending_segment, segment_sizes[send_chunk] * vsize, MPI_CHAR, nbrs[1], spar::MessageType::kAllGather, communicator,
                          &requests[0]);
            }
        } else {
            MPI_Isend(sending_segment, segment_sizes[send_chunk] * vsize, MPI_CHAR, nbrs[1], spar::MessageType::kAllGather, communicator,
                      &requests[0]);
        }
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Irecv(recv_buffer, segment_sizes[recv_chunk] * vsize, MPI_CHAR, nbrs[0], spar::MessageType::kAllGather, communicator, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        int nnz = 0;
        MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
        if (nnz < segment_sizes[recv_chunk] * vsize) {
            nnz = nnz / (isize + vsize);
            memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
            for (int j = 0; j < segment_sizes[recv_chunk]; ++j) {
                updating_segment[j] = 0;
            }
            for (int j = 0; j < nnz; ++j) {
                updating_segment[index_buffer[j]] = recv_buffer[j];
            }
            need_check = true;
        } else {
            for (int j = 0; j < segment_sizes[recv_chunk]; ++j) {
                updating_segment[j] = recv_buffer[j];
            }
            need_check = false;
        }
    }
    delete[] recv_buffer;
    delete[] value_buffer;
    delete[] index_buffer;
}


