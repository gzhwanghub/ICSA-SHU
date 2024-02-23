//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_GROUP_ADMM_H
#define GR_ADMM_GROUP_ADMM_H


#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR

#include<mpi.h>
#include<stdio.h>
#include<iostream>
#include<cmath>
#include<fstream>
#include<time.h>
#include<stdio.h>
#include"prob.h"
#include"conf_util.h"
#include "neighbors.h"
#include "sparse_dataset.h"

using namespace std;


class ADMM {
public:
    ADMM(args_t *args, SparseDataset *sparseDataset);

    ~ADMM();

    void x_update(double *x, double *sumX, int dimension, double rho, SparseDataset *dataset);

    void x_update_OneOrder(double *x, double *sumX, int dimension, double c, double rho, SparseDataset *dataset);

    void alpha_update(double *w, double *sumx);

    void group_train(clock_t start_time);

    double predict();

    double LossValue();

    ofstream of;
    neighbors nears;//邻居节点包括本节点
    SparseDataset *test_data_;
    double sum_comm;
    double sum_cal;
    int DynamicGroup;
    int QuantifyPart;
    int Comm_method;
    int Update_method;
    double **msgbuf;
    double **al;
//    double Q_time=0.0;
    int Sparse_comm;
private:
    int32_t dataNum, dim;//数据的行数 维度
    int32_t myid, procnum;//当前节点的ID 总进程数
    double *x, *alpha, *C, *msgX;
    double rho;//惩罚参数

    //problem *prob;
    SparseDataset *sparseDataset;
    MPI_Status *statuses;
//    string outfile;
//    double wait_time;
    int maxIteration;
    int nodesOfGroup;//每个组的节点数
};


#endif //GR_ADMM_GROUP_ADMM_H
