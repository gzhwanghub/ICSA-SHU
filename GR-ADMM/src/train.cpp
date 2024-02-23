//
// Created by cluster on 2020/10/14.
//
#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include "../include/conf_util.h"
#include "../include/group_admm.h"
#include "../include/sparse_dataset.h"
#include "../include/properties.h"
#include "../utils/utils.h"
#include "../allreduce/reduceoperator.h"
#include "../allreduce/simpleallreduce.h"

using namespace std;

//目的：每次迭代前节点划分组，组内集合通信
int main(int argc, char **argv) {
    int myid, procnum;
    double start_time, end_time;
    char filename[100];
    char testfilename[100];
    int maxdim;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
    //获取配置文件的信息并存如argument
    args_t *args = new args_t(myid, procnum);
    Properties properties(argc, argv);
    double rho = properties.GetDouble("rho");
    int maxIteration = properties.GetInt("maxIteration");
    int nodesOfGroup = properties.GetInt("nodesOfGroup");
    int DynamicGroup = properties.GetInt("DynamicGroup");
    int QuantifyPart = properties.GetInt("QuantifyPart");
    int Comm_method = properties.GetInt("Comm_method");
    int Update_method = properties.GetInt("Update_method");
    int Repeat_iter = properties.GetInt("repeat_iter");
    int Sparse_comm = properties.GetInt("sparse_comm");
    std::string train_data_path = properties.GetString("train_data_path");
    std::string test_data_path = properties.GetString("test_data_path");
    if (myid == procnum - 1) {//最后一个节点作为组生成器 不需要读文件
        utils utils(Repeat_iter);
        utils.MasterNodes(procnum, nodesOfGroup, DynamicGroup, maxIteration);
        MPI_Finalize();
    } else {
        //训练数据
        sprintf(filename, train_data_path.c_str(), procnum - 1, myid);
        string dataname(filename);
        string inputfile = dataname;
        //测试数据
        sprintf(testfilename, test_data_path.c_str());
        string testinputfile = testfilename;
        SparseDataset *sparseDataset = new SparseDataset(inputfile);
        int temp = 0;
        maxdim = sparseDataset->GetDimension();
        temp = maxdim;
        vector<int> nodelist;
        for (int i = 0; i < procnum - 1; i++)
            nodelist.push_back(i);
        spar::SimpleAllreduce<spar::MaxOperator, int>(&temp, 1, myid, nodelist, MPI_COMM_WORLD);
        sparseDataset->SetDimension(temp);
        if (myid == 0)
            cout << "datanum " << sparseDataset->GetSampleNumber() << " dim " << sparseDataset->GetDimension() << endl;
        args->train_data_path = inputfile;
        args->test_data_path = testinputfile;
        args->maxIteration = maxIteration;
        args->nodesOfGroup = nodesOfGroup;
        args->rho = rho;
        args->Comm_method = Comm_method;
        args->Update_method = Update_method;
        args->Repeat_iter = Repeat_iter;
        ADMM admm(args, sparseDataset);
        admm.test_data_ = new SparseDataset(testinputfile);
        admm.sum_cal = 0.0;
        admm.sum_comm = 0.0;
        admm.QuantifyPart = QuantifyPart;
        admm.DynamicGroup = DynamicGroup;
        admm.Update_method = Update_method;
        admm.Sparse_comm = Sparse_comm;
        //输出相关配置信息
        if (myid == 0)
            args->print_args();
        start_time = MPI_Wtime();
        admm.group_train(start_time);
        end_time = MPI_Wtime();
        MPI_Finalize();
        if (myid == 0) {
            double temp = (double) (end_time - start_time);
            cout << "run time: " << temp << "  "
                 << "comm time:" << admm.sum_comm << "  "
                 << "cal  time:" << temp - admm.sum_comm << endl;
        }
    }
}