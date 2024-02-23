//
// Created by cluster on 2020/10/14.
//
#include <iostream>
#include "../include/group_admm.h"
#include "../allreduce/common.h"
#include "../allreduce/reduceoperator.h"
#include "../allreduce/ringallreduce.h"
#include "../include/sparse_dataset.h"
#include "../include/simple_algebra.h"
#include "../include/lr_tron_optimizer.h"
#include "../include/gradient_decent_optimizer.h"
#include "../include/admm_lr_function.h"
#include <vector>
#include <unistd.h>

#define  random(x)(rand()%x)
using namespace spar;


ADMM::ADMM(args_t *args, SparseDataset *sp)//problem *prob1)
{
    sparseDataset = sp;
    dataNum = sp->GetSampleNumber();
    dim = sp->GetDimension();
    myid = args->myid;
    procnum = args->procnum;
    rho = args->rho;
    x = new double[dim];
    Comm_method = args->Comm_method;
    alpha = new double[dim];
    C = new double[dim];
    msgX = new double[dim];
    for (int i = 0; i < dim; i++) {
        x[i] = 0;
        alpha[i] = 0;
        C[i] = rho;
        msgX[i] = 0;
    }
//    char *outname = new char[100];
//    sprintf(outname,"admmresult_rho.dat");
//    string resname(outname);
//    if(myid ==0)
//        of.open(outfile.c_str());

//    eps = 1e-3;
//    eps_cg = 0.1;

    int pos = 0;
    int neg = 0;
    for (int i = 0; i < sparseDataset->GetSampleNumber(); i++) {
        if (sparseDataset->GetLabel(i) > 0)
            pos++;
    }
    neg = sparseDataset->GetSampleNumber() - pos;

//    primal_solver_tol = eps*max(min(pos,neg), 1)/sparseDataset->GetSampleNumber();
    maxIteration = args->maxIteration;
    nodesOfGroup = args->nodesOfGroup;
    nears.neighborsNums = nodesOfGroup;
    nears.neighs = new int[nears.neighborsNums];
    msgbuf = new double *[procnum - 1];
    al = new double *[procnum - 1];
    for (int i = 0; i < procnum - 1; i++) {
        msgbuf[i] = new double[dim];
        al[i] = new double[dim];
        for (int j = 0; j < dim; ++j) {
            msgbuf[i][j] = 0;
            al[i][j] = 0;
        }
    }
}

ADMM::~ADMM() {
    delete[]x;
    delete[]alpha;
    delete[]C;
    delete[] msgbuf;
    delete[] al;
    delete[] msgX;
}


void ADMM::x_update(double *x, double *sumX, int dimension, double rho, SparseDataset *dataset) {
    LRTronOptimizer *lr = new LRTronOptimizer(alpha, sumX, dimension, rho, 1000, 1e-3, 0.1, dataset, &nears);
    lr->Optimize(x);
    delete lr;


//    AdmmLRFunction *func=new AdmmLRFunction(alpha,sumX,dimension,rho,dataset,&nears);
//    GradientDecentOptimizer *grad=new GradientDecentOptimizer(func,dimension);
//    grad->Optimize(x);
//    delete func;
//    delete grad;
}


void Get_gradient(const double *x, double *g, SparseDataset *dataset_) {
    int sample_num = dataset_->GetSampleNumber();
    for (int i = 0; i < sample_num; ++i) {
        double temp = Sigmoid(dataset_->GetLabel(i) * Dot(x, dataset_->GetSample(i)));
        temp = (temp - 1) * dataset_->GetLabel(i);
        const Feature *sample = dataset_->GetSample(i);
        while (sample->index != -1) {
            g[sample->index] += (sample->value * temp);
            ++sample;
        }
    }
}

void ADMM::x_update_OneOrder(double *x, double *sumX, int dimension, double c, double rho, SparseDataset *dataset) {
    double *g = new double[dimension];
    int numsofnear = nodesOfGroup - 1;
    for (int i = 0; i < dimension; i++)
        g[i] = 0;
    Get_gradient(x, g, dataset);
    for (int i = 0; i < dimension; i++) {
        x[i] -= (g[i] + alpha[i] + c * numsofnear * x[i] - c * sumX[i]) / (2 * c * numsofnear + rho);
    }
    delete[] g;
}


void ADMM::alpha_update(double *w, double *sumx) {
    int numsofneighbors = nears.neighborsNums - 1;
    for (int32_t j = 0; j < dim; j++) {
        alpha[j] += rho * ((numsofneighbors) * w[j] - sumx[j]);
    }
//      for(int i=0;i<nears.neighborsNums;i++)
//          for (int j = 0; j <dim ; ++j) {
//              al[nears.neighs[i]][j]+=-(0.5*(al[nears.neighs[i]][j]+msgbuf[nears.neighs[i]][j])+2*rho*w[j]);
//          }
}

double ADMM::LossValue() {
    if (test_data_ == NULL) {
        return 0;
    } else {
        double sum = 0;
        int sample_num = test_data_->GetSampleNumber();
        for (int i = 0; i < sample_num; ++i) {
            sum += std::log(1 + std::exp(-test_data_->GetLabel(i) * Dot(x, test_data_->GetSample(i))));
        }
        return sum;///sample_num;
    }
}


double ADMM::predict() {
    if (test_data_ == NULL) {
        return 0;
    } else {
        int counter = 0;
        int notuse = 0;
        int sample_num = test_data_->GetSampleNumber();
        for (int i = 0; i < sample_num; ++i) {
            double temp = 1.0 / (1 + exp(-1 * Dot(x, test_data_->GetSample(i))));
            if (test_data_->GetLabel(i) == 1 && temp >= 0.5) {
                ++counter;
            } else if (test_data_->GetLabel(i) == -1 && temp < 0.5) {
                ++counter;
            }
        }
        return counter * 100.0 / sample_num;
    }
}


void
comm_RingAllreduce(double *msgX, double *x, int dim, int myid, vector<int> nodes, int QuantifyPart, int Sparse_comm) {
    for (size_t i = 0; i < dim; i++) {
        msgX[i] = x[i];
    }

    if (Sparse_comm == 0)
        RingAllreduce<spar::SumOperator, double>(msgX, dim, myid, nodes, MPI_COMM_WORLD, QuantifyPart);//邻居节点 x(k)
    else if (Sparse_comm == 1)
        SparseRingAllreduce<spar::SumOperator, double>(msgX, dim, myid, nodes, MPI_COMM_WORLD);//邻居节点 x(k)
    for (size_t i = 0; i < dim; i++) {
        msgX[i] -= x[i];
    }
}

void comm_PointToPoint(double *msgX, double *x, double **tempX, int dim, neighbors nears, int myid, MPI_Status status,
                       MPI_Request *requests, MessageType mes) {
    int indextemp = 0;
    for (size_t i = 0; i < dim; i++) {
        msgX[i] = 0;
    }
    for (int i = 0; i < nears.neighborsNums; i++) {
        if (nears.neighs[i] != myid) {
            MPI_Isend(x, dim, MPI_DOUBLE, nears.neighs[i], mes, MPI_COMM_WORLD, &requests[indextemp++]);
            MPI_Irecv(tempX[i], dim, MPI_DOUBLE, nears.neighs[i], mes, MPI_COMM_WORLD, &requests[indextemp++]);
        }
    }
    MPI_Waitall(2 * (nears.neighborsNums - 1), requests, MPI_STATUSES_IGNORE);

    for (int i = 0; i < nears.neighborsNums - 1; i++) {
        for (int j = 0; j < dim; j++) {
            msgX[j] += tempX[i][j];
        }
    }
}


void ADMM::group_train(clock_t start_time) {
    bool is_slownode = false;//人工定义慢节点
    double b_time, e_time;
    double comm_btime, comm_etime, cal_btime, cal_etime;
    double comm_time, cal_time, fore_time = 0.0;
    MPI_Status status;
    MPI_Request *doublerequest;
    double **tempX;
    int32_t k = 1;
    vector<int> nodes;
    double sparseCount = 0;

    tempX = new double *[nears.neighborsNums - 1];
    doublerequest = new MPI_Request[2 * (nears.neighborsNums - 1)];

    for (int i = 0; i < nears.neighborsNums - 1; i++) {
        tempX[i] = new double[dim];
        for (int j = 0; j < dim; ++j) {
            tempX[i][j] = 0;
        }
    }
    if (myid == 0)
        printf("%3s %12s %12s %12s %12s %12s %12s\n", "#", "current", "loss", "time", "comm_time", "cal_time",
               "end_time");

    b_time = start_time;
    while (k <= maxIteration) {
        //向组生成器要求组生成 发送所处当前迭代
        MPI_Send(&k, 1, MPI_INT, procnum - 1, 1, MPI_COMM_WORLD);
        //获取生成的组
        MPI_Probe(procnum - 1, 2, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &nears.neighborsNums);
        MPI_Recv(nears.neighs, nears.neighborsNums, MPI_INT, procnum - 1, 2, MPI_COMM_WORLD, &status);
        nodes.clear();
        for (int i = 0; i < nears.neighborsNums; i++) {
            nodes.push_back(nears.neighs[i]);
        }

        //先更新x后更新al 包括两次通信 !!!!不用管update_method
        // 实际上先x后alpha，下一次x的更新可以用上一次x的通信值 实际上也是一次通信
        comm_time = 0;
        //x更新
        cal_btime = MPI_Wtime();
        if (k != 1) {
            for (int i = 0; i < dim; i++) {
                msgX[i] = alpha[i] - rho * ((nears.neighborsNums - 1) * x[i] + msgX[i]);
            }
        }
        x_update(x, msgX, sparseDataset->GetDimension(), rho, sparseDataset);
        cal_etime = MPI_Wtime();


        comm_btime = MPI_Wtime();
        //通信获得x(k+1)
        if (Comm_method == 0) {
            comm_RingAllreduce(msgX, x, dim, myid, nodes, QuantifyPart, Sparse_comm);
        } else {
            comm_PointToPoint(msgX, x, tempX, dim, nears, myid, status, doublerequest, MessageType::decencomm2);
        }
        comm_etime = MPI_Wtime();
        comm_time += (double) (comm_etime - comm_btime);
        //al更新
        alpha_update(x, msgX);


        e_time = MPI_Wtime();
        if (myid == 0) {
            double sum_time = (double) (e_time - b_time);
            cal_time = (double) (cal_etime - cal_btime);
            sparseCount = 0;
            for (int i = 0; i < dim; i++) {
                if (x[i] < 1e-5) {
                    sparseCount++;
                }
            }
            printf("%3d %12f %12f %12f %12f %12f %12f %12f\n", k, predict(), LossValue(), sum_time - fore_time,
                   comm_time, cal_time, sum_time, sparseCount / dim);
            fore_time = sum_time;
            sum_comm += comm_time;
        }
        k++;
    }
    delete[]tempX;

}






//void comm_PointToPoint(double *msgX,double *x,double **tempX,int dim,neighbors nears,int myid,MPI_Status status,MPI_Request *requests,MessageType mes)
//{
//    int indextemp=0;
//    for(size_t i=0;i<dim;i++)
//    {
//        msgX[i]=0;
//    }
//    for(int i=0;i<nears.neighborsNums;i++)
//    {
//        if(nears.neighs[i]!=myid)
//        {
//            MPI_Isend(x,dim,MPI_DOUBLE,nears.neighs[i],mes,MPI_COMM_WORLD,&requests[indextemp++]);
//            MPI_Irecv(tempX[i],dim,MPI_DOUBLE,nears.neighs[i],mes,MPI_COMM_WORLD,&requests[indextemp++]);
//        }
//    }
////    for(int i=0;i<nears.neighborsNums-1;i++)
////    {
//////        MPI_Probe(MPI_ANY_SOURCE, mes , MPI_COMM_WORLD, &status);
////        MPI_Irecv(tempX[i],dim,MPI_DOUBLE,status.MPI_SOURCE,mes,MPI_COMM_WORLD,&requests[indextemp++]);
////    }
//    MPI_Waitall(2*(nears.neighborsNums-1), requests, MPI_STATUSES_IGNORE);
////    MPI_Waitall(nears.neighborsNums-1, sendrequest, MPI_STATUSES_IGNORE);
////    for(int i=0;i<nears.neighborsNums;i++)
////    {
////        if(nears.neighs[i]!=myid)
////        {
////            MPI_Send_init(x, dim, MPI_DOUBLE, nears.neighs[i] ,mes, MPI_COMM_WORLD, &requests[indextemp++]);//创建持续发送对象，初始化发射请求
////            MPI_Recv_init(tempX[i], dim, MPI_DOUBLE, nears.neighs[i] ,mes, MPI_COMM_WORLD, &requests[indextemp++]);//创建持续接收请求
////        }
////    }
////    MPI_Startall(2*(nears.neighborsNums-1), requests);
////    MPI_Waitall(2*(nears.neighborsNums-1), requests, status1);
//
//    for(int i=0;i<nears.neighborsNums-1;i++)
//    {
//        for(int j=0;j<dim;j++)
//        {
//            msgX[j]+=tempX[i][j];
//        }
//    }
//}



//void ADMM::group_train(clock_t start_time)
//{
//    //人工定义慢节点
//    bool is_slownode= false;
//    double b_time,e_time;
//    double comm_btime,comm_etime,cal_btime,cal_etime;
//    double comm_time,cal_time,fore_time=0.0;
////    wait_time = 0;
//    MPI_Status status;
////    MPI_Request *sendrequest;
////    MPI_Request *recvrequest;
//    MPI_Request *doublerequest;
//    double **tempX;
//    int32_t k = 1;
//    vector<int>nodes;
//
//    tempX=new double*[nears.neighborsNums-1];
//    doublerequest=new MPI_Request[2*(nears.neighborsNums-1)];
////    sendrequest=new MPI_Request[nears.neighborsNums-1];
////    recvrequest=new MPI_Request[nears.neighborsNums-1];
//
//    for(int i=0;i<nears.neighborsNums-1;i++)
//    {
//        tempX[i]=new double[dim];
//        for (int j = 0; j <dim ; ++j) {
//            tempX[i][j]=0;
//        }
//    }
//
////    //if(DynamicGroup)
////    {
////        srand((int)time(0));
////        int flag=random(procnum-1);
////        if(flag<4)
////            is_slownode= true;
////    }
////    if(myid==0)
////        is_slownode= true;
//
//    if(myid == 0)
//        printf("%3s %12s %12s %12s %12s %12s %12s\n","#", "current","loss","time","comm_time","cal_time","end_time");
//
//    b_time=start_time;
//    while(k <= maxIteration)
//    {
//        //向组生成器要求组生成 发送所处当前迭代
//        MPI_Send(&k, 1, MPI_INT, procnum-1 ,1, MPI_COMM_WORLD);
//        //获取生成的组
//        MPI_Probe(procnum-1, 2 , MPI_COMM_WORLD, &status);
//        MPI_Get_count(&status, MPI_INT, &nears.neighborsNums);
//        MPI_Recv(nears.neighs, nears.neighborsNums, MPI_INT, procnum-1,2, MPI_COMM_WORLD, &status);
//        nodes.clear();
//        for(int i=0;i<nears.neighborsNums;i++)
//        {
//            nodes.push_back(nears.neighs[i]);
//        }
//
//        if(Update_method==0||Update_method==1) //先更新x后更新al 包括两次通信 !!!!不用管update_method
//        {
//            //实际上先x后alpha，下一次x的更新可以用上一次x的通信值 实际上也是一次通信
//            comm_time=0;
//            //x更新
//            cal_btime=MPI_Wtime();
//            if(k!=1)
//            {
//                for(int i=0;i<dim;i++)
//                {
//                    msgX[i]=alpha[i]-rho*((nears.neighborsNums-1)*x[i]+msgX[i]);
//                }
//            }
////            x_update_OneOrder(x,msgX,dim,rho,rho,sparseDataset);
//            x_update(x,msgX,sparseDataset->GetDimension(),rho,sparseDataset);
////            cal_etime=MPI_Wtime();
////            if(is_slownode)
////            {
////                double temp= (double)(cal_etime-cal_btime);
////                sleep(temp);
////            }
//            cal_etime=MPI_Wtime();
//
//
//            comm_btime=MPI_Wtime();
//            //通信获得x(k+1)
//            if(Comm_method==0)
//            {
//                comm_RingAllreduce(msgX,x,dim,myid,nodes,QuantifyPart,Sparse_comm);
//            }
//            else
//            {
//                comm_PointToPoint(msgX, x,tempX,dim,nears,myid,status,doublerequest,MessageType::decencomm2);
//            }
//            comm_etime=MPI_Wtime();
//            comm_time+=(double)(comm_etime-comm_btime);
//            //al更新
//            alpha_update(x,msgX);
//        }
//        else if(Update_method==10086)// 先更新 al 后更新x 只需要一次通信
//        {
//            comm_btime=MPI_Wtime();
//            if(Comm_method==0)
//            {
//                comm_RingAllreduce(msgX,x,dim,myid,nodes,QuantifyPart,Sparse_comm);
//            }
//            else
//            {
//                comm_PointToPoint(msgX,x,tempX,dim,nears,myid,status,doublerequest,MessageType::decencomm1);
//            }
//            comm_etime=MPI_Wtime();
//
//            comm_time=(double)(comm_etime-comm_btime);
//
//            alpha_update(x,msgX);
//
//            cal_btime=MPI_Wtime();
//            for(int i=0;i<dim;i++)
//            {
//                msgX[i]=alpha[i]-rho*((nears.neighborsNums-1)*x[i]+msgX[i]);
//            }
//            x_update(x,msgX,sparseDataset->GetDimension(),rho,sparseDataset);
//            cal_etime=MPI_Wtime();
//
////            if(is_slownode)
////            {
////                int temp= (double)(cal_etime-cal_btime);;
////            }
//        }
//        e_time=MPI_Wtime();
//        if(myid==0)
//        {
//            double sum_time=(double)(e_time-b_time);
//            cal_time=(double)(cal_etime-cal_btime);
//            printf("%3d %12f %12f %12f %12f %12f %12f\n",k, predict(),LossValue(),sum_time-fore_time,comm_time,cal_time,sum_time);
//            fore_time=sum_time;
//            sum_comm+=comm_time;
//        }
//        k++;
//    }
//    delete []tempX;
////    delete []sendrequest;
////    delete []recvrequest;
//}
