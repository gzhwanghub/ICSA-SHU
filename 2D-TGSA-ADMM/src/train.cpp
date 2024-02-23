/*************************************************************************
    > File Name: train.cpp
    > Author: Guozheng Wang
    > Date: 2021-7-16
 ************************************************************************/

#include <mpi.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include "sync_admm.h"
#include "collective.h"
#include "tron.h"
#include "prob.h"
#include<cstring>
using namespace std;

int main(int argc, char ** argv)
{
    int myid, procnum;
    double begin_time,end_time;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procnum);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	/*write a optimizer selection*/
    Properties properties("../conf/admm.conf");  //读取默认的配置文件
	string train_data_path = properties.GetString("train_data_path");
    string test_data_path = properties.GetString("test_data_path");
    int maxdim;
	maxdim = properties.GetInt("dimension");
    if(myid==0)
        cout<<"dim="<<maxdim<<endl;
    char filename[100];
    problem *prob;
    sprintf(filename,train_data_path.c_str(), procnum, myid);
    string dataname(filename);
    string inputfile = dataname;
    prob = new problem(inputfile.c_str());
    prob->n = maxdim;
    args_t *args = new args_t(myid, procnum,properties);
    Collective *collective = new Collective(args, prob);
    begin_time=MPI_Wtime(); //clock();
    ADMM admm(args, prob, test_data_path, collective);
    admm.train();
    end_time=MPI_Wtime();//clock();
    //admm.draw();
    if(myid == 0)
    {    cout << "Total train time is " <<  (double)(end_time-begin_time)<< " second." << endl;
    }
    MPI_Finalize();
}
