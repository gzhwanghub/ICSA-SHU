#ifndef SPARSEALLREDUCE_COMMUNICATOR_H
#define SPARSEALLREDUCE_COMMUNICATOR_H

#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <map>


#ifdef _WIN32

#include <Windows.h>

#else

#include <unistd.h>
#endif

#include "logging/simple_logging.h"
#include "internal/common.h"
#include "internal/group_allreduce.h"
#include "internal/simple_broadcast.h"
#include "internal/communicator_singleton.h"
#include "other/neighbors.h"

namespace spar {

    template<class O, class T>
    class Communicator {
    public:
        static Communicator *GetInstance();

        void Run();

        int Get_leaderID() { return member_list_.front(); };

        int Get_MyID() { return id_; };

       void Creat_inter_Group(int k);

        template<class O2, class T2>
        void SyncAllreduce(T2 *buffer, int count);


        template<class O1,class T1>
          void AllReduce(T1 *buffer, int count);
    private:
        int id_;
        int coordinator_id_;
        int worker_number_;
        std::vector<int> inter_nodes;
        //储存用户调用AsynAllreduce时提供的数组地址
        T *user_buffer_;
        //在调用回调函数时传递的参数
        void *callback_parameter_;
        //进程所在组的所有成员的id
        std::vector<int> member_list_;
        //所有Leader进程的id
        std::vector<int> leader_list_;
        //所有Worker进程的id
        std::vector<int> worker_list_;
        //邻居节点包括本节点
        neighbors nears;

        //物理机之间通信
        Communicator();

        void CreateGroup();

    };

    template<class O, class T>
    Communicator<O, T> *Communicator<O, T>::GetInstance() {
        //设计成单例模式
        static Communicator communicator;

        CommunicatorSingleton::Set(&communicator);
        return &communicator;
    }

    template<class O, class T>
    Communicator<O, T>::Communicator() {
        int flag;
        MPI_Initialized(&flag);
        CHECK_EQ(flag, 1) << "请在调用spar::Init()之后启动Communicator";
        MPI_Comm_rank(MPI_COMM_WORLD, &id_);
        MPI_Comm_size(MPI_COMM_WORLD, &worker_number_);
        // Worker的数量为总进程数减一
        --worker_number_;
        CHECK_NE(id_, worker_number_) << "不要在Coordinator节点上启动Communicator";
        coordinator_id_ = worker_number_;
        nears.neighborsNums=12;
        nears.neighs = new int [nears.neighborsNums];
        /*calculator_state_ = CalculatorState::kCalculating;
        communicator_state_ = CommunicatorState::kWaitToRun;*/
    }


    template<class O, class T>
    void Communicator<O, T>::Run() {
        //首先创建进程分组
        CreateGroup();
        //std::cout<<"communicate create group"<<std::endl;
    }


    /*这部分是worker的工作逻辑*/
    template<class O, class T>
    void Communicator<O, T>::CreateGroup() {
        char name[50];
        int name_length;
        //把本进程的主机名发送给Coordinator，Coordinator会以此来进行分组
        MPI_Get_processor_name(name, &name_length);
        MPI_Send(name, name_length, MPI_CHAR, coordinator_id_, MessageType::kCreateGroup1, MPI_COMM_WORLD);
        MPI_Status status;
        int list_length;
        /*动态创建接收数组的缓存大小*/
        MPI_Probe(coordinator_id_, MessageType::kCreateGroup1, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &list_length);
        int *recv_buf = new int[list_length];
        MPI_Recv(recv_buf, list_length, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //接收本组的所有成员id
        member_list_.assign(recv_buf, recv_buf + list_length);
        delete[] recv_buf;
        //组中第一个Worker作为Leader
        int leader_id_ = member_list_.front();
        if (id_ == leader_id_) {
            //Leader还需要接收其他所有Leader的id列表
            MPI_Probe(coordinator_id_, MessageType::kCreateGroup2, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &list_length);
            recv_buf = new int[list_length];
            MPI_Recv(recv_buf, list_length, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            leader_list_.assign(recv_buf, recv_buf + list_length);
            delete[] recv_buf;
        }
        for (int i = 0; i < worker_number_; ++i) {
            worker_list_.push_back(i);
        }
    }

    template<class O, class T>
    void Communicator<O,T>::Creat_inter_Group(int k) {

        inter_nodes.clear();

        MPI_Status status;
        //向组生成器要求组生成 发送所处当前迭代
        MPI_Send(&k, 1, MPI_INT, coordinator_id_, 1, MPI_COMM_WORLD);
        //获取生成的组
        MPI_Probe(worker_number_, 2, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &nears.neighborsNums);
        MPI_Recv(nears.neighs, nears.neighborsNums, MPI_INT, worker_number_, 2, MPI_COMM_WORLD, &status);

        //std::cout<<"creat_inter_group"<<std::endl;
        for (int i = 0; i < nears.neighborsNums; i++) {

            inter_nodes.push_back(nears.neighs[i]);
        }
    }


    template<class O, class T>
    template<class O2, class T2>
    void Communicator<O, T>::SyncAllreduce(T2 *buffer, int count) {
        int leader_id = member_list_.front();
        IntragroupReduce<O>(buffer, count, id_, member_list_, MPI_COMM_WORLD);
        if (id_ == leader_id) {
           /* for (int i = 0; i < inter_nodes.size(); ++i) {
                std::cout<<inter_nodes[i]<<" ";
            }
            std::cout<<std::endl;*/
            SparsePsAllreduce<O>(buffer, count, id_,inter_nodes, MPI_COMM_WORLD);
        }
        IntragroupBroadcast(buffer, count, id_, member_list_, MPI_COMM_WORLD);

    }
    template<class O, class T>
    template<class O1,class T1>
    void Communicator<O, T>::AllReduce(T1 *buffer, int count){
        SparseGroupPsAllreduce<O1>(buffer, count, id_, member_list_, leader_list_, MPI_COMM_WORLD);
    }

}
#endif //SPARSEALLREDUCE_COMMUNICATOR_H
