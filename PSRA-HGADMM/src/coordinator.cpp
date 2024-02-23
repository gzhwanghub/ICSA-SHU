#include <vector>
#include <algorithm>
#include <mpi.h>

#ifdef _WIN32

#include <Windows.h>

#else

#include <unistd.h>

#endif

#include "../include/logging/simple_logging.h"
#include "../include/internal/common.h"
#include "../include/coordinator.h"

namespace spar {

    Coordinator *Coordinator::GetInstance() {
        //设计成单例模式
        static Coordinator coordinator;
        return &coordinator;
    }

    Coordinator::Coordinator() {
        int flag;
        MPI_Initialized(&flag);
        CHECK_EQ(flag, 1) << "请在调用spar::Init()之后启动Coordinator";
        MPI_Comm_rank(MPI_COMM_WORLD, &id_);
        MPI_Comm_size(MPI_COMM_WORLD, &worker_number_);
        // Worker的数量为总进程数减一
        --worker_number_;
        // Coordinator的id应该为总进程数减一
        CHECK_EQ(id_, worker_number_) << "不要在Worker节点上启动Coordinator";

    }

    void Coordinator::Run() {

        //cout<<"coordinator run"<<endl;
        //首先进行进程分组，将在同一个主机上的进程分为同一组
        CreateGroup();
        //cout<<"creat group "<<endl;
        //持续等待每个机器的leader请求分组通信
        MasterNodes();


    }

    void Coordinator::CreateGroup() {
        int count = 0;
        char name[50];
        int name_length;
        std::string hostname;
        // 主机名->{id1, id2, id3,...}，其中第一个进程为leader,每个主机是多核所以可以有多个进程
        std::map<std::string, std::vector<int>> groups;
        MPI_Status status;
        // 等待所有进程将自己所在的主机名发送过来以便进行分组
        while (count < worker_number_) {
            MPI_Probe(MPI_ANY_SOURCE, MessageType::kCreateGroup1, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_CHAR, &name_length);
            MPI_Recv(name, name_length, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            hostname.assign(name, name_length);
            groups[hostname].push_back(status.MPI_SOURCE);
            ++count;
        }
        /*将按机器分好组的节点发送到每台机器的工作进程上*/
        for (auto map_it = groups.begin(); map_it != groups.end(); ++map_it) {
            // 将每个组进行排序，id最小的作为leader
            std::sort(map_it->second.begin(), map_it->second.end());
            leader_list.push_back(map_it->second.front());
            //cout<<"group: ";
            for (auto list_it = map_it->second.begin(); list_it != map_it->second.end(); ++list_it) {
                MPI_Send(map_it->second.data(), map_it->second.size(), MPI_INT, *list_it, MessageType::kCreateGroup1,
                         MPI_COMM_WORLD);
            }

        }
        // 将leader列表发送给所有leader
        std::sort(leader_list.begin(), leader_list.end());
        for (auto it = leader_list.begin(); it != leader_list.end(); ++it) {
            MPI_Send(leader_list.data(), leader_list.size(), MPI_INT, *it, MessageType::kCreateGroup2, MPI_COMM_WORLD);
        }
        /*for (int i = 0; i < leader_list.size(); ++i) {
            cout<<leader_list[i]<<" ";
        }
        cout<<endl;*/

    }

///
/// \param data  单次迭代的节点排序
/// \param GroupNum  组数
/// \param Group1   交换节点的组1
/// \param Group2   交换节点的组2
/// \param part     交换哪一个部分 0 前一半  1 后一半
    vector<int> Coordinator::exchangeElement(vector<int> data, int GroupNum, int Group1, int Group2, int part) {
        int nums = data.size() / GroupNum;
        vector<int> vec1, vec2;
        for (int i = nums * Group1; i < nums * (Group1 + 1); i++)
            vec1.push_back(data[i]);
        for (int i = nums * Group2; i < nums * (Group2 + 1); i++)
            vec2.push_back(data[i]);
        int index = 0;
        if (part == 1)
            index = nums / 2;
        for (int i = nums * Group1 + (nums / 2) * part; i < nums * (Group1 + 1) - (nums / 2) * (1 - part); i++) {
            data[i] = vec2[index++];
        }
        if (part == 1)
            index = nums / 2;
        else
            index = 0;
        for (int i = nums * Group2 + (nums / 2) * part; i < nums * (Group2 + 1) - (nums / 2) * (1 - part); i++) {
            data[i] = vec1[index++];
        }
        return data;
    }

///
/// \param nodes 节点列表
/// \param groupNums 分组数
/// \return
    vector<vector<int>> Coordinator::divideGroup(vector<int> nodes, int groupNums) {
        vector<vector<int>> returnAns;
        int nodesNums = nodes.size();//总结点数
//    int numsOfGroup=nodesNums/groupNums;//一般设置为单个或多个机器节点开的进程节点总数
        vector<int> tempVec;
        int iter = 0;
        for (int i = 0; i < nodes.size(); i++) {
            tempVec.push_back(nodes[i]);
        }
        returnAns.push_back(tempVec);
        tempVec.clear();
        ++iter;
        int part = 0;
        int exchange = groupNums / 2;
        int u = 0;
        while (iter < repeatIter) {
            vector<int> temp;
            if ((exchange * u + 1) % groupNums == 0)
                temp = exchangeElement(returnAns[iter - 1], groupNums, 0, (exchange * u + 2) % groupNums, (part++) % 2);
            else
                temp = exchangeElement(returnAns[iter - 1], groupNums, 0, (exchange * u + 1) % groupNums, (part++) % 2);
            for (int i = 2; i < groupNums - 1; i++) {
                if ((exchange * u + i + 1) % groupNums == i)
                    temp = exchangeElement(returnAns[iter - 1], groupNums, i, (exchange * u + 2) % groupNums,
                                           (part++) % 2);
                else
                    temp = exchangeElement(temp, groupNums, i, (exchange * u + i + 1) % groupNums, (part++) % 2);
            }
            returnAns.push_back(temp);
            iter++;
            if (groupNums != 2)
                part++;
            u++;
        }
        return returnAns;
    }

/// 判断index节点在vec处于第几小（不包括0）,越小说明越快
/// \param vec
/// \param index
/// \return
    int Coordinator::position(double *vec, int size, int index) {
        int ans = 0;
        for (int i = 0; i < size; i++) {
            if (vec[i] != 0) {
                if (vec[i] < vec[index])
                    ans++;
            }
        }
        return ans;
    }

/// 找到快节点
/// \param time 时间 时间的下标 即快节点的id
/// \param group
/// \param node 请求节点 需要排除
/// \param node 需要找到numsofGrup-1个快节点
/// \return 需要返回在当前迭代的group的下标
    vector<int> Coordinator::findFastNodes(double *time, vector<int> group, int node, int numsofGrup, int size) {
        vector<int> fastnodes;
        //选择排序思想
        for (int i = 0; i < size - 1; i++) {
            int index = i;
            /*找出时间最小的下标*/
            for (int j = i + 1; j < size; j++) {
                if (time[j] < time[index]) {
                    index = j;
                }
            }
            int temp;
            /*将最小值放置在数组头部*/
            temp = time[index];
            time[index] = time[i];
            time[i] = temp;
            if (index == node)
                continue;
            for (int j = 0; j < group.size(); j++) {
                if (index == group[j]) {
                    fastnodes.push_back(j);
                    break;
                }
            }
            if (fastnodes.size() == numsofGrup - 1)
                break;
        }
        return fastnodes;

    }

#include <iostream>

    using namespace std;

/// 替换分组中的节点为慢快节点
/// \param vec 分组向量
/// \param node
/// \param fastVec  快节点的下标
/// \param numsOfgroup 每组有多少个节点
    void Coordinator::changeGroup(vector<vector<int>> &data, int node, vector<int> fastVec, int numsOfgroup, int iter) {
        vector<int> vec;
        vec = data[(iter - 1) % repeatIter];
        for (int i = 0; i < data[(iter - 1) % repeatIter].size(); i++)
            vec.push_back(data[(iter - 1) % repeatIter][i]);
        int index = 0;//node所处下标
        for (index; index < vec.size(); index++) {
            if (vec[index] == node)
                break;
        }
        int j = 0;
        for (int i = index / numsOfgroup * numsOfgroup; i < (1 + index / numsOfgroup) * numsOfgroup; i++) {
            //交换
            if (i != index) {
                int temp = vec[i];
                vec[i] = vec[fastVec[j]];
                vec[fastVec[j++]] = temp;
            }
        }
        data.push_back(vec);
    }

    void Coordinator::MasterNodes() {
        double *node_beforeTime;
        double *node_afterTime;
        double *node_caltime;//用于分组器 记录每个节点单次的计算时间 时间长的判断为慢节点
        vector<int> nodes;
        vector<vector<int>> Group;
        int procnum = leader_list.size();
        int stopped = 1;

        //接受组生成通信请求
        int nodetemp;
        int iter = 0;
        MPI_Status status;
        int iterTemp = 0;
        int *sendNodes;
        int c = 0;
        node_caltime = new double[procnum];
        node_beforeTime = new double[procnum];
        node_afterTime = new double[procnum];
        sendNodes = new int[nodesOfGroup];
        for (int i = 0; i < procnum; i++) {
            node_caltime[i] = 0.0;
            node_beforeTime[i] = 0.0;
            node_afterTime[i] = 0.0;
        }
        //预定义分组规则
        for (int i = 0; i < procnum; i++)
            nodes.push_back(leader_list[i]);
        Group = divideGroup(nodes, (procnum) / nodesOfGroup);

        //循环等待分组请求
        while (true) {
            //cout<<"wait for request group"<<endl;

            MPI_Probe(MPI_ANY_SOURCE,MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	    if(status.MPI_TAG==MessageType::kTerminateCommand)
		break;
            nodetemp = status.MPI_SOURCE;
            MPI_Recv(&iter, 1, MPI_INT, nodetemp, 1, MPI_COMM_WORLD, &status);
            vector<int> tempVec;
            if (DynamicGroup == 1) {
                //根据每次迭代第一次发送的节点 根据上次的计算时间重新修改部分分组方式
                //根据第一个节点 为快节点分配快节点 为慢节点分配快节点。 第一次迭代不用管
                //前3次迭代不用管
                if (iter > repeatIter && iter > iterTemp) {
                    iterTemp = iter;
                    if ((iter - 1) % repeatIter == 0) {
//                        vector<int> grouptemp;
//                        grouptemp=Group[0];//分组规则的第1个是节点内的通信，速度快，所以不变
                        Group.push_back(Group[0]);
                    } else {
                        //只有快节点和慢节点需要处理
                        //修改每组一半的节点
                        int pos = position(node_caltime, procnum, nodetemp);
                        if (pos < (procnum) / 4 || pos >= (procnum) / 4 * 3)//中间的不用管 分四份 第一份和最后一份需要给他分配快节点
                        {
                            //cout << "iter " << iter << " change Group !!!" << endl;
                            //修改Group[iter-1] 修改nodetemp所在组其他节点为快节点。
                            vector<int> fastnodex = findFastNodes(node_caltime, Group[(iter - 1) % repeatIter],
                                                                  nodetemp, nodesOfGroup, procnum);
                            changeGroup(Group, nodetemp, fastnodex, nodesOfGroup, iter);
                            //Group.push_back(newvectemp);
                        } else {
                            Group.push_back(Group[(iter - 1) % repeatIter]);
                        }
                    }
                }
                //更新所有节点的迭代间隔信息
                node_beforeTime[nodetemp] = node_afterTime[nodetemp];
                node_afterTime[nodetemp] = (double) (clock()) / CLOCKS_PER_SEC;
                node_caltime[nodetemp] = node_afterTime[nodetemp] - node_beforeTime[nodetemp];
                tempVec = Group[iter - 1];
                //tempVec=Group[(iter-1)%3];
            } else {
                tempVec = Group[(iter - 1) % repeatIter];
            }

            int u = 0;
            for (u = 0; u < procnum; u++) {
                if (tempVec[u] == nodetemp) {
                    break;
                }
            }
            int tempIndex = 0;
            // cout<<nodetemp<<":";
            for (int v = u / nodesOfGroup * nodesOfGroup; v < (u / nodesOfGroup + 1) * nodesOfGroup; v++) {
                //cout<<tempVec[v]<<" ";
                sendNodes[tempIndex++] = tempVec[v];
            }

            MPI_Send(sendNodes, nodesOfGroup, MPI_INT, nodetemp, 2, MPI_COMM_WORLD);
           // c++;
            //if (c > max_iterations_ * (procnum)) {
              //  break;
           // }


        }
        delete[] node_caltime;
        delete[] node_beforeTime;
        delete[] node_afterTime;
        delete[]sendNodes;

    }
}
