//
// Created by cluster on 2020/10/14.
//

#include "utils.h"
#include "mpi.h"
#include <vector>

using namespace std;


utils::utils(int nums) {
    repeatIter = 3;
    this->repeatIter = nums;
}

///
/// \param data  单次迭代的节点排序
/// \param GroupNum  组数
/// \param Group1   交换节点的组1
/// \param Group2   交换节点的组2
/// \param part     交换哪一个部分 0 前一半  1 后一半
vector<int> utils::exchangeElement(vector<int> data, int GroupNum, int Group1, int Group2, int part) {
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
vector<vector<int>> utils::divideGroup(vector<int> nodes, int groupNums) {
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
                temp = exchangeElement(returnAns[iter - 1], groupNums, i, (exchange * u + 2) % groupNums, (part++) % 2);
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
int utils::position(double *vec, int size, int index) {
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
vector<int> utils::findFastNodes(double *time, vector<int> group, int node, int numsofGrup, int size) {
    vector<int> fastnodes;
    //选择排序思想
    for (int i = 0; i < size - 1; i++) {
        int index = i;
        for (int j = i + 1; j < size; j++) {
            if (time[j] < time[index]) {
                index = j;
            }
        }
        int temp;
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
void utils::changeGroup(vector<vector<int>> &data, int node, vector<int> fastVec, int numsOfgroup, int iter) {
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


void utils::MasterNodes(int procnum, int nodesOfGroup, int DynamicGroup, int maxIteration) {
    double *node_beforeTime;
    double *node_afterTime;
    double *node_caltime;//用于分组器 记录每个节点单次的计算时间 时间长的判断为慢节点
    vector<int> nodes;
    vector<vector<int>> Group;
    //接受组生成通信请求
    int nodetemp;
    int iter = 0;
    MPI_Status status;
    int iterTemp = 0;
    int c = 1;
    int *sendNodes;

    node_caltime = new double[procnum - 1];
    node_beforeTime = new double[procnum - 1];
    node_afterTime = new double[procnum - 1];
    sendNodes = new int[nodesOfGroup];
    for (int i = 0; i < procnum - 1; i++) {
        node_caltime[i] = 0.0;
        node_beforeTime[i] = 0.0;
        node_afterTime[i] = 0.0;
    }
    //预定义分组规则
    for (int i = 0; i < procnum - 1; i++)
        nodes.push_back(i);
    Group = divideGroup(nodes, (procnum - 1) / nodesOfGroup);

    while (true) {
        MPI_Probe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
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
                    int pos = position(node_caltime, procnum - 1, nodetemp);
                    if (pos < (procnum - 1) / 4 || pos >= (procnum - 1) / 4 * 3)//中间的不用管 分四份 第一份和最后一份需要给他分配快节点
                    {
                        cout << "iter " << iter << " change Group !!!" << endl;
                        //修改Group[iter-1] 修改nodetemp所在组其他节点为快节点。
                        vector<int> fastnodex = findFastNodes(node_caltime, Group[(iter - 1) % repeatIter], nodetemp,
                                                              nodesOfGroup, procnum - 1);
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
        //输出组/
//        if(status.MPI_SOURCE==0)
//        {
//            cout<<"iter "<<iter<<": ";
//            for(int i=0;i<procnum-1;i++)
//                cout<<tempVec[i]<<" ";
//            cout<<endl;
//        }
        int u = 0;
        for (u = 0; u < procnum; u++) {
            if (tempVec[u] == nodetemp) {
                break;
            }
        }
        int tempIndex = 0;
        //cout<<nodetemp<<":";
        for (int v = u / nodesOfGroup * nodesOfGroup; v < (u / nodesOfGroup + 1) * nodesOfGroup; v++) {
            //cout<<tempVec[v]<<" ";
            sendNodes[tempIndex++] = tempVec[v];
        }
        //cout<<endl;
        MPI_Send(sendNodes, nodesOfGroup, MPI_INT, nodetemp, 2, MPI_COMM_WORLD);
        c++;
        if (c > maxIteration * (procnum - 1)) {
            break;
        }
    }
    delete[] node_caltime;
    delete[] node_beforeTime;
    delete[] node_afterTime;
    delete[]sendNodes;

}