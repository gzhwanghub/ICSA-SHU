/*************************************************************************
    > File Name: group_strategy.cpp
    > Description: Group exchange
    > Author: Xin Huang
    > Created Time: 2020-10-14
 ************************************************************************/

#include "include/group_strategy.h"
#include "mpi.h"
#include <vector>
#include <iostream>

using namespace std;

GroupStrategy::GroupStrategy(int nums) {
    repeatIter = 3;
    this->repeatIter = nums;
}

/// \param data  Node sorting in a single iteration.
/// \param GroupNum  Group count.
/// \param Group1   Group 1 of switching nodes.
/// \param Group2   Group 2 of switching nodes.
/// \param part     Which part 0, the first half and the second half are exchanged?
vector<int> GroupStrategy::exchangeElement(vector<int> data, int GroupNum, int Group1, int Group2, int part) {
    int nums = data.size() / GroupNum;// Number of worker in each group.
    // The packet vector to exchange.
    vector<int> vec1, vec2;
    for (int i = nums * Group1; i < nums * (Group1 + 1); i++)
        vec1.push_back(data[i]);
    for (int i = nums * Group2; i < nums * (Group2 + 1); i++)
        vec2.push_back(data[i]);
    int index = 0;
    // Grouped parts: 0 is the first half and 1 is the second half.
    if (part == 1)
        index = nums / 2;
    for (int i = nums * Group1 + (nums / 2) * part; i < nums * (Group1 + 1) - (nums / 2) * (1 - part); i++) {
        data[i] = vec2[index++]; // Switch worker in groups, and switch before and after according to part.
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
/// \param nodes Node list.
/// \param groupNums Grouping number.
/// \return
vector<vector<int>> GroupStrategy::divideGroup(vector<int> nodes, int groupNums) {
    vector<vector<int>> returnAns;
    int nodesNums = nodes.size(); // Total number of nodes.
//    int numsOfGroup=nodesNums/groupNums; // Generally, it is set to the total number of process nodes opened for single or multiple machine nodes.
    vector<int> tempVec;// Temporary worker vector.
    int iter = 0;
    for (int i = 0; i < nodes.size(); i++) {
        tempVec.push_back(nodes[i]);
    }
    returnAns.push_back(tempVec);
    tempVec.clear(); // Empty array.
    ++iter;
    int part = 0;
    int exchange = groupNums / 2;
    int u = 0;
    // Multiple packet exchanges to generate initialization packets.
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

/// Judge the smallest index node in vec (excluding 0), and the smaller it is, the faster it will be.
/// \param vec
/// \param index
/// \return
int GroupStrategy::position(double *vec, int size, int index) {
    int ans = 0;
    for (int i = 0; i < size; i++) {
        if (vec[i] != 0) {
            if (vec[i] < vec[index])
                ans++;
        }
    }
    return ans;
}

/// Find the fast node.
/// \param time The subscript of time is the id of the fast node.
/// \param group
/// \param node The requesting node needs to be excluded.
/// \param node Need to find numsofGrup-1 fast nodes.
/// \return Need to return the subscript of the group in the current iteration.
vector<int> GroupStrategy::findFastNodes(double *time, vector<int> group, int node, int numsofGrup, int size) {
    vector<int> fastnodes;
    // Choose the sorting idea, sort the worker according to the computing speed of nodes, and find out the fast nodes.
    for (int i = 0; i < size - 1; i++) {
        int index = i;
        for (int j = i + 1; j < size; j++) {
            if (time[j] < time[index]) {
                index = j;
            }
        }
        // Exchange the time corresponding to I and index.
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

/// Replace the nodes in the grouping with slow and fast nodes.
/// \param vec Grouping vector.
/// \param node
/// \param fastVec  Subscript of fast node.
/// \param numsOfgroup How many nodes are there in each group?
void GroupStrategy::changeGroup(vector<vector<int>> &data, int node, vector<int> fastVec, int numsOfgroup, int iter) {
    vector<int> vec;
    vec = data[(iter - 1) % repeatIter];
    for (int i = 0; i < data[(iter - 1) % repeatIter].size(); i++)
        vec.push_back(data[(iter - 1) % repeatIter][i]);
    int index = 0;
    for (index; index < vec.size(); index++) {
        if (vec[index] == node)
            break;
    }
    int j = 0;
    for (int i = index / numsOfgroup * numsOfgroup; i < (1 + index / numsOfgroup) * numsOfgroup; i++) {
        // Exchange
        if (i != index) {
            int temp = vec[i];
            vec[i] = vec[fastVec[j]];
            vec[fastVec[j++]] = temp;
        }
    }
    data.push_back(vec);
}

void GroupStrategy::MasterNodes(int procnum, int nodesOfGroup, int DynamicGroup, int maxIteration) {
    double *node_beforeTime;
    double *node_afterTime;
    double *node_caltime; // It is judged as a slow node for the packet recorder to record a single calculation time of each node.
    vector<int> nodes;
    vector<vector<int>> Group;
    // Accept the communication request generated by the group.
    int nodetemp;
    int iter = 0;
    MPI_Status status;
    int iterTemp = 0;
    int c = 1;
    int *sendNodes;
    // Record the calculation time, and judge the slow nodes by the calculation time.
    node_caltime = new double[procnum - 1];
    node_beforeTime = new double[procnum - 1];
    node_afterTime = new double[procnum - 1];
    sendNodes = new int[nodesOfGroup];
    for (int i = 0; i < procnum - 1; i++) {
        node_caltime[i] = 0.0;
        node_beforeTime[i] = 0.0;
        node_afterTime[i] = 0.0;
    }
    // Predefine grouping rules and initialize grouping.
    for (int i = 0; i < procnum - 1; i++)
        nodes.push_back(i);
    Group = divideGroup(nodes, (procnum - 1) / nodesOfGroup);
    while (true) {
        MPI_Probe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        nodetemp = status.MPI_SOURCE;
        MPI_Recv(&iter, 1, MPI_INT, nodetemp, 1, MPI_COMM_WORLD, &status);
        vector<int> tempVec;
        if (DynamicGroup == 1) {
            // Re-modify some grouping methods according to the node that sent for the first time in each iteration and the last calculation time.
            // Assign fast nodes to fast nodes according to the first node and assign fast nodes to slow nodes. Never mind the first iteration.
            // Never mind the first 3 iterations.
            if (iter > repeatIter && iter > iterTemp) {
                iterTemp = iter;
                if ((iter - 1) % repeatIter ==
                    0) { // Every iteration of repeatIter, return to the original grouping form.
                    Group.push_back(Group[0]);
                } else {
                    // Only fast nodes and slow nodes need to be processed.
                    // Modify half the nodes in each group.
                    int pos = position(node_caltime, procnum - 1,
                                       nodetemp); // Find the number of fast worker by calculating the time.
                    if (pos < (procnum - 1) / 4 || pos >= (procnum - 1) / 4 *
                                                          3) { // Don't worry about the middle one, it is divided into four parts, and the first group and the last group need to assign him fast nodes.
                        cout << "iter " << iter << " change Group !!!" << endl;
                        // Modify Group[iter-1] and modify other nodes in the group where nodetemp belongs to be fast nodes.
                        vector<int> fastnodex = findFastNodes(node_caltime, Group[(iter - 1) % repeatIter], nodetemp,
                                                              nodesOfGroup, procnum - 1);
                        changeGroup(Group, nodetemp, fastnodex, nodesOfGroup, iter);
                        //Group.push_back(newvectemp);
                    } else {
                        Group.push_back(Group[(iter - 1) % repeatIter]);
                    }
                }
            }
            // Update the iteration interval information of all nodes.
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
    delete[] sendNodes;
}