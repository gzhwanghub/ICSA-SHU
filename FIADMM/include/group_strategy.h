/*************************************************************************
    > File Name: group_strategy.h
    > Description: Group exchange
    > Author: Xin Huang
    > Created Time: 2020-08-21
 ************************************************************************/

#ifndef FIADMM_GROUP_STRATEGY_H
#define FIADMM_GROUP_STRATEGY_H

#include <vector>
using namespace std;

class GroupStrategy {
public:
    int repeatIter;
    GroupStrategy(int iternums);
    vector<int> exchangeElement(vector<int> data,int GroupNum,int Group1,int Group2,int part);
    vector<vector<int>> divideGroup(vector<int> nodes,int groupNums);
    int position(double *vec,int size,int index);
    vector<int> findFastNodes(double * time,vector<int> group,int node,int numsofGrup,int size);
    void changeGroup(vector<vector<int>> &data,int node,vector<int> fastVec,int numsOfgroup,int iter);
    void MasterNodes(int procnum,int nodesOfGroup,int DynamicGroup,int maxIteration);
};

//namespace comlkit{
//    inline double min(double a, double b)
//    {
//        if (a < b) {
//            return a;
//        }
//        else{
//            return b;
//        }
//    }
//
//    inline double max(double a, double b)
//    {
//        if (a > b) {
//            return a;
//        }
//        else{
//            return b;
//        }
//    }
//
//    inline int sign(double x){
//        return (0 < x) - (x < 0);
//    }
//}

#endif //FIADMM_GROUP_STRATEGY_H
