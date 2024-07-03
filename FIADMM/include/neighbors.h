/*************************************************************************
    > File Name: neighbors.h
    > Description: Set neighbours
    > Author: Xin Huang
    > Created Time: 2020-10-14
 ************************************************************************/

#ifndef FIADMM_NEIGHBORS_H
#define FIADMM_NEIGHBORS_H


class neighbors {
public:
    int neighborsNums;
    int *neighs;
    void setNeighbours(int nums,int *set);
    void clearNeighbours();
};


#endif //FIADMM_NEIGHBORS_H
