//
// Created by cluster on 2022/5/10.
//

#ifndef DMLC_NEIGHBORS_H
#define DMLC_NEIGHBORS_H


class neighbors {
public:
    int neighborsNums;
    int *neighs;
    void setNeighbours(int nums,int *set);
    void clearNeighbours();
};


#endif //DMLC_NEIGHBORS_H
