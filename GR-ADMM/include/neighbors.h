//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_NEIGHBORS_H
#define GR_ADMM_NEIGHBORS_H


class neighbors {
public:
    int neighborsNums;
    int *neighs;

    void setNeighbours(int nums, int *set);

    void clearNeighbours();
};


#endif //GR_ADMM_NEIGHBORS_H
