//
// Created by cluster on 2022/5/10.
//

#include "../include/neighbors.h"
void neighbors::setNeighbours(int nums, int *set) {
    neighborsNums=nums;
    for(int i=0;i<nums;i++)
    {
        neighs[i]=set[i];
    }
}

void neighbors::clearNeighbours()
{
    this->neighborsNums==0;

}