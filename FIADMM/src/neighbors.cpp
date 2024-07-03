/*************************************************************************
    > File Name: neighbors.cpp
    > Description: Set neighbours
    > Author: Xin Huang
    > Created Time: 2020-10-14
 ************************************************************************/

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
