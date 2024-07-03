//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_COMMON_H
#define GR_ADMM_COMMON_H

namespace spar {

enum MessageType {
    kSimpleAllreduce1=0x10000,
    kSimpleAllreduce2,
    kScatterReduce,
    kAllGather,
    kmaxandmin,
    decencomm1,
    decencomm2
};

}


#endif //GR_ADMM_COMMON_H
