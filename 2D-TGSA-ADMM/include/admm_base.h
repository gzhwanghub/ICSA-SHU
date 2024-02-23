/*******************************************************************************
 *Copyright (c) [2022] [Guozheng Wang <gzh.wang@outlook.com>]  
 *Filename: admm_base.h
 *Description: 
 *Created by Guozheng Wang on 2022/2/11.
*******************************************************************************/
#include "properties.h"
#include "prob.h"
#include <string>

#ifndef SPARSE_ADMM_ADMM_BASE_H
#define SPARSE_ADMM_ADMM_BASE_H
class AdmmBase{
public:
    AdmmBase(args_t *args, problem *prob, string test_file_path);
    ~AdmmBase();

private:

};



#endif //SPARSE_ADMM_ADMM_BASE_H
