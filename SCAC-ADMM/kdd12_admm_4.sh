#!/bin/bash


yhrun -N 4 -n 64 ./bin/admm ./conf/$1.conf train_data=./data/kdd12/64/shitu_train_ pred_data=./data/kdd12/shitu_test_0 >> ./result_$2/kdd12/$1_4_64.txt
