#!/bin/bash


yhrun -N 4 -n 64 ./bin/admm ./conf/$1.conf train_data=./data/news/64/shitu_train_ pred_data=./data/news/shitu_test_0 >> ./result_$2/news/$1_4_64.txt
