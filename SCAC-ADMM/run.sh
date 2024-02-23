#!/bin/bash

yhrun -N 4 -n 32 ./bin/admm ./conf/$1.conf train_data=./data/$2/32/shitu_train_ >> ./result/$2/$1_4_32.txt



#yhrun -N 2 -n 8 ./bin/admm ./conf/$1.conf train_data=./data/$2/8/shitu_train_ >> ./result/$2/$1_2_8.txt


#yhrun -N 4 -n 16 ./bin/admm ./conf/$1.conf train_data=./data/$2/16/shitu_train_ >> ./result/$2/$1_4_16.txt

#yhrun -N 8 -n 8 ./bin/admm ./conf/$1.conf train_data=./data/$2/8/shitu_train_ >> ./result/$2/$1_8_8.txt
#yhrun -N 8 -n 16 ./bin/admm ./conf/$1.conf train_data=./data/$2/16/shitu_train_ >> ./result/$2/$1_8_16.txt
#yhrun -N 8 -n 32 ./bin/admm ./conf/$1.conf train_data=./data/$2/32/shitu_train_ >> ./result/$2/$1_8_32.txt

#mpirun -f ./bin/hostfile_8_8 -np 8 ./bin/admm ./conf/$1.conf train_data=./data/$2/8/shitu_train_ >> ./result/$2/$1_8_8.txt
#mpirun -np 1 ./bin/admm conf/$1.conf task=pred num_fea=10000007 train_data=./data/$2/shitu_test_ model_in=lr_model.dat >> ./result/$2/$1_8_8.txt
#mpirun -f ./bin/hostfile_8_16 -np 16 ./bin/admm ./conf/$1.conf train_data=./data/$2/16/shitu_train_ >> ./result/$2/$1_8_16.txt
#mpirun -np 1 ./bin/admm conf/$1.conf task=pred num_fea=10000007 train_data=./data/$2/shitu_test_ model_in=lr_model.dat >> ./result/$2/$1_8_16.txt
#mpirun -f ./bin/hostfile_8_16 -np 32 ./bin/admm ./conf/$1.conf train_data=./data/$2/32/shitu_train_ >> ./result/$2/$1_8_32.txt
#mpirun -np 1 ./bin/admm conf/$1.conf task=pred num_fea=10000007 train_data=./data/$2/shitu_test_ model_in=lr_model.dat >> ./result/$2/$1_8_32.txt



