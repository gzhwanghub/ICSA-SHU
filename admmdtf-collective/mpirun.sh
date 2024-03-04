#!/bin/bash
if [ ! -d "build"]; then
	mkdir build
else
	rm -rf build/*
fi
cd build
cmake ..
make
cp ../try.py .
mpirun -n 16 -hostfile ../hostfile python try.py
#mpirun -n 9 --hosts node0,node1,node3,node4,node5,node6,node7 python try.py
#mpirun -np 16 -hostfile ../hostfile python try.py


