#!/bin/bash
mpirun -f ./hostfile -np $1  ./gcadmm   
