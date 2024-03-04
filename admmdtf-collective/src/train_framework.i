%module train_framework

%{
#include "train_framework.h"
%}
%include mpi4py/mpi4py.i
%mpi4py_typemap(Comm, MPI_Comm);
%include "train_framework.h"
