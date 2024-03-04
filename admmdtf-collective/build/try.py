import train_framework as ho
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('My rank is ',rank)
# x = ho.sayhello(comm)
ho.test_main(comm)
# ho.test_main2(comm)
# try:
#     ho.test_main(list())
# except:
#     pass
# else:
#     assert 0, "exception not raised"
# MPI.Finalize()
# print("rank is %d, x= %d"%(rank,x))
#
## ----- Object creation -----
#

