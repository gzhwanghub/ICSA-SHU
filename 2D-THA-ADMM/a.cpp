#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size, color, node_rank, node_size;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    printf("Rank %d is on node %d \n", rank, node_rank);
    printf("Rank %d is on processor %s \n", rank, processor_name);
    
    
	
    MPI_Finalize();
    return 0;
}

