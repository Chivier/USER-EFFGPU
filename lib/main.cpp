#include "eff_gpu.h"
#include <memory>
#include <vector>
#include <iostream>
#include <cstdio>

#include <mpi.h>

using namespace std;

int main(int args, char **argv) 
{
  MPI_Init(&args, &argv);

  int myID;
  int nrPS;
  
  MPI_Comm_size(MPI_COMM_WORLD, &nrPS);
  MPI_Comm_rank(MPI_COMM_WORLD, &myID);
  
  int types = 1;
  int type_map[] = {0};
  
//   EPH_GPU eph_gpu = allocate_EPH_GPU(beta, types, type_map);
  
//   deallocate_EPH_GPU(eph_gpu);
  
  MPI_Finalize();
  return 0;
}
