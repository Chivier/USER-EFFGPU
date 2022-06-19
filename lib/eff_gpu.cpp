#include "eff_gpu.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

struct EFF_GPU cuda_AllocateStructure() {
  struct EFF_GPU eff_gpu;

  eff_gpu.natoms_gpu = 0;
  eff_gpu.x_gpu = nullptr;
  eff_gpu.f_gpu = nullptr;
  eff_gpu.q_gpu = nullptr;
  eff_gpu.erforce_gpu = nullptr;
  eff_gpu.eradius_gpu = nullptr;
  eff_gpu.spin_gpu = nullptr;
  eff_gpu.type_gpu = nullptr;
  eff_gpu.nlocal_gpu = 0;

  eff_gpu.newton_pair_gpu = 0;
  eff_gpu.qqrd2e_gpu = 0;

  eff_gpu.inum = 0;
  eff_gpu.ilist_gpu = nullptr;
  eff_gpu.numneigh_gpu = nullptr;
  eff_gpu.numneigh_offset_gpu = nullptr;
  eff_gpu.firstneigh_gpu = nullptr;
  return eff_gpu;
}

void cuda_DeallocateStructure(EFF_GPU &eff_gpu) {
  if (eff_gpu.x_gpu != nullptr) {
    cudaFree(eff_gpu.x_gpu);
  }
  if (eff_gpu.f_gpu != nullptr) {
    cudaFree(eff_gpu.f_gpu);
  }
  if (eff_gpu.q_gpu != nullptr) {
    cudaFree(eff_gpu.q_gpu);
  }
  if (eff_gpu.erforce_gpu != nullptr) {
    cudaFree(eff_gpu.erforce_gpu);
  }
  if (eff_gpu.eradius_gpu != nullptr) {
    cudaFree(eff_gpu.eradius_gpu);
  }
  if (eff_gpu.spin_gpu != nullptr) {
    cudaFree(eff_gpu.spin_gpu);
  }
  if (eff_gpu.type_gpu != nullptr) {
    cudaFree(eff_gpu.type_gpu);
  }
  if (eff_gpu.ilist_gpu != nullptr) {
    cudaFree(eff_gpu.ilist_gpu);
  }
  if (eff_gpu.numneigh_gpu != nullptr) {
    cudaFree(eff_gpu.numneigh_gpu);
  }
  if (eff_gpu.numneigh_offset_gpu != nullptr) {
    cudaFree(eff_gpu.numneigh_offset_gpu);
  }
  if (eff_gpu.firstneigh_gpu != nullptr) {
    cudaFree(eff_gpu.firstneigh_gpu);
  }
}

void cuda_HostToDeviceCopy(void *dst, void *src, int size) {
  cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cuda_DeviceToHostCopy(void *dst, void *src, int size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void cuda_FetchData(EFF_GPU &eff_gpu, int natoms_cpu, double **x_cpu,
                    double **f_cpu, double *q_cpu, double *erforce_cpu,
                    double *eradius_cpu, int *spin_cpu, int *type_cpu,
                    int nlocal_cpu, int newton_pair_cpu, double qqrd2e_cpu,
                    int inum_cpu, int *ilist_cpu, int *numneigh_cpu,
                    int **firstneigh_cpu) {
  // natoms : copy
  eff_gpu.natoms_gpu = natoms_cpu;
  int n = natoms_cpu;

  // x : allocate and copy
  cudaMallocManaged((void **)&(eff_gpu.x_gpu), n * sizeof(double3d));
  cudaMallocManaged((void **)&(eff_gpu.f_gpu), n * sizeof(double3d));
  cudaMallocManaged((void **)&(eff_gpu.q_gpu), n * sizeof(int));
  cudaMallocManaged((void **)&(eff_gpu.erforce_gpu), n * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.eradius_gpu), n * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.spin_gpu), n * sizeof(int));
  cudaMallocManaged((void **)&(eff_gpu.type_gpu), n * sizeof(int));

  for (int index = 0; index < n; ++index) {
    eff_gpu.x_gpu[index][0] = x_cpu[index][0];
    eff_gpu.x_gpu[index][1] = x_cpu[index][1];
    eff_gpu.x_gpu[index][2] = x_cpu[index][2];
    eff_gpu.f_gpu[index][0] = f_cpu[index][0];
    eff_gpu.f_gpu[index][1] = f_cpu[index][1];
    eff_gpu.f_gpu[index][2] = f_cpu[index][2];
    eff_gpu.q_gpu[index] = q_cpu[index];
    eff_gpu.erforce_gpu[index] = erforce_cpu[index];
    eff_gpu.eradius_gpu[index] = eradius_cpu[index];
    eff_gpu.spin_gpu[index] = spin_cpu[index];
    eff_gpu.type_gpu[index] = type_cpu[index];
  }
  // nlocal : copy
  eff_gpu.nlocal_gpu = nlocal_cpu;

  // newton_pair : copy
  eff_gpu.newton_pair_gpu = newton_pair_cpu;

  // qqrd2e : copy
  eff_gpu.qqrd2e_gpu = qqrd2e_cpu;

  // inum : copy
  eff_gpu.inum = inum_cpu;

  // ilist : allocate and copy
  cudaMallocManaged((void **)&(eff_gpu.ilist_gpu), n * sizeof(int));
  cudaMemcpy((void *)(eff_gpu.ilist_gpu), (void *)ilist_cpu, n * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMallocManaged((void **)&(eff_gpu.numneigh_gpu), n * sizeof(int));
  cuda_HostToDeviceCopy((void *)(eff_gpu.numneigh_gpu), (void *)numneigh_cpu,
                        n * sizeof(int));

  // numneigh_offset : calculate
  int index;
  int total_length = 0;
  int *numneigh_offset_cpu;
  int *firstneigh_temparary;
  numneigh_offset_cpu = (int *)malloc(n * sizeof(int));
  cudaMallocManaged((void **)&(eff_gpu.numneigh_offset_gpu), n * sizeof(int));
  for (index = 0; index < n; ++index) {
    numneigh_offset_cpu[index] = total_length;
    total_length += numneigh_cpu[index];
  }
  cuda_HostToDeviceCopy((void *)(eff_gpu.numneigh_offset_gpu),
                        (void *)numneigh_offset_cpu, n * sizeof(int));
  free(numneigh_offset_cpu);

  // firstneigh : allocate and copy
  int firstneigh_pos = 0;
  // TODO: Optimize this copy with memcpy
  firstneigh_temparary = (int *)malloc(total_length * sizeof(int));
  for (int index = 0; index < n; ++index) {
    for (int subindex = 0; subindex < numneigh_cpu[index]; ++subindex) {
      firstneigh_temparary[firstneigh_pos++] = firstneigh_cpu[index][subindex];
    }
  }
  cudaMallocManaged((void **)&(eff_gpu.firstneigh_gpu),
                    total_length * sizeof(int));
  cuda_HostToDeviceCopy((void *)(eff_gpu.firstneigh_gpu),
                        (void *)firstneigh_temparary, total_length);
  free(firstneigh_temparary);
}

void cuda_FetchBackData(EFF_GPU &eff_gpu, int natoms_cpu, double **x_cpu,
                        double **f_cpu, double *q_cpu, double *erforce_cpu,
                        double *eradius_cpu, int *spin_cpu, int *type_cpu,
                        int nlocal_cpu, int newton_pair_cpu, double qqrd2e_cpu,
                        int inum_cpu, int *ilist_cpu, int *numneigh_cpu,
                        int **firstneigh_cpu) {
  // natoms : copy
  int n = natoms_cpu;

  cuda_test_add(eff_gpu.ilist_gpu, n);
  cudaDeviceSynchronize();

  // x : copy
  for (int index = 0; index < n; ++index) {
    x_cpu[index][0] = eff_gpu.x_gpu[index][0];
    x_cpu[index][1] = eff_gpu.x_gpu[index][1];
    x_cpu[index][2] = eff_gpu.x_gpu[index][2];
    f_cpu[index][0] = eff_gpu.f_gpu[index][0];
    f_cpu[index][1] = eff_gpu.f_gpu[index][1];
    f_cpu[index][2] = eff_gpu.f_gpu[index][2];
    q_cpu[index] = eff_gpu.q_gpu[index];
    erforce_cpu[index] = eff_gpu.erforce_gpu[index];
    eradius_cpu[index] = eff_gpu.eradius_gpu[index];
    spin_cpu[index] = eff_gpu.spin_gpu[index];
    type_cpu[index] = eff_gpu.type_gpu[index];
  }
  cuda_DeviceToHostCopy((void *)ilist_cpu, (void *)(eff_gpu.ilist_gpu),
                        n * sizeof(int));
  cudaDeviceSynchronize();
}

void cuda_eff_test(struct EFF_GPU &eff_gpu) {}