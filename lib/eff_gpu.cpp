#include "eff_gpu.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace std;

struct EFF_GPU cuda_AllocateStructure() {
  struct EFF_GPU eff_gpu;

  eff_gpu.natoms_gpu = 0;        // atom data 1
  eff_gpu.x_gpu = nullptr;       // atom data 2
  eff_gpu.f_gpu = nullptr;       // atom data 3
  eff_gpu.q_gpu = nullptr;       // atom data 4
  eff_gpu.erforce_gpu = nullptr; // atom data 5
  eff_gpu.eradius_gpu = nullptr; // atom data 6
  eff_gpu.spin_gpu = nullptr;    // atom data 7
  eff_gpu.type_gpu = nullptr;    // atom data 8
  eff_gpu.nlocal_gpu = 0;        // atom data 9
  eff_gpu.ntypes_gpu = 0;        // atom data 10

  eff_gpu.newton_pair_gpu = 0; // force data 1
  eff_gpu.qqrd2e_gpu = 0;      // force data 2

  eff_gpu.hhmss2e_gpu = 0;                     // eff data 1
  eff_gpu.h2e_gpu = 0;                         // eff data 2
  eff_gpu.pressure_with_evirials_flag_gpu = 0; // eff data 3
  eff_gpu.cutsq_gpu = nullptr;                 // eff data 4
  eff_gpu.PAULI_CORE_A_gpu = nullptr;          // eff data 5
  eff_gpu.PAULI_CORE_B_gpu = nullptr;          // eff data 6
  eff_gpu.PAULI_CORE_C_gpu = nullptr;          // eff data 7
  eff_gpu.PAULI_CORE_D_gpu = nullptr;          // eff data 8
  eff_gpu.PAULI_CORE_E_gpu = nullptr;          // eff data 9
  eff_gpu.ecp_type_gpu = nullptr;              // eff data 10
  eff_gpu.limit_eradius_flag_gpu = 0;          // eff data 11

  eff_gpu.inum_gpu = 0;                  // list data 1
  eff_gpu.ilist_gpu = nullptr;           // list data 2
  eff_gpu.numneigh_gpu = nullptr;        // list data 3
  eff_gpu.numneigh_offset_gpu = nullptr; // list data 4
  eff_gpu.firstneigh_gpu = nullptr;      // list data 5

  eff_gpu.evflag_gpu = 0;        // pair data 1
  eff_gpu.eflag_either_gpu = 0;  // pair data 2
  eff_gpu.eflag_global_gpu = 0;  // pair data 3
  eff_gpu.eflag_atom_gpu = 0;    // pair data 4
  eff_gpu.vflag_either_gpu = 0;  // pair data 5
  eff_gpu.vflag_global_gpu = 0;  // pair data 6
  eff_gpu.vflag_atom_gpu = 0;    // pair data 7
  eff_gpu.pvector_gpu = nullptr; // pair data 8

  eff_gpu.eng_coul_gpu = 0;     // pair statistic data 1
  eff_gpu.eng_vdwl_gpu = 0;     // pair statistic data 2
  eff_gpu.eatom_gpu = nullptr;  // pair statistic data 3
  eff_gpu.vatom_gpu = nullptr;  // pair statistic data 4
  eff_gpu.virial_gpu = nullptr; // pair statistic data 5

  eff_gpu.domain_xperiodic_gpu = 0; // domain_info data 1
  eff_gpu.domain_yperiodic_gpu = 0; // domain_info data 2
  eff_gpu.domain_zperiodic_gpu = 0; // domain_info data 3
  eff_gpu.domain_delx_gpu = 0;      // domain_info data 4
  eff_gpu.domain_dely_gpu = 0;      // domain_info data 5
  eff_gpu.domain_delz_gpu = 0;      // domain_info data 6

  return eff_gpu;
}

void cuda_DeallocateStructure(EFF_GPU &eff_gpu) {
  // free atom data
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
  // free eff data
  if (eff_gpu.cutsq_gpu != nullptr) {
    cudaFree(eff_gpu.cutsq_gpu);
  }
  if (eff_gpu.PAULI_CORE_A_gpu != nullptr) {
    cudaFree(eff_gpu.PAULI_CORE_A_gpu);
  }
  if (eff_gpu.PAULI_CORE_B_gpu != nullptr) {
    cudaFree(eff_gpu.PAULI_CORE_B_gpu);
  }
  if (eff_gpu.PAULI_CORE_C_gpu != nullptr) {
    cudaFree(eff_gpu.PAULI_CORE_C_gpu);
  }
  if (eff_gpu.PAULI_CORE_D_gpu != nullptr) {
    cudaFree(eff_gpu.PAULI_CORE_D_gpu);
  }
  if (eff_gpu.PAULI_CORE_E_gpu != nullptr) {
    cudaFree(eff_gpu.PAULI_CORE_E_gpu);
  }
  if (eff_gpu.ecp_type_gpu != nullptr) {
    cudaFree(eff_gpu.ecp_type_gpu);
  }
  // free list data
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
  // free pair statistic data
  if (eff_gpu.eatom_gpu != nullptr) {
    cudaFree(eff_gpu.eatom_gpu);
  }
  if (eff_gpu.vatom_gpu != nullptr) {
    cudaFree(eff_gpu.vatom_gpu);
  }
  if (eff_gpu.virial_gpu != nullptr) {
    cudaFree(eff_gpu.virial_gpu);
  }
}

void cuda_HostToDeviceCopy(void *dst, void *src, int size) {
  cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cuda_DeviceToHostCopy(void *dst, void *src, int size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void cuda_FetchData(EFF_GPU &eff_gpu,                    // structure
                    int natoms_cpu,                      // atom data 1
                    double **x_cpu,                      // atom data 2
                    double **f_cpu,                      // atom data 3
                    double *q_cpu,                       // atom data 4
                    double *erforce_cpu,                 // atom data 5
                    double *eradius_cpu,                 // atom data 6
                    int *spin_cpu,                       // atom data 7
                    int *type_cpu,                       // atom data 8
                    int nlocal_cpu,                      // atom data 9
                    int ntypes_cpu,                      // atom data 10
                    int newton_pair_cpu,                 // force data 1
                    double qqrd2e_cpu,                   // force data 2
                    double hhmss2e_cpu,                  // eff data 1
                    double h2e_cpu,                      // eff data 2
                    int pressure_with_evirials_flag_cpu, // eff data 3
                    double **cutsq_cpu,                  // eff data 4
                    double *PAULI_CORE_A_cpu,            // eff data 5
                    double *PAULI_CORE_B_cpu,            // eff data 6
                    double *PAULI_CORE_C_cpu,            // eff data 7
                    double *PAULI_CORE_D_cpu,            // eff data 8
                    double *PAULI_CORE_E_cpu,            // eff data 9
                    int *ecp_type_cpu,                   // eff data 10
                    int limit_eradius_flag_cpu,          // eff data 11
                    int inum_cpu,                        // list data 1
                    int *ilist_cpu,                      // list data 2
                    int *numneigh_cpu,                   // list data 3 (gen 4)
                    int **firstneigh_cpu,                // list data 5
                    int evflag_cpu,                      // pair data 1
                    int eflag_either_cpu,                // pair data 2
                    int eflag_global_cpu,                // pair data 3
                    int eflag_atom_cpu,                  // pair data 4
                    int vflag_either_cpu,                // pair data 5
                    int vflag_global_cpu,                // pair data 6
                    int vflag_atom_cpu,                  // pair data 7
                    double *pvector_cpu,                 // pair data 8
                    double eng_coul_cpu,      // pair statistic data 1
                    double eng_vdwl_cpu,      // pair statistic data 2
                    double *eatom_cpu,        // pair statistic data 3
                    double **vatom_cpu,       // pair statistic data 4
                    double *virial_cpu,       // pair statistic data 5
                    int domain_xperiodic_cpu, // domain_info data 1
                    int domain_yperiodic_cpu, // domain_info data 2
                    int domain_zperiodic_cpu, // domain_info data 3
                    double domain_delx_cpu,   // domain_info data 4
                    double domain_dely_cpu,   // domain_info data 5
                    double domain_delz_cpu    // domain_info data 6
) {
  int index, subindex;
  eff_gpu.natoms_gpu = natoms_cpu;
  eff_gpu.nlocal_gpu = nlocal_cpu;
  printf("n = %d\n", natoms_cpu);
  printf("nlocal = %d\n", nlocal_cpu);
  printf("ntypes = %d\n", ntypes_cpu);
  int n = natoms_cpu;

  // * Mallocs

  cudaMallocManaged((void **)&(eff_gpu.x_gpu), n * sizeof(double3d));
  cudaMallocManaged((void **)&(eff_gpu.f_gpu), n * sizeof(double3d));
  cudaMallocManaged((void **)&(eff_gpu.q_gpu), n * sizeof(int));
  cudaMallocManaged((void **)&(eff_gpu.erforce_gpu), n * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.eradius_gpu), n * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.spin_gpu), n * sizeof(int));
  cudaMallocManaged((void **)&(eff_gpu.type_gpu), n * sizeof(int));
  cudaMallocManaged((void **)&(eff_gpu.ilist_gpu), n * sizeof(int));
  cudaMallocManaged((void **)&(eff_gpu.numneigh_offset_gpu),
                    (n + 1) * sizeof(int));

  cudaMallocManaged((void **)&(eff_gpu.numneigh_gpu), n * sizeof(int));

  cudaMallocManaged((void **)&(eff_gpu.PAULI_CORE_A_gpu), 100 * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.PAULI_CORE_B_gpu), 100 * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.PAULI_CORE_C_gpu), 100 * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.PAULI_CORE_D_gpu), 100 * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.PAULI_CORE_E_gpu), 100 * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.ecp_type_gpu), 100 * sizeof(int));
  cudaMallocManaged((void **)&(eff_gpu.pvector_gpu), 4 * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.eatom_gpu), nlocal_cpu * sizeof(double));
  cudaMallocManaged((void **)&(eff_gpu.vatom_gpu),
                    nlocal_cpu * sizeof(double6d));
  cudaMallocManaged((void **)&(eff_gpu.virial_gpu), 6 * sizeof(double));

  // * atom data
  /*
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
  */

  cuda_HostToDeviceCopy((void *)(eff_gpu.x_gpu), (void *)(x_cpu),
                        n * sizeof(double3d));
  cuda_HostToDeviceCopy((void *)(eff_gpu.f_gpu), (void *)(f_cpu),
                        n * sizeof(double3d));
  cuda_HostToDeviceCopy((void *)(eff_gpu.erforce_gpu), (void *)(erforce_cpu),
                        n * sizeof(double));
  cuda_HostToDeviceCopy((void *)(eff_gpu.eradius_gpu), (void *)(eradius_cpu),
                        n * sizeof(double));
  cuda_HostToDeviceCopy((void *)(eff_gpu.q_gpu), (void *)(q_cpu),
                        n * sizeof(int));
  cuda_HostToDeviceCopy((void *)(eff_gpu.spin_gpu), (void *)(spin_cpu),
                        n * sizeof(int));
  cuda_HostToDeviceCopy((void *)(eff_gpu.type_gpu), (void *)(type_cpu),
                        n * sizeof(int));
  eff_gpu.ntypes_gpu = ntypes_cpu;

  // * force data
  eff_gpu.newton_pair_gpu = newton_pair_cpu;
  eff_gpu.qqrd2e_gpu = qqrd2e_cpu;

  // * list data
  eff_gpu.inum_gpu = inum_cpu;

  cudaMemcpy((void *)(eff_gpu.ilist_gpu), (void *)ilist_cpu, n * sizeof(int),
             cudaMemcpyHostToDevice);

  cuda_HostToDeviceCopy((void *)(eff_gpu.numneigh_gpu), (void *)numneigh_cpu,
                        n * sizeof(int));

  // numneigh_offset : calculate
  int total_length = 0;
  int *numneigh_offset_cpu;
  int *firstneigh_temparary;
  numneigh_offset_cpu = (int *)malloc((n + 1) * sizeof(int));
  for (index = 0; index < n; ++index) {
    numneigh_offset_cpu[index] = total_length;
    total_length += numneigh_cpu[index];
  }
  numneigh_offset_cpu[n] = total_length;
  cuda_HostToDeviceCopy((void *)(eff_gpu.numneigh_offset_gpu),
                        (void *)numneigh_offset_cpu, (n + 1) * sizeof(int));
  free(numneigh_offset_cpu);

  // firstneigh : flatten mode
  int firstneigh_pos = 0;
  // TODO: Optimize this copy with memcpy
  firstneigh_temparary = (int *)malloc(total_length * sizeof(int));
  for (index = 0; index < n; ++index) {
    for (subindex = 0; subindex < numneigh_cpu[index]; ++subindex) {
      firstneigh_temparary[firstneigh_pos++] = firstneigh_cpu[index][subindex];
    }
  }
  cudaMallocManaged((void **)&(eff_gpu.firstneigh_gpu),
                    total_length * sizeof(int));
  cuda_HostToDeviceCopy((void *)(eff_gpu.firstneigh_gpu),
                        (void *)firstneigh_temparary, total_length);
  free(firstneigh_temparary);

  // * eff data
  eff_gpu.hhmss2e_gpu = hhmss2e_cpu;
  eff_gpu.h2e_gpu = h2e_cpu;
  eff_gpu.pressure_with_evirials_flag_gpu = pressure_with_evirials_flag_cpu;
  cudaMallocManaged((void **)&(eff_gpu.cutsq_gpu),
                    (ntypes_cpu + 1) * (ntypes_cpu + 1) * sizeof(double));
  double *cutsq_gpu_temp;
  cutsq_gpu_temp =
      (double *)malloc(sizeof(double) * (ntypes_cpu + 1) * (ntypes_cpu + 1));
  for (index = 0; index <= ntypes_cpu; ++index) {
    for (subindex = 0; subindex <= ntypes_cpu; ++subindex) {
      cutsq_gpu_temp[index * (ntypes_cpu + 1) + subindex] =
          cutsq_cpu[index][subindex];
    }
  }
  cuda_HostToDeviceCopy((void *)(eff_gpu.cutsq_gpu), (void *)(cutsq_gpu_temp),
                        sizeof(double) * (ntypes_cpu + 1) * (ntypes_cpu + 1));
  free(cutsq_gpu_temp);

  cuda_HostToDeviceCopy((void *)(eff_gpu.PAULI_CORE_A_gpu),
                        (void *)PAULI_CORE_A_cpu, 100 * sizeof(double));

  cuda_HostToDeviceCopy((void *)(eff_gpu.PAULI_CORE_B_gpu),
                        (void *)PAULI_CORE_B_cpu, 100 * sizeof(double));

  cuda_HostToDeviceCopy((void *)(eff_gpu.PAULI_CORE_C_gpu),
                        (void *)PAULI_CORE_C_cpu, 100 * sizeof(double));

  cuda_HostToDeviceCopy((void *)(eff_gpu.PAULI_CORE_D_gpu),
                        (void *)PAULI_CORE_D_cpu, 100 * sizeof(double));

  cuda_HostToDeviceCopy((void *)(eff_gpu.PAULI_CORE_E_gpu),
                        (void *)PAULI_CORE_E_cpu, 100 * sizeof(double));

  cuda_HostToDeviceCopy((void *)(eff_gpu.ecp_type_gpu), (void *)(ecp_type_cpu),
                        100 * sizeof(int));
  eff_gpu.limit_eradius_flag_gpu = limit_eradius_flag_cpu;

  // * pair flag data
  eff_gpu.evflag_gpu = evflag_cpu;
  eff_gpu.eflag_either_gpu = eflag_either_cpu;
  eff_gpu.eflag_global_gpu = eflag_global_cpu;
  eff_gpu.eflag_atom_gpu = eflag_atom_cpu;
  eff_gpu.vflag_either_gpu = vflag_either_cpu;
  eff_gpu.vflag_global_gpu = vflag_global_cpu;
  eff_gpu.vflag_atom_gpu = vflag_atom_cpu;

  eff_gpu.pvector_gpu[0] = 0;
  eff_gpu.pvector_gpu[1] = 0;
  eff_gpu.pvector_gpu[2] = 0;
  eff_gpu.pvector_gpu[3] = 0;

  // * pair statistic data
  eff_gpu.eng_coul_gpu = eng_coul_cpu;
  eff_gpu.eng_vdwl_gpu = eng_vdwl_cpu;

  cuda_HostToDeviceCopy((void *)(eff_gpu.eatom_gpu), (void *)(eatom_cpu),
                        nlocal_cpu * sizeof(double));

  cuda_HostToDeviceCopy((void *)(eff_gpu.vatom_gpu), (void *)(vatom_cpu),
                        nlocal_cpu * sizeof(double6d));

  // * domain_info data
  eff_gpu.domain_xperiodic_gpu = domain_xperiodic_cpu;
  eff_gpu.domain_yperiodic_gpu = domain_yperiodic_cpu;
  eff_gpu.domain_zperiodic_gpu = domain_zperiodic_cpu;
  eff_gpu.domain_delx_gpu = domain_delx_cpu;
  eff_gpu.domain_dely_gpu = domain_dely_cpu;
  eff_gpu.domain_delz_gpu = domain_delz_cpu;

  // Unified Memory Check
  // ofstream output_file;
  // output_file.open("gpu_data_check.txt");
  // output_file << " x = \n";
  // for (index = 0; index < 10; ++index)
  //   output_file << eff_gpu.x[index][0] << " ";
  // output_file.close();
}

void cuda_FetchBackData(EFF_GPU &eff_gpu,         // structure
                        double **x_cpu,           // atom data 2
                        double **f_cpu,           // atom data 3
                        double *q_cpu,            // atom data 4
                        double *erforce_cpu,      // atom data 5
                        double *eradius_cpu,      // atom data 6
                        int *spin_cpu,            // atom data 7
                        int *type_cpu,            // atom data 8
                        int &newton_pair_cpu,     // force data 1
                        double &qqrd2e_cpu,       // force data 2
                        double &hhmss2e_cpu,      // eff data 1
                        double &h2e_cpu,          // eff data 2
                        double *PAULI_CORE_A_cpu, // eff data 5
                        double *PAULI_CORE_B_cpu, // eff data 6
                        double *PAULI_CORE_C_cpu, // eff data 7
                        double *PAULI_CORE_D_cpu, // eff data 8
                        double *PAULI_CORE_E_cpu, // eff data 9
                        int *ecp_type_cpu,        // eff data 10
                        double *pvector_cpu,      // pair data 8
                        double &eng_coul_cpu,     // pair statistic data 1
                        double &eng_vdwl_cpu,     // pair statistic data 2
                        double *eatom_cpu,        // pair statistic data 3
                        double **vatom_cpu,       // pair statistic data 4
                        double *virial_cpu        // pair statistic data 5
) {
  cudaDeviceSynchronize();
  // * atom data
  // natoms : copy
  int n = eff_gpu.natoms_gpu;

  // for (int index = 0; index < n; ++index) {
  //   x_cpu[index][0] = eff_gpu.x_gpu[index][0];
  //   x_cpu[index][1] = eff_gpu.x_gpu[index][1];
  //   x_cpu[index][2] = eff_gpu.x_gpu[index][2];
  //   f_cpu[index][0] = eff_gpu.f_gpu[index][0];
  //   f_cpu[index][1] = eff_gpu.f_gpu[index][1];
  //   f_cpu[index][2] = eff_gpu.f_gpu[index][2];
  //   q_cpu[index] = eff_gpu.q_gpu[index];
  //   erforce_cpu[index] = eff_gpu.erforce_gpu[index];
  //   eradius_cpu[index] = eff_gpu.eradius_gpu[index];
  //   spin_cpu[index] = eff_gpu.spin_gpu[index];
  //   type_cpu[index] = eff_gpu.type_gpu[index];
  // }

  cuda_DeviceToHostCopy((void *)(x_cpu), (void *)(eff_gpu.x_gpu),
                        n * sizeof(double3d));
  cuda_DeviceToHostCopy((void *)(f_cpu), (void *)(eff_gpu.f_gpu),
                        n * sizeof(double3d));
  cuda_DeviceToHostCopy((void *)(erforce_cpu), (void *)(eff_gpu.erforce_gpu),
                        n * sizeof(double));
  cuda_DeviceToHostCopy((void *)(eradius_cpu), (void *)(eff_gpu.eradius_gpu),
                        n * sizeof(double));
  cuda_DeviceToHostCopy((void *)(q_cpu), (void *)(eff_gpu.q_gpu),
                        n * sizeof(int));
  cuda_DeviceToHostCopy((void *)(spin_cpu), (void *)(eff_gpu.spin_gpu),
                        n * sizeof(int));
  cuda_DeviceToHostCopy((void *)(type_cpu), (void *)(eff_gpu.type_gpu),
                        n * sizeof(int));

  // * force data
  newton_pair_cpu = eff_gpu.newton_pair_gpu;
  qqrd2e_cpu = eff_gpu.qqrd2e_gpu;

  // * eff data
  hhmss2e_cpu = eff_gpu.hhmss2e_gpu;
  h2e_cpu = eff_gpu.h2e_gpu;
  cuda_DeviceToHostCopy((void *)PAULI_CORE_A_cpu,
                        (void *)eff_gpu.PAULI_CORE_A_gpu, sizeof(double) * 100);
  cuda_DeviceToHostCopy((void *)PAULI_CORE_B_cpu,
                        (void *)eff_gpu.PAULI_CORE_B_gpu, sizeof(double) * 100);
  cuda_DeviceToHostCopy((void *)PAULI_CORE_C_cpu,
                        (void *)eff_gpu.PAULI_CORE_C_gpu, sizeof(double) * 100);
  cuda_DeviceToHostCopy((void *)PAULI_CORE_D_cpu,
                        (void *)eff_gpu.PAULI_CORE_D_gpu, sizeof(double) * 100);
  cuda_DeviceToHostCopy((void *)PAULI_CORE_E_cpu,
                        (void *)eff_gpu.PAULI_CORE_E_gpu, sizeof(double) * 100);
  cuda_DeviceToHostCopy((void *)ecp_type_cpu, (void *)eff_gpu.ecp_type_gpu,
                        sizeof(int) * 100);

  // * pair data
  pvector_cpu[0] = eff_gpu.pvector_gpu[0];
  pvector_cpu[1] = eff_gpu.pvector_gpu[1];
  pvector_cpu[2] = eff_gpu.pvector_gpu[2];
  pvector_cpu[3] = eff_gpu.pvector_gpu[3];

  eng_coul_cpu = eff_gpu.eng_coul_gpu;
  eng_vdwl_cpu = eff_gpu.eng_vdwl_gpu;

  cuda_DeviceToHostCopy((void *)(vatom_cpu), (void *)(eff_gpu.vatom_gpu),
                        n * sizeof(double6d));

  cuda_DeviceToHostCopy((void *)(eatom_cpu), (void *)(eff_gpu.eatom_gpu),
                        n * sizeof(double));
  virial_cpu[0] = eff_gpu.virial_gpu[0];
  virial_cpu[1] = eff_gpu.virial_gpu[1];
  virial_cpu[2] = eff_gpu.virial_gpu[2];
  virial_cpu[3] = eff_gpu.virial_gpu[3];
  virial_cpu[4] = eff_gpu.virial_gpu[4];
  virial_cpu[5] = eff_gpu.virial_gpu[5];
}
