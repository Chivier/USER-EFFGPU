#ifndef LMP_EFF_GPU_H
#define LMP_EFF_GPU_H

#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

#define CUDA_PAULI_RE 0.9
#define CUDA_PAULI_RC 1.125
#define CUDA_PAULI_RHO -0.2

#define CUDA_ERF_TERMS1 12
#define CUDA_ERF_TERMS2 7
#define CUDA_DERF_TERMS 13

typedef double double3d[3];

struct EFF_GPU {
  int natoms_gpu;
  double3d *x_gpu;
  double3d *f_gpu;
  double *q_gpu;
  double *erforce_gpu;
  double *eradius_gpu;
  int *spin_gpu;
  int *type_gpu;
  int nlocal_gpu;

  int newton_pair_gpu;
  double qqrd2e_gpu;

  int inum;
  int *ilist_gpu;
  int *numneigh_gpu;
  int *numneigh_offset_gpu;
  int *firstneigh_gpu;
};

// memory behaviors
struct EFF_GPU cuda_AllocateStructure();
void cuda_DeallocateStructure(EFF_GPU &eff_gpu);
void cuda_HostToDeviceCopy(void *dst, void *src, int size);
void cuda_DeviceToHostCopy(void *dst, void *src, int size);
void cuda_FetchData(EFF_GPU &eff_gpu, int natoms_cpu, double **x_cpu,
                    double **f_cpu, double *q_cpu, double *erforce_cpu,
                    double *eradius_cpu, int *spin_cpu, int *type_cpu,
                    int nlocal_cpu, int newton_pair_cpu, double qqrd2e_cpu,
                    int inum_cpu, int *ilist_cpu, int *numneigh_cpu,
                    int **firstneigh_cpu);
void cuda_FetchBackData(EFF_GPU &eff_gpu, int natoms_cpu, double **x_cpu,
                        double **f_cpu, double *q_cpu, double *erforce_cpu,
                        double *eradius_cpu, int *spin_cpu, int *type_cpu,
                        int nlocal_cpu, int newton_pair_cpu, double qqrd2e_cpu,
                        int inum_cpu, int *ilist_cpu, int *numneigh_cpu,
                        int **firstneigh_cpu);

// inline cuda kernels
double cuda_ipoly02(double x);
double cuda_ipoly1(double x);
double cuda_ipoly01(double x);
double cuda_ierfoverx1(double x, double *df);
void cuda_KinElec(double radius, double *eke, double *frc);
void cuda_ElecNucNuc(double q, double rc, double *ecoul, double *frc);
void cuda_ElecNucElec(double q, double rc, double re1, double *ecoul,
                      double *frc, double *fre1);
void cuda_ElecElecElec(double rc, double re1, double re2, double *ecoul,
                       double *frc, double *fre1, double *fre2);
void cuda_ElecCoreNuc(double q, double rc, double re1, double *ecoul,
                      double *frc);
void cuda_ElecCoreCore(double q, double rc, double re1, double re2,
                       double *ecoul, double *frc);
void cuda_ElecCoreElec(double q, double rc, double re1, double re2,
                       double *ecoul, double *frc, double *fre2);
void cuda_PauliElecElec(int samespin, double rc, double re1, double re2,
                        double *epauli, double *frc, double *fre1,
                        double *fre2);
void cuda_PauliCoreElec(double rc, double re2, double *epauli, double *frc,
                        double *fre2, double PAULI_CORE_A, double PAULI_CORE_B,
                        double PAULI_CORE_C);
void cuda_PauliCorePElec(double rc, double re2, double *epauli, double *frc,
                         double *fre2, double PAULI_CORE_P_A,
                         double PAULI_CORE_P_B, double PAULI_CORE_P_C,
                         double PAULI_CORE_P_D, double PAULI_CORE_P_E);
void cuda_RForce(double dx, double dy, double dz, double rc, double force,
                 double *fx, double *fy, double *fz);
void cuda_SmallRForce(double dx, double dy, double dz, double rc, double force,
                      double *fx, double *fy, double *fz);
double cuda_cutoff(double x);
double cuda_dcutoff(double x);
void cuda_test_add(int *arr, int n);

void cuda_eff_test(struct EFF_GPU &eff_gpu);
void cuda_eff_compute_kernel();

#endif