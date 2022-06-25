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
typedef double double6d[6];

struct EFF_GPU {
  // atom data
  int natoms_gpu;      // atom data 1
  double3d *x_gpu;     // atom data 2
  double3d *f_gpu;     // atom data 3
  double *q_gpu;       // atom data 4
  double *erforce_gpu; // atom data 5
  double *eradius_gpu; // atom data 6
  int *spin_gpu;       // atom data 7
  int *type_gpu;       // atom data 8
  int nlocal_gpu;      // atom data 9
  int ntypes_gpu;      // atom data 10

  // force data
  int newton_pair_gpu; // force data 1
  double qqrd2e_gpu;   // force data 2

  // eff data
  double hhmss2e_gpu;                  // eff data 1
  double h2e_gpu;                      // eff data 2
  int pressure_with_evirials_flag_gpu; // eff data 3
  double *cutsq_gpu;                   // eff data 4
  double *PAULI_CORE_A_gpu;            // eff data 5
  double *PAULI_CORE_B_gpu;            // eff data 6
  double *PAULI_CORE_C_gpu;            // eff data 7
  double *PAULI_CORE_D_gpu;            // eff data 8
  double *PAULI_CORE_E_gpu;            // eff data 9
  int *ecp_type_gpu;                   // eff data 10
  int limit_eradius_flag_gpu;          // eff data 11

  // list data
  int inum_gpu;             // list data 1
  int *ilist_gpu;           // list data 2
  int *numneigh_gpu;        // list data 3
  int *numneigh_offset_gpu; // list data 4
  int *firstneigh_gpu;      // list data 5

  // pair flag data
  int evflag_gpu;       // pair data 1
  int eflag_either_gpu; // pair data 2
  int eflag_global_gpu; // pair data 3
  int eflag_atom_gpu;   // pair data 4
  int vflag_either_gpu; // pair data 5
  int vflag_global_gpu; // pair data 6
  int vflag_atom_gpu;   // pair data 7
  double *pvector_gpu;  // pair data 8

  // pair statistic data
  double eng_coul_gpu; // pair statistic data 1
  double eng_vdwl_gpu; // pair statistic data 2
  double *eatom_gpu;   // pair statistic data 3
  double6d *vatom_gpu; // pair statistic data 4
  double *virial_gpu;  // pair statistic data 5

  // domain_info data
  int domain_xperiodic_gpu; // domain_info data 1
  int domain_yperiodic_gpu; // domain_info data 2
  int domain_zperiodic_gpu; // domain_info data 3
  double domain_delx_gpu;   // domain_info data 4
  double domain_dely_gpu;   // domain_info data 5
  double domain_delz_gpu;   // domain_info data 6
};

// memory behaviors
struct EFF_GPU cuda_AllocateStructure();
void cuda_DeallocateStructure(EFF_GPU &eff_gpu);
void cuda_HostToDeviceCopy(void *dst, void *src, int size);
void cuda_DeviceToHostCopy(void *dst, void *src, int size);
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
);

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
);

// inline cuda kernels
// double cuda_ipoly02(double x);
// double cuda_ipoly1(double x);
// double cuda_ipoly01(double x);
// double cuda_ierfoverx1(double x, double *df);
// void cuda_KinElec(double radius, double *eke, double *frc);
// void cuda_ElecNucNuc(double q, double rc, double *ecoul, double *frc);
// void cuda_ElecNucElec(double q, double rc, double re1, double *ecoul,
//                       double *frc, double *fre1);
// void cuda_ElecElecElec(double rc, double re1, double re2, double *ecoul,
//                        double *frc, double *fre1, double *fre2);
// void cuda_ElecCoreNuc(double q, double rc, double re1, double *ecoul,
//                       double *frc);
// void cuda_ElecCoreCore(double q, double rc, double re1, double re2,
//                        double *ecoul, double *frc);
// void cuda_ElecCoreElec(double q, double rc, double re1, double re2,
//                        double *ecoul, double *frc, double *fre2);
// void cuda_PauliElecElec(int samespin, double rc, double re1, double re2,
//                         double *epauli, double *frc, double *fre1,
//                         double *fre2);
// void cuda_PauliCoreElec(double rc, double re2, double *epauli, double *frc,
//                         double *fre2, double PAULI_CORE_A, double
//                         PAULI_CORE_B, double PAULI_CORE_C);
// void cuda_PauliCorePElec(double rc, double re2, double *epauli, double *frc,
//                          double *fre2, double PAULI_CORE_P_A,
//                          double PAULI_CORE_P_B, double PAULI_CORE_P_C,
//                          double PAULI_CORE_P_D, double PAULI_CORE_P_E);
// void cuda_RForce(double dx, double dy, double dz, double rc, double force,
//                  double *fx, double *fy, double *fz);
// void cuda_SmallRForce(double dx, double dy, double dz, double rc, double
// force,
//                       double *fx, double *fy, double *fz);
// double cuda_cutoff(double x);
// double cuda_dcutoff(double x);
// void cuda_test_add(int *arr, int n);

void cuda_eff_test(struct EFF_GPU &eff_gpu, int eflag, int vflag);

#endif