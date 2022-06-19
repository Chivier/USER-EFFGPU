// clang-format off
#ifdef PAIR_CLASS

PairStyle(eff/cut/gpu, PairEffCutGPU)

#else
// clang-format on

#ifndef LMP_PAIR_EFF_CUT_GPU_H
#define LMP_PAIR_EFF_CUT_GPU_H

#include "pair.h"

namespace LAMMPS_NS {

class PairEffCutGPU : public Pair {
public:
  PairEffCutGPU(class LAMMPS *);
  virtual ~PairEffCutGPU();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void min_pointers(double **, double **);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);

  void min_xf_pointers(int, double **, double **);
  void min_xf_get(int);
  void min_x_set(int);
  double memory_usage();

private:
  int limit_eradius_flag, pressure_with_evirials_flag;
  double cut_global;
  double **cut;
  int ecp_type[100];
  double PAULI_CORE_A[100], PAULI_CORE_B[100], PAULI_CORE_C[100],
      PAULI_CORE_D[100], PAULI_CORE_E[100];
  double hhmss2e, h2e;

  int nmax;
  double *min_eradius, *min_erforce;

  void allocate();
  void virial_eff_compute();
  void ev_tally_eff(int, int, int, int, double, double);
};

} // namespace LAMMPS_NS

#endif
#endif
