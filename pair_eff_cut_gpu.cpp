/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Andres Jaramillo-Botero
------------------------------------------------------------------------- */

#include "eff_gpu.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "min.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair_eff_cut_gpu.h"
#include "update.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairEffCutGPU::PairEffCutGPU(LAMMPS *lmp) : Pair(lmp) {
  single_enable = 0;

  nmax = 0;
  min_eradius = NULL;
  min_erforce = NULL;
  nextra = 4;
  pvector = new double[nextra];
}

/* ---------------------------------------------------------------------- */

PairEffCutGPU::~PairEffCutGPU() {
  delete[] pvector;
  memory->destroy(min_eradius);
  memory->destroy(min_erforce);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}

/* ---------------------------------------------------------------------- */

void PairEffCutGPU::compute(int eflag, int vflag) {
  struct EFF_GPU eff_gpu;
  eff_gpu = cuda_AllocateStructure();
  cuda_FetchData(eff_gpu,                             // structure
                 atom->natoms,                        // atom data 1
                 atom->x,                             // atom data 2
                 atom->f,                             // atom data 3
                 atom->q,                             // atom data 4
                 atom->erforce,                       // atom data 5
                 atom->eradius,                       // atom data 6
                 atom->spin,                          // atom data 7
                 atom->type,                          // atom data 8
                 atom->nlocal,                        // atom data 9
                 atom->ntypes,                        // atom data 10
                 force->newton_pair,                  // force data 1
                 force->qqrd2e,                       // force data 2
                 hhmss2e,                             // eff data 1
                 h2e,                                 // eff data 2
                 pressure_with_evirials_flag,         // eff data 3
                 cutsq,                               // eff data 4
                 PAULI_CORE_A,                        // eff data 5
                 PAULI_CORE_B,                        // eff data 6
                 PAULI_CORE_C,                        // eff data 7
                 PAULI_CORE_D,                        // eff data 8
                 PAULI_CORE_E,                        // eff data 9
                 ecp_type,                            // eff data 10
                 limit_eradius_flag,                  // eff data 11
                 list->inum,                          // list data 1
                 list->ilist,                         // list data 2
                 list->numneigh,                      // list data 3 (gen 4)
                 list->firstneigh,                    // list data 5
                 evflag,                              // pair data 1
                 eflag_either,                        // pair data 2
                 eflag_global,                        // pair data 3
                 eflag_atom,                          // pair data 4
                 vflag_either,                        // pair data 5
                 vflag_global,                        // pair data 6
                 vflag_atom,                          // pair data 7
                 pvector,                             // pair data 8
                 eng_coul,                            // pair statistic data 1
                 eng_vdwl,                            // pair statistic data 2
                 eatom,                               // pair statistic data 3
                 vatom,                               // pair statistic data 4
                 virial,                              // pair statistic data 5
                 domain->xperiodic,                   // domain_info data 1
                 domain->yperiodic,                   // domain_info data 2
                 domain->zperiodic,                   // domain_info data 3
                 domain->boxhi[0] - domain->boxlo[0], // domain_info data 4
                 domain->boxhi[1] - domain->boxlo[1], // domain_info data 5
                 domain->boxhi[2] - domain->boxlo[2]  // domain_info data 6
  );
  cuda_eff_test(eff_gpu, eflag, vflag);
  cuda_FetchBackData(eff_gpu,            // structure
                     atom->x,            // atom data 2
                     atom->f,            // atom data 3
                     atom->q,            // atom data 4
                     atom->erforce,      // atom data 5
                     atom->eradius,      // atom data 6
                     atom->spin,         // atom data 7
                     atom->type,         // atom data 8
                     force->newton_pair, // force data 1
                     force->qqrd2e,      // force data 2
                     hhmss2e,            // eff data 1
                     h2e,                // eff data 2
                     PAULI_CORE_A,       // eff data 5
                     PAULI_CORE_B,       // eff data 6
                     PAULI_CORE_C,       // eff data 7
                     PAULI_CORE_D,       // eff data 8
                     PAULI_CORE_E,       // eff data 9
                     ecp_type,           // eff data 10
                     pvector,            // pair data 8
                     eng_coul,           // pair statistic data 1
                     eng_vdwl,           // pair statistic data 2
                     eatom,              // pair statistic data 3
                     vatom,              // pair statistic data 4
                     virial              // pair statistic data 5
  );
  cuda_DeallocateStructure(eff_gpu);
  if (vflag_fdotr) {
    virial_fdotr_compute();
    if (pressure_with_evirials_flag)
      virial_eff_compute();
  }
}

/* ----------------------------------------------------------------------
   eff-specific contribution to global virial
------------------------------------------------------------------------- */

void PairEffCutGPU::virial_eff_compute() {
  double *eradius = atom->eradius;
  double *erforce = atom->erforce;
  double e_virial;
  int *spin = atom->spin;

  // sum over force on all particles including ghosts

  if (neighbor->includegroup == 0) {
    int nall = atom->nlocal + atom->nghost;
    for (int i = 0; i < nall; i++) {
      if (spin[i]) {
        e_virial = erforce[i] * eradius[i] / 3;
        virial[0] += e_virial;
        virial[1] += e_virial;
        virial[2] += e_virial;
      }
    }

    // neighbor includegroup flag is set
    // sum over force on initial nfirst particles and ghosts

  } else {
    int nall = atom->nfirst;
    for (int i = 0; i < nall; i++) {
      if (spin[i]) {
        e_virial = erforce[i] * eradius[i] / 3;
        virial[0] += e_virial;
        virial[1] += e_virial;
        virial[2] += e_virial;
      }
    }

    nall = atom->nlocal + atom->nghost;
    for (int i = atom->nlocal; i < nall; i++) {
      if (spin[i]) {
        e_virial = erforce[i] * eradius[i] / 3;
        virial[0] += e_virial;
        virial[1] += e_virial;
        virial[2] += e_virial;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   tally eng_vdwl and virial into per-atom accumulators
   for virial radial electronic contributions
------------------------------------------------------------------------- */

void PairEffCutGPU::ev_tally_eff(int i, int j, int nlocal, int newton_pair,
                                 double energy, double e_virial) {
  double energyhalf;
  double partial_evirial = e_virial / 3.0;
  double half_partial_evirial = partial_evirial / 2;

  int *spin = atom->spin;

  if (eflag_either) {
    if (eflag_global) {
      if (newton_pair)
        eng_coul += energy;
      else {
        energyhalf = 0.5 * energy;
        if (i < nlocal)
          eng_coul += energyhalf;
        if (j < nlocal)
          eng_coul += energyhalf;
      }
    }
    if (eflag_atom) {
      if (newton_pair || i < nlocal)
        eatom[i] += 0.5 * energy;
      if (newton_pair || j < nlocal)
        eatom[j] += 0.5 * energy;
    }
  }

  if (vflag_either) {
    if (vflag_global) {
      if (spin[i] && i < nlocal) {
        virial[0] += half_partial_evirial;
        virial[1] += half_partial_evirial;
        virial[2] += half_partial_evirial;
      }
      if (spin[j] && j < nlocal) {
        virial[0] += half_partial_evirial;
        virial[1] += half_partial_evirial;
        virial[2] += half_partial_evirial;
      }
    }
    if (vflag_atom) {
      if (spin[i]) {
        if (newton_pair || i < nlocal) {
          vatom[i][0] += half_partial_evirial;
          vatom[i][1] += half_partial_evirial;
          vatom[i][2] += half_partial_evirial;
        }
      }
      if (spin[j]) {
        if (newton_pair || j < nlocal) {
          vatom[j][0] += half_partial_evirial;
          vatom[j][1] += half_partial_evirial;
          vatom[j][2] += half_partial_evirial;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairEffCutGPU::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");
}

/* ---------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairEffCutGPU::settings(int narg, char **arg) {
  if (narg < 1)
    error->all(FLERR, "Illegal pair_style command");

  // Defaults ECP parameters for C (radius=0.154)
  PAULI_CORE_A[6] = 22.721015;
  PAULI_CORE_B[6] = 0.728733;
  PAULI_CORE_C[6] = 1.103199;
  PAULI_CORE_D[6] = 17.695345;
  PAULI_CORE_E[6] = 6.693621;

  // Defaults ECP parameters for N (radius=0.394732)
  PAULI_CORE_A[7] = 16.242367;
  PAULI_CORE_B[7] = 0.602818;
  PAULI_CORE_C[7] = 1.081856;
  PAULI_CORE_D[7] = 7.150803;
  PAULI_CORE_E[7] = 5.351936;

  // Defaults p-element ECP parameters for Oxygen (radius=0.15)
  PAULI_CORE_A[8] = 29.5185;
  PAULI_CORE_B[8] = 0.32995;
  PAULI_CORE_C[8] = 1.21676;
  PAULI_CORE_D[8] = 11.98757;
  PAULI_CORE_E[8] = 3.073417;

  // Defaults ECP parameters for Al (radius=1.660)
  PAULI_CORE_A[13] = 0.486;
  PAULI_CORE_B[13] = 1.049;
  PAULI_CORE_C[13] = 0.207;
  PAULI_CORE_D[13] = 0.0;
  PAULI_CORE_E[13] = 0.0;

  // Defaults ECP parameters for Si (radius=1.691)
  PAULI_CORE_A[14] = 0.320852;
  PAULI_CORE_B[14] = 2.283269;
  PAULI_CORE_C[14] = 0.814857;
  PAULI_CORE_D[14] = 0.0;
  PAULI_CORE_E[14] = 0.0;

  cut_global = force->numeric(FLERR, arg[0]);
  limit_eradius_flag = 0;
  pressure_with_evirials_flag = 0;

  int atype;
  int iarg = 1;
  int ecp_found = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "limit/eradius") == 0) {
      limit_eradius_flag = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg], "pressure/evirials") == 0) {
      pressure_with_evirials_flag = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg], "ecp") == 0) {
      iarg += 1;
      while (iarg < narg) {
        atype = force->inumeric(FLERR, arg[iarg]);
        if (strcmp(arg[iarg + 1], "C") == 0)
          ecp_type[atype] = 6;
        else if (strcmp(arg[iarg + 1], "N") == 0)
          ecp_type[atype] = 7;
        else if (strcmp(arg[iarg + 1], "O") == 0)
          ecp_type[atype] = 8;
        else if (strcmp(arg[iarg + 1], "Al") == 0)
          ecp_type[atype] = 13;
        else if (strcmp(arg[iarg + 1], "Si") == 0)
          ecp_type[atype] = 14;
        else
          error->all(
              FLERR,
              "Note: there are no default parameters for this atom ECP\n");
        iarg += 2;
        ecp_found = 1;
      }
    }
  }

  if (!ecp_found && atom->ecp_flag)
    error->all(FLERR, "Need to specify ECP type on pair_style command");

  // Need to introduce 2 new constants w/out changing update.cpp
  if (force->qqr2e == 332.06371) {  // i.e. Real units chosen
    h2e = 627.509;                  // hartree->kcal/mol
    hhmss2e = 175.72044219620075;   // hartree->kcal/mol * (Bohr->Angstrom)^2
  } else if (force->qqr2e == 1.0) { // electron units
    h2e = 1.0;
    hhmss2e = 1.0;
  } else
    error->all(FLERR, "Check your units");

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j])
          cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairEffCutGPU::init_style() {
  // error and warning checks

  if (!atom->q_flag || !atom->spin_flag || !atom->eradius_flag ||
      !atom->erforce_flag)
    error->all(FLERR, "Pair eff/cut requires atom attributes "
                      "q, spin, eradius, erforce");

  // add hook to minimizer for eradius and erforce

  if (update->whichflag == 2)
    update->minimize->request(this, 1, 0.01);

  // make sure to use the appropriate timestep when using real units

  if (update->whichflag == 1) {
    if (force->qqr2e == 332.06371 && update->dt == 1.0)
      error->all(FLERR,
                 "You must lower the default real units timestep for pEFF ");
  }

  // need a half neigh list and optionally a granular history neigh list

  neighbor->request(this, instance_me);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type electron pairs (ECP-only)
------------------------------------------------------------------------- */

void PairEffCutGPU::coeff(int narg, char **arg) {
  if (!allocated)
    allocate();

  if ((strcmp(arg[0], "*") == 0) || (strcmp(arg[1], "*") == 0)) {
    int ilo, ihi, jlo, jhi;
    force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
    force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);

    double cut_one = cut_global;
    if (narg == 3)
      cut_one = force->numeric(FLERR, arg[2]);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo, i); j <= jhi; j++) {
        cut[i][j] = cut_one;
        setflag[i][j] = 1;
        count++;
      }
    }
    if (count == 0)
      error->all(FLERR, "Incorrect args for pair coefficients");
  } else {
    int ecp;
    ecp = force->inumeric(FLERR, arg[0]);
    if (strcmp(arg[1], "s") == 0) {
      PAULI_CORE_A[ecp_type[ecp]] = force->numeric(FLERR, arg[2]);
      PAULI_CORE_B[ecp_type[ecp]] = force->numeric(FLERR, arg[3]);
      PAULI_CORE_C[ecp_type[ecp]] = force->numeric(FLERR, arg[4]);
      PAULI_CORE_D[ecp_type[ecp]] = 0.0;
      PAULI_CORE_E[ecp_type[ecp]] = 0.0;
    } else if (strcmp(arg[1], "p") == 0) {
      PAULI_CORE_A[ecp_type[ecp]] = force->numeric(FLERR, arg[2]);
      PAULI_CORE_B[ecp_type[ecp]] = force->numeric(FLERR, arg[3]);
      PAULI_CORE_C[ecp_type[ecp]] = force->numeric(FLERR, arg[4]);
      PAULI_CORE_D[ecp_type[ecp]] = force->numeric(FLERR, arg[5]);
      PAULI_CORE_E[ecp_type[ecp]] = force->numeric(FLERR, arg[6]);
    } else
      error->all(FLERR, "Illegal pair_coeff command");
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairEffCutGPU::init_one(int i, int j) {
  if (setflag[i][j] == 0)
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairEffCutGPU::write_restart(FILE *fp) {
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j])
        fwrite(&cut[i][j], sizeof(double), 1, fp);
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairEffCutGPU::read_restart(FILE *fp) {
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0)
        utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, NULL, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0)
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, NULL, error);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairEffCutGPU::write_restart_settings(FILE *fp) {
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairEffCutGPU::read_restart_settings(FILE *fp) {
  if (comm->me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, NULL, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, NULL, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, NULL, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   returns pointers to the log() of electron radius and corresponding force
   minimizer operates on log(radius) so radius never goes negative
   these arrays are stored locally by pair style
------------------------------------------------------------------------- */

void PairEffCutGPU::min_xf_pointers(int /*ignore*/, double **xextra,
                                    double **fextra) {
  // grow arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(min_eradius);
    memory->destroy(min_erforce);
    nmax = atom->nmax;
    memory->create(min_eradius, nmax, "pair:min_eradius");
    memory->create(min_erforce, nmax, "pair:min_erforce");
  }

  *xextra = min_eradius;
  *fextra = min_erforce;
}

/* ----------------------------------------------------------------------
   minimizer requests the log() of electron radius and corresponding force
   calculate and store in min_eradius and min_erforce
------------------------------------------------------------------------- */

void PairEffCutGPU::min_xf_get(int /*ignore*/) {
  double *eradius = atom->eradius;
  double *erforce = atom->erforce;
  int *spin = atom->spin;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (spin[i]) {
      min_eradius[i] = log(eradius[i]);
      min_erforce[i] = eradius[i] * erforce[i];
    } else
      min_eradius[i] = min_erforce[i] = 0.0;
}

/* ----------------------------------------------------------------------
   minimizer has changed the log() of electron radius
   propagate the change back to eradius
------------------------------------------------------------------------- */

void PairEffCutGPU::min_x_set(int /*ignore*/) {
  double *eradius = atom->eradius;
  int *spin = atom->spin;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (spin[i])
      eradius[i] = exp(min_eradius[i]);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairEffCutGPU::memory_usage() {
  double bytes = maxeatom * sizeof(double);
  bytes += maxvatom * 6 * sizeof(double);
  bytes += 2 * nmax * sizeof(double);
  return bytes;
}
