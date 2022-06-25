#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include "eff_gpu.h"
#define CUDANEIGHMASK 0x3FFFFFFF

const __device__ double CUDA_E1[] = {1.483110564084803581889448079057,
                                     -3.01071073386594942470731046311E-1,
                                     6.8994830689831566246603180718E-2,
                                     -1.3916271264722187682546525687E-2,
                                     2.420799522433463662891678239E-3,
                                     -3.65863968584808644649382577E-4,
                                     4.8620984432319048282887568E-5,
                                     -5.749256558035684835054215E-6,
                                     6.11324357843476469706758E-7,
                                     -5.8991015312958434390846E-8,
                                     5.207009092068648240455E-9,
                                     -4.23297587996554326810E-10,
                                     3.1881135066491749748E-11,
                                     -2.236155018832684273E-12,
                                     1.46732984799108492E-13,
                                     -9.044001985381747E-15,
                                     5.25481371547092E-16,
                                     -2.8874261222849E-17,
                                     1.504785187558E-18,
                                     -7.4572892821E-20,
                                     3.522563810E-21,
                                     -1.58944644E-22,
                                     6.864365E-24,
                                     -2.84257E-25,
                                     1.1306E-26,
                                     -4.33E-28,
                                     1.6E-29,
                                     -1.0E-30};

const __device__ double CUDA_E2[] = {1.077977852072383151168335910348,
                                     -2.6559890409148673372146500904E-2,
                                     -1.487073146698099509605046333E-3,
                                     -1.38040145414143859607708920E-4,
                                     -1.1280303332287491498507366E-5,
                                     -1.172869842743725224053739E-6,
                                     -1.03476150393304615537382E-7,
                                     -1.1899114085892438254447E-8,
                                     -1.016222544989498640476E-9,
                                     -1.37895716146965692169E-10,
                                     -9.369613033737303335E-12,
                                     -1.918809583959525349E-12,
                                     -3.7573017201993707E-14,
                                     -3.7053726026983357E-14,
                                     2.627565423490371E-15,
                                     -1.121322876437933E-15,
                                     1.84136028922538E-16,
                                     -4.9130256574886E-17,
                                     1.0704455167373E-17,
                                     -2.671893662405E-18,
                                     6.49326867976E-19,
                                     -1.65399353183E-19,
                                     4.2605626604E-20,
                                     -1.1255840765E-20,
                                     3.025617448E-21,
                                     -8.29042146E-22,
                                     2.31049558E-22,
                                     -6.5469511E-23,
                                     1.8842314E-23,
                                     -5.504341E-24,
                                     1.630950E-24,
                                     -4.89860E-25,
                                     1.49054E-25,
                                     -4.5922E-26,
                                     1.4318E-26,
                                     -4.516E-27,
                                     1.440E-27,
                                     -4.64E-28,
                                     1.51E-28,
                                     -5.0E-29,
                                     1.7E-29,
                                     -6.0E-30,
                                     2.0E-30,
                                     -1.0E-30};

const __device__ double CUDA_DE1[] = {-0.689379974848418501361491576718,
                                      0.295939056851161774752959335568,
                                      -0.087237828075228616420029484096,
                                      0.019959734091835509766546612696,
                                      -0.003740200486895490324750329974,
                                      0.000593337912367800463413186784,
                                      -0.000081560801047403878256504204,
                                      9.886099179971884018535968E-6,
                                      -1.071209234904290565745194E-6,
                                      1.0490945447626050322784E-7,
                                      -9.370959271038746709966E-9,
                                      7.6927263488753841874E-10,
                                      -5.8412335114551520146E-11,
                                      4.125393291736424788E-12,
                                      -2.72304624901729048E-13,
                                      1.6869717361387012E-14,
                                      -9.84565340276638E-16,
                                      5.4313471880068E-17,
                                      -2.840458699772E-18,
                                      1.4120512798E-19,
                                      -6.688772574E-21,
                                      3.0257558E-22,
                                      -1.3097526E-23,
                                      5.4352E-25,
                                      -2.1704E-26,
                                      8.32E-28,
                                      -5.4E-29};

const __device__ double CUDA_DE2[] = {0.717710208167480928473053690384,
                                      -0.379868973985143305103199928808,
                                      0.125832094465157378967135019248,
                                      -0.030917661684228839423081992424,
                                      0.006073689914144320367855343072,
                                      -0.000996057789064916825079352632,
                                      0.000140310790466315733723475232,
                                      -0.000017328176496070286001302184,
                                      1.90540194670935746397168e-6,
                                      -1.8882873760163694937908e-7,
                                      1.703176613666840587056e-8,
                                      -1.40955218086201517976e-9,
                                      1.0776816914256065828e-10,
                                      -7.656138112778696256e-12,
                                      5.07943557413613792e-13,
                                      -3.1608615530282912e-14,
                                      1.852036572003432e-15,
                                      -1.02524641430496e-16,
                                      5.37852808112e-18,
                                      -2.68128238704e-19,
                                      1.273321788e-20,
                                      -5.77335744e-22,
                                      2.504352e-23,
                                      -1.0446e-24,
                                      4.16e-26,
                                      -2.808e-27};

__device__ double cuda_ipoly02(double x) {
  /* P(x) in the range x > 2 */
  int i;
  double b0, b1, b2;
  b1 = 0.0;
  b0 = 0.0;
  x *= 2;
  for (i = CUDA_ERF_TERMS2; i >= 0; i--) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + CUDA_E2[i];
  }
  return 0.5 * (b0 - b2);
}

__device__ double cuda_ipoly1(double x) {
  /* First derivative P'(x) in the range x < 2 */
  int i;
  double b0, b1, b2;

  b1 = 0.0;
  b0 = 0.0;
  x *= 2;
  for (i = CUDA_DERF_TERMS; i >= 0; i--) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + CUDA_DE1[i];
  }
  return 0.5 * (b0 - b2);
}

__device__ double cuda_ipoly01(double x) {
  // P(x) in the range x < 2

  int i;
  double b0, b1, b2;
  b1 = 0.0;
  b0 = 0.0;
  x *= 2;
  for (i = CUDA_ERF_TERMS1; i >= 0; i--) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + CUDA_E1[i];
  }
  return 0.5 * (b0 - b2);
}

__device__ double cuda_ierfoverx1(double x, double *df) {
  // Computes Erf(x)/x and its first derivative

  double t, f;
  double x2; // x squared
  double exp_term, recip_x;

  if (x < 2.0) {
    /* erf(x) = x * y(t)     */
    /* t = 2 * (x/2)^2 - 1.  */
    t = 0.5 * x * x - 1;
    f = cuda_ipoly01(t);
    *df = cuda_ipoly1(t) * x;
  } else {
    /* erf(x) = 1 - exp(-x^2)/x * y(t) */
    /* t = (10.5 - x^2) / (2.5 + x^2)  */
    x2 = x * x;
    t = (10.5 - x2) / (2.5 + x2);
    exp_term = exp(-x2);
    recip_x = 1.0 / x;
    f = 1.0 / x - (exp_term / x2) * cuda_ipoly02(t);
    *df = (1.12837916709551257389615890312 * exp_term - f) * recip_x;
  }
  return f;
}

__device__ void cuda_KinElec(double radius, double *eke, double *frc) {
  *eke += 1.5 / (radius * radius);
  *frc += 3.0 / (radius * radius * radius);
}

__device__ void cuda_ElecNucNuc(double q, double rc, double *ecoul,
                                double *frc) {
  *ecoul += q / rc;
  *frc += q / (rc * rc);
}

__device__ void cuda_ElecNucElec(double q, double rc, double re1, double *ecoul,
                                 double *frc, double *fre1) {
  double a, arc;
  double coeff_a;

  /* sqrt(2) */
  coeff_a = 1.4142135623730951;

  /* E = -Z/r Erf(a r / re) */
  /* constants: sqrt(2), 2 / sqrt(pi) */
  a = coeff_a / re1;
  arc = a * rc;

  /* Interaction between nuclear point charge and Gaussian electron */
  double E, dEdr, dEdr1, f, df;

  f = cuda_ierfoverx1(arc, &df);
  dEdr = q * a * a * df;
  dEdr1 = -q * (a / re1) * (f + arc * df);
  E = -q * a * f;

  *ecoul += E;
  *frc += dEdr;
  *fre1 += dEdr1;
}

__device__ void cuda_ElecElecElec(double rc, double re1, double re2,
                                  double *ecoul, double *frc, double *fre1,
                                  double *fre2) {
  double a, arc, re, fre;
  double coeff_a;

  /* sqrt(2) */
  coeff_a = 1.4142135623730951;

  re = sqrt(re1 * re1 + re2 * re2);

  /* constants: sqrt(2), 2 / sqrt(pi) */
  a = coeff_a / re;
  arc = a * rc;

  /* V_elecelec = E * F                              */
  /* where E = -Z/r Erf(a r / re)                    */
  /*       F = (1 - (b s + c s^2) exp(-d s^2))       */
  /* and s = r / re                                  */

  double E, dEdr, dEdr1, dEdr2, f, df;

  f = cuda_ierfoverx1(arc, &df);
  dEdr = -a * a * df;
  fre = a * (f + arc * df) / (re * re);
  dEdr1 = fre * re1;
  dEdr2 = fre * re2;

  E = a * f;

  *ecoul += E;
  *frc += dEdr;
  *fre1 += dEdr1;
  *fre2 += dEdr2;
}

__device__ void cuda_ElecCoreNuc(double q, double rc, double re1, double *ecoul,
                                 double *frc) {
  double a, arc;
  double coeff_a;
  double E, dEdr, df, f;

  coeff_a = 1.4142135623730951; /* sqrt(2) */
  a = coeff_a / re1;
  arc = a * rc;

  f = cuda_ierfoverx1(arc, &df);
  dEdr = -q * a * a * df;
  E = q * a * f;

  *ecoul += E;
  *frc += dEdr;
}

__device__ void cuda_ElecCoreCore(double q, double rc, double re1, double re2,
                                  double *ecoul, double *frc) {
  double a, arc, re;
  double coeff_a;
  double E, dEdr, f, df;

  coeff_a = 1.4142135623730951;

  re = sqrt(re1 * re1 + re2 * re2);
  a = coeff_a / re;
  arc = a * rc;

  f = cuda_ierfoverx1(arc, &df);
  dEdr = -q * a * a * df;
  E = q * a * f;

  *ecoul += E;
  *frc += dEdr;
}

__device__ void cuda_ElecCoreElec(double q, double rc, double re1, double re2,
                                  double *ecoul, double *frc, double *fre2) {
  double a, arc, re;
  double coeff_a;
  double E, dEdr, dEdr2, f, df, fre;

  /* sqrt(2) */
  coeff_a = 1.4142135623730951;

  /*
  re1: core size
  re2: electron size
  re3: size of the core, obtained from the electron density function rho(r) of
  core e.g. rho(r) = a1*exp(-((r)/b1)^2), a1 =157.9, b1 = 0.1441 -> re3 = 0.1441
  for Si4+
  */

  re = sqrt(re1 * re1 + re2 * re2);

  a = coeff_a / re;
  arc = a * rc;

  f = cuda_ierfoverx1(arc, &df);
  E = -q * a * f;
  dEdr = -q * a * df * a;
  fre = q * a * (f + arc * df) / (re * re);
  dEdr2 = fre * re2;

  *ecoul += E;
  *frc -= dEdr;
  *fre2 -= dEdr2;
}

__device__ void cuda_PauliElecElec(int samespin, double rc, double re1,
                                   double re2, double *epauli, double *frc,
                                   double *fre1, double *fre2) {
  double ree, rem;
  double S, t1, t2, tt;
  double dSdr1, dSdr2, dSdr;
  double dTdr1, dTdr2, dTdr;
  double O, dOdS, ratio;

  re1 *= CUDA_PAULI_RE;
  re2 *= CUDA_PAULI_RE;
  rc *= CUDA_PAULI_RC;
  ree = re1 * re1 + re2 * re2;
  rem = re1 * re1 - re2 * re2;

  S = (2.82842712474619 / pow((re2 / re1 + re1 / re2), 1.5)) *
      exp(-rc * rc / ree);

  t1 = 1.5 * (1 / (re1 * re1) + 1 / (re2 * re2));
  t2 = 2.0 * (3 * ree - 2 * rc * rc) / (ree * ree);
  tt = t1 - t2;

  dSdr1 = (-1.5 / re1) * (rem / ree) + 2 * re1 * rc * rc / (ree * ree);
  dSdr2 = (1.5 / re2) * (rem / ree) + 2 * re2 * rc * rc / (ree * ree);
  dSdr = -2 * rc / ree;
  dTdr1 = -3 / (re1 * re1 * re1) - 12 * re1 / (ree * ree) +
          8 * re1 * (-2 * rc * rc + 3 * ree) / (ree * ree * ree);
  dTdr2 = -3 / (re2 * re2 * re2) - 12 * re2 / (ree * ree) +
          8 * re2 * (-2 * rc * rc + 3 * ree) / (ree * ree * ree);
  dTdr = 8 * rc / (ree * ree);

  if (samespin == 1) {
    O = S * S / (1.0 - S * S) + (1 - CUDA_PAULI_RHO) * S * S / (1.0 + S * S);
    dOdS = 2 * S / ((1.0 - S * S) * (1.0 - S * S)) +
           (1 - CUDA_PAULI_RHO) * 2 * S / ((1.0 + S * S) * (1.0 + S * S));
  } else {
    O = -CUDA_PAULI_RHO * S * S / (1.0 + S * S);
    dOdS = -CUDA_PAULI_RHO * 2 * S / ((1.0 + S * S) * (1.0 + S * S));
  }

  ratio = tt * dOdS * S;
  *fre1 -= CUDA_PAULI_RE * (dTdr1 * O + ratio * dSdr1);
  *fre2 -= CUDA_PAULI_RE * (dTdr2 * O + ratio * dSdr2);
  *frc -= CUDA_PAULI_RC * (dTdr * O + ratio * dSdr);
  *epauli += tt * O;
}

__device__ void cuda_PauliCoreElec(double rc, double re2, double *epauli,
                                   double *frc, double *fre2,
                                   double PAULI_CORE_A, double PAULI_CORE_B,
                                   double PAULI_CORE_C) {
  double E, dEdrc, dEdre2, rcsq, ssq;

  rcsq = rc * rc;
  ssq = re2 * re2;

  E = PAULI_CORE_A * exp((-PAULI_CORE_B * rcsq) / (ssq + PAULI_CORE_C));

  dEdrc = -2 * PAULI_CORE_A * PAULI_CORE_B * rc *
          exp(-PAULI_CORE_B * rcsq / (ssq + PAULI_CORE_C)) /
          (ssq + PAULI_CORE_C);

  dEdre2 = 2 * PAULI_CORE_A * PAULI_CORE_B * re2 * rcsq *
           exp(-PAULI_CORE_B * rcsq / (ssq + PAULI_CORE_C)) /
           ((PAULI_CORE_C + ssq) * (PAULI_CORE_C + ssq));

  *epauli += E;
  *frc -= dEdrc;
  *fre2 -= dEdre2;
}

__device__ void
cuda_PauliCorePElec(double rc, double re2, double *epauli, double *frc,
                    double *fre2, double PAULI_CORE_P_A, double PAULI_CORE_P_B,
                    double PAULI_CORE_P_C, double PAULI_CORE_P_D,
                    double PAULI_CORE_P_E) {
  double E, dEdrc, dEdre2;

  E = PAULI_CORE_P_A *
      pow((2.0 / (PAULI_CORE_P_B / re2 + re2 / PAULI_CORE_P_B)), 5.0) *
      pow((rc - PAULI_CORE_P_C * re2), 2.0) *
      exp(-PAULI_CORE_P_D * pow((rc - PAULI_CORE_P_C * re2), 2.0) /
          (PAULI_CORE_P_E + re2 * re2));

  dEdrc = PAULI_CORE_P_A *
              pow((2.0 / (PAULI_CORE_P_B / re2 + re2 / PAULI_CORE_P_B)), 5.0) *
              2.0 * (rc - PAULI_CORE_P_C * re2) *
              exp(-PAULI_CORE_P_D * pow((rc - PAULI_CORE_P_C * re2), 2.0) /
                  (PAULI_CORE_P_E + re2 * re2)) +
          E * (-PAULI_CORE_P_D * 2.0 * (rc - PAULI_CORE_P_C * re2) /
               (PAULI_CORE_P_E + re2 * re2));

  dEdre2 = E * (-5.0 / (PAULI_CORE_P_B / re2 + re2 / PAULI_CORE_P_B) *
                (-PAULI_CORE_P_B / (re2 * re2) + 1.0 / PAULI_CORE_P_B)) +
           PAULI_CORE_P_A *
               pow((2.0 / (PAULI_CORE_P_B / re2 + re2 / PAULI_CORE_P_B)), 5.0) *
               2.0 * (rc - PAULI_CORE_P_C * re2) * (-PAULI_CORE_P_C) *
               exp(-PAULI_CORE_P_D * pow((rc - PAULI_CORE_P_C * re2), 2.0) /
                   (PAULI_CORE_P_E + re2 * re2)) +
           E * (2.0 * PAULI_CORE_P_D * (rc - PAULI_CORE_P_C * re2) *
                (PAULI_CORE_P_C * PAULI_CORE_P_E + rc * re2) /
                pow((PAULI_CORE_P_E + re2 * re2), 2.0));

  *epauli += E;
  *frc -= dEdrc;
  *fre2 -= dEdre2;
}

__device__ void cuda_RForce(double dx, double dy, double dz, double rc,
                            double force, double *fx, double *fy, double *fz) {
  force /= rc;
  *fx = force * dx;
  *fy = force * dy;
  *fz = force * dz;
}

__device__ void cuda_SmallRForce(double dx, double dy, double dz, double rc,
                                 double force, double *fx, double *fy,
                                 double *fz) {
  /* Handles case where rc is small to avoid division by zero */

  if (rc > 0.000001) {
    force /= rc;
    *fx = force * dx;
    *fy = force * dy;
    *fz = force * dz;
  } else {
    if (dx != 0)
      *fx = force / sqrt(1 + (dy * dy + dz * dz) / (dx * dx));
    else
      *fx = 0.0;
    if (dy != 0)
      *fy = force / sqrt(1 + (dx * dx + dz * dz) / (dy * dy));
    else
      *fy = 0.0;
    if (dz != 0)
      *fz = force / sqrt(1 + (dx * dx + dy * dy) / (dz * dz));
    else
      *fz = 0.0;
    //                if (dx < 0) *fx = -*fx;
    //                if (dy < 0) *fy = -*fy;
    //                if (dz < 0) *fz = -*fz;
  }
}

__device__ double cuda_cutoff(double x) {
  /*  cubic: return x * x * (2.0 * x - 3.0) + 1.0; */
  /*  quintic: return -6 * pow(x, 5) + 15 * pow(x, 4) - 10 * pow(x, 3) + 1; */

  /* Seventh order spline */
  //      return 20 * pow(x, 7) - 70 * pow(x, 6) + 84 * pow(x, 5) - 35 * pow(x,
  //      4) + 1;
  return (((20 * x - 70) * x + 84) * x - 35) * x * x * x * x + 1;
}

__device__ double cuda_dcutoff(double x) {
  /*  cubic: return (6.0 * x * x - 6.0 * x); */
  /*  quintic: return -30 * pow(x, 4) + 60 * pow(x, 3) - 30 * pow(x, 2); */

  /* Seventh order spline */
  //      return 140 * pow(x, 6) - 420 * pow(x, 5) + 420 * pow(x, 4) - 140 *
  //      pow(x, 3);
  return (((140 * x - 420) * x + 420) * x - 140) * x * x * x;
}

__global__ void kernel(int *arr, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    arr[i] += 1;
  }
}

__device__ void cuda_ev_tally_eff(int eflag_either, // Input
                                  int eflag_global, int eflag_atom,
                                  int vflag_either, int vflag_global,
                                  int vflag_atom, int i, int j, int spini,
                                  int spinj, int nlocal, int newton_pair,
                                  double energy, double e_virial,
                                  double &eng_coul, // Output
                                  double *eatom, double6d *vatom,
                                  double *virial) {
  double energyhalf;
  double partial_evirial = e_virial / 3.0;
  double half_partial_evirial = partial_evirial / 2;

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
      if (spini && i < nlocal) {
        virial[0] += half_partial_evirial;
        virial[1] += half_partial_evirial;
        virial[2] += half_partial_evirial;
      }
      if (spinj && j < nlocal) {
        virial[0] += half_partial_evirial;
        virial[1] += half_partial_evirial;
        virial[2] += half_partial_evirial;
      }
    }
    if (vflag_atom) {
      if (spini) {
        if (newton_pair || i < nlocal) {
          vatom[i][0] += half_partial_evirial;
          vatom[i][1] += half_partial_evirial;
          vatom[i][2] += half_partial_evirial;
        }
      }
      if (spinj) {
        if (newton_pair || j < nlocal) {
          vatom[j][0] += half_partial_evirial;
          vatom[j][1] += half_partial_evirial;
          vatom[j][2] += half_partial_evirial;
        }
      }
    }
  }
}

__device__ void cuda_ev_tally_xyz(int eflag_either, // Input
                                  int eflag_global, int eflag_atom,
                                  int vflag_either, int vflag_global,
                                  int vflag_atom, int i, int j, int nlocal,
                                  int newton_pair, double evdwl, double ecoul,
                                  double fx, double fy, double fz, double delx,
                                  double dely, double delz,
                                  double &eng_vdwl, // Output
                                  double &eng_coul, double *eatom,
                                  double *virial, double6d *vatom) {
  double evdwlhalf, ecoulhalf, epairhalf, v[6];

  if (eflag_either) {
    if (eflag_global) {
      if (newton_pair) {
        eng_vdwl += evdwl;
        eng_coul += ecoul;
      } else {
        evdwlhalf = 0.5 * evdwl;
        ecoulhalf = 0.5 * ecoul;
        if (i < nlocal) {
          eng_vdwl += evdwlhalf;
          eng_coul += ecoulhalf;
        }
        if (j < nlocal) {
          eng_vdwl += evdwlhalf;
          eng_coul += ecoulhalf;
        }
      }
    }
    if (eflag_atom) {
      epairhalf = 0.5 * (evdwl + ecoul);
      if (newton_pair || i < nlocal)
        eatom[i] += epairhalf;
      if (newton_pair || j < nlocal)
        eatom[j] += epairhalf;
    }
  }

  if (vflag_either) {
    v[0] = delx * fx;
    v[1] = dely * fy;
    v[2] = delz * fz;
    v[3] = delx * fy;
    v[4] = delx * fz;
    v[5] = dely * fz;

    if (vflag_global) {
      if (newton_pair) {
        virial[0] += v[0];
        virial[1] += v[1];
        virial[2] += v[2];
        virial[3] += v[3];
        virial[4] += v[4];
        virial[5] += v[5];
      } else {
        if (i < nlocal) {
          virial[0] += 0.5 * v[0];
          virial[1] += 0.5 * v[1];
          virial[2] += 0.5 * v[2];
          virial[3] += 0.5 * v[3];
          virial[4] += 0.5 * v[4];
          virial[5] += 0.5 * v[5];
        }
        if (j < nlocal) {
          virial[0] += 0.5 * v[0];
          virial[1] += 0.5 * v[1];
          virial[2] += 0.5 * v[2];
          virial[3] += 0.5 * v[3];
          virial[4] += 0.5 * v[4];
          virial[5] += 0.5 * v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_pair || i < nlocal) {
        vatom[i][0] += 0.5 * v[0];
        vatom[i][1] += 0.5 * v[1];
        vatom[i][2] += 0.5 * v[2];
        vatom[i][3] += 0.5 * v[3];
        vatom[i][4] += 0.5 * v[4];
        vatom[i][5] += 0.5 * v[5];
      }
      if (newton_pair || j < nlocal) {
        vatom[j][0] += 0.5 * v[0];
        vatom[j][1] += 0.5 * v[1];
        vatom[j][2] += 0.5 * v[2];
        vatom[j][3] += 0.5 * v[3];
        vatom[j][4] += 0.5 * v[4];
        vatom[j][5] += 0.5 * v[5];
      }
    }
  }
}

__global__ void cuda_compute_kernel(struct EFF_GPU &eff_gpu, int eflag,
                                    int vflag) {
  int ii, jj;
  int i, j;
  int jlist_begin;
  int jlist_end;
  int itype;
  int jtype;

  int natoms = eff_gpu.natoms_gpu;       // atom data 1
  double3d *x = eff_gpu.x_gpu;           // atom data 2
  double3d *f = eff_gpu.f_gpu;           // atom data 3
  double *q = eff_gpu.q_gpu;             // atom data 4
  double *erforce = eff_gpu.erforce_gpu; // atom data 5
  double *eradius = eff_gpu.eradius_gpu; // atom data 6
  int *spin = eff_gpu.spin_gpu;          // atom data 7
  int *type = eff_gpu.type_gpu;          // atom data 8
  int nlocal = eff_gpu.nlocal_gpu;       // atom data 9
  int ntypes = eff_gpu.ntypes_gpu;       // atom data 10

  int newton_pair = eff_gpu.newton_pair_gpu; // force data 1
  double qqrd2e = eff_gpu.qqrd2e_gpu;        // force data 2

  int inum = eff_gpu.inum_gpu;                        // list data 1
  int *ilist = eff_gpu.ilist_gpu;                     // list data 2
  int *numneigh = eff_gpu.numneigh_gpu;               // list data 3
  int *numneigh_offset = eff_gpu.numneigh_offset_gpu; // list data 4
  int *firstneigh = eff_gpu.firstneigh_gpu;           // list data 5

  int evflag = eff_gpu.evflag_gpu;             // pair data 1
  int eflag_either = eff_gpu.eflag_either_gpu; // pair data 2
  int eflag_global = eff_gpu.eflag_global_gpu; // pair data 3
  int eflag_atom = eff_gpu.eflag_atom_gpu;     // pair data 4
  int vflag_either = eff_gpu.vflag_either_gpu; // pair data 5
  int vflag_global = eff_gpu.vflag_global_gpu; // pair data 6
  int vflag_atom = eff_gpu.vflag_atom_gpu;     // pair data 7
  double *pvector = eff_gpu.pvector_gpu;       // pair data 8

  double eng_coul = eff_gpu.eng_coul_gpu; // pair statistic data 1
  double eng_vdwl = eff_gpu.eng_vdwl_gpu; // pair statistic data 2
  double *eatom = eff_gpu.eatom_gpu;      // pair statistic data 3
  double6d *vatom = eff_gpu.vatom_gpu;     // pair statistic data 4
  double *virial = eff_gpu.virial_gpu;    // pair statistic data 5

  double hhmss2e = eff_gpu.hhmss2e_gpu; // eff data 1
  double h2e = eff_gpu.h2e_gpu;         // eff data 2
  int pressure_with_evirials_flag =
      eff_gpu.pressure_with_evirials_flag_gpu;             // eff data 3
  double *cutsq = eff_gpu.cutsq_gpu;                       // eff_data 4
  double *PAULI_CORE_A = eff_gpu.PAULI_CORE_A_gpu;         // eff data 5
  double *PAULI_CORE_B = eff_gpu.PAULI_CORE_B_gpu;         // eff data 6
  double *PAULI_CORE_C = eff_gpu.PAULI_CORE_C_gpu;         // eff data 7
  double *PAULI_CORE_D = eff_gpu.PAULI_CORE_D_gpu;         // eff data 8
  double *PAULI_CORE_E = eff_gpu.PAULI_CORE_E_gpu;         // eff data 9
  int *ecp_type = eff_gpu.ecp_type_gpu;                    // eff data 10
  int limit_eradius_flag = eff_gpu.limit_eradius_flag_gpu; // eff data 11

  int domain_xperiodic = eff_gpu.domain_xperiodic_gpu; // domain_info data 1
  int domain_yperiodic = eff_gpu.domain_yperiodic_gpu; // domain_info data 2
  int domain_zperiodic = eff_gpu.domain_zperiodic_gpu; // domain_info data 3
  double domain_delx = eff_gpu.domain_delx_gpu;        // domain_info data 4
  double domain_dely = eff_gpu.domain_dely_gpu;        // domain_info data 5
  double domain_delz = eff_gpu.domain_delz_gpu;        // domain_info data 6

  double xtmp, ytmp, ztmp, delx, dely, delz, energy;
  double eke, ecoul, epauli, errestrain, halfcoul, halfpauli;
  double fpair, fx, fy, fz;
  double e1rforce, e2rforce, e1rvirial, e2rvirial;
  double s_fpair, s_e1rforce, s_e2rforce;
  double ecp_epauli, ecp_fpair, ecp_e1rforce, ecp_e2rforce;
  double rsq, rc;

  energy = eke = epauli = ecp_epauli = ecoul = errestrain = 0.0;

  // pvector = [KE, Pauli, ecoul, radial_restraint]
  for (i = 0; i < 4; i++)
    pvector[i] = 0.0;

  for (ii = threadIdx.x; ii < inum; ii += blockDim.x) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist_begin = numneigh_offset[i];
    jlist_end = numneigh_offset[i + 1];

    if (abs(spin[i]) == 1 || spin[i] == 2) {
      // reset energy and force temp variables
      eke = epauli = ecoul = 0.0;
      fpair = e1rforce = e2rforce = 0.0;
      s_fpair = 0.0;

      cuda_KinElec(eradius[i], &eke, &e1rforce);

      // Fixed-core
      if (spin[i] == 2) {
        // KE(2s)+Coul(1s-1s)+Coul(2s-nuclei)+Pauli(2s)
        eke *= 2;
        cuda_ElecNucElec(q[i], 0.0, eradius[i], &ecoul, &fpair, &e1rforce);
        cuda_ElecNucElec(q[i], 0.0, eradius[i], &ecoul, &fpair, &e1rforce);
        cuda_ElecElecElec(0.0, eradius[i], eradius[i], &ecoul, &fpair,
                          &e1rforce, &e2rforce);

        // opposite spin electron interactions
        cuda_PauliElecElec(0, 0.0, eradius[i], eradius[i], &epauli, &s_fpair,
                           &e1rforce, &e2rforce);

        // fix core electron size, i.e. don't contribute to ervirial
        e2rforce = e1rforce = 0.0;
      }

      // apply unit conversion factors
      eke *= hhmss2e;
      ecoul *= qqrd2e;
      fpair *= qqrd2e;
      epauli *= hhmss2e;
      s_fpair *= hhmss2e;
      e1rforce *= hhmss2e;

      // Sum up contributions
      energy = eke + epauli + ecoul;
      fpair = fpair + s_fpair;

      erforce[i] += e1rforce;

      // Tally energy and compute radial atomic virial contribution
      if (evflag) {
        cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom, vflag_either,
                          vflag_global, vflag_atom, i, i, spin[i], spin[i],
                          nlocal, newton_pair, energy, 0.0, eng_coul, eatom,
                          vatom, virial);
        if (pressure_with_evirials_flag)
          cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                            vflag_either, vflag_global, vflag_atom, i, i,
                            spin[i], spin[i], nlocal, newton_pair, 0.0,
                            e1rforce * eradius[i], eng_coul, eatom, vatom,
                            virial);
      }
      if (eflag_global) {
        pvector[0] += eke;
        pvector[1] += epauli;
        pvector[2] += ecoul;
      }
    }
    for (jj = jlist_begin; jj < jlist_end; jj++) {
      j = firstneigh[jj];
      j &= CUDANEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      rc = sqrt(rsq);

      jtype = type[j];

      if (rsq < cutsq[itype * (ntypes + 1) + jtype]) {

        energy = ecoul = epauli = ecp_epauli = 0.0;
        fx = fy = fz = fpair = s_fpair = ecp_fpair = 0.0;

        double taper = sqrt(cutsq[itype * (ntypes + 1) + jtype]);
        double dist = rc / taper;
        double spline = cuda_cutoff(dist);
        double dspline = cuda_dcutoff(dist) / taper;

        // nucleus (i) - nucleus (j) Coul interaction

        if (spin[i] == 0 && spin[j] == 0) {
          double qxq = q[i] * q[j];

          cuda_ElecNucNuc(qxq, rc, &ecoul, &fpair);
        }

        // fixed-core (i) - nucleus (j) nuclear Coul interaction
        else if (spin[i] == 2 && spin[j] == 0) {
          double qxq = q[i] * q[j];
          e1rforce = 0.0;

          cuda_ElecNucNuc(qxq, rc, &ecoul, &fpair);
          cuda_ElecNucElec(q[j], rc, eradius[i], &ecoul, &fpair, &e1rforce);
          cuda_ElecNucElec(q[j], rc, eradius[i], &ecoul, &fpair, &e1rforce);
        }

        // nucleus (i) - fixed-core (j) nuclear Coul interaction
        else if (spin[i] == 0 && spin[j] == 2) {
          double qxq = q[i] * q[j];
          e1rforce = 0.0;

          cuda_ElecNucNuc(qxq, rc, &ecoul, &fpair);
          cuda_ElecNucElec(q[i], rc, eradius[j], &ecoul, &fpair, &e1rforce);
          cuda_ElecNucElec(q[i], rc, eradius[j], &ecoul, &fpair, &e1rforce);
        }

        // pseudo-core nucleus (i) - nucleus (j) interaction
        else if (spin[i] == 3 && spin[j] == 0) {
          double qxq = q[i] * q[j];

          cuda_ElecCoreNuc(qxq, rc, eradius[i], &ecoul, &fpair);
        }

        else if (spin[i] == 4 && spin[j] == 0) {
          double qxq = q[i] * q[j];

          cuda_ElecCoreNuc(qxq, rc, eradius[i], &ecoul, &fpair);
        }

        // nucleus (i) - pseudo-core nucleus (j) interaction
        else if (spin[i] == 0 && spin[j] == 3) {
          double qxq = q[i] * q[j];

          cuda_ElecCoreNuc(qxq, rc, eradius[j], &ecoul, &fpair);
        }

        else if (spin[i] == 0 && spin[j] == 4) {
          double qxq = q[i] * q[j];

          cuda_ElecCoreNuc(qxq, rc, eradius[j], &ecoul, &fpair);
        }

        // nucleus (i) - electron (j) Coul interaction

        else if (spin[i] == 0 && abs(spin[j]) == 1) {
          e1rforce = 0.0;

          cuda_ElecNucElec(q[i], rc, eradius[j], &ecoul, &fpair, &e1rforce);

          e1rforce = spline * qqrd2e * e1rforce;
          erforce[j] += e1rforce;

          // Radial electron virial, iff flexible pressure flag set
          if (evflag && pressure_with_evirials_flag) {
            e1rvirial = eradius[j] * e1rforce;

            cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                              vflag_either, vflag_global, vflag_atom, j, j,
                              spin[j], spin[j], nlocal, newton_pair, 0.0,
                              e1rvirial, eng_coul, eatom, vatom, virial);
          }
        }

        // electron (i) - nucleus (j) Coul interaction

        else if (abs(spin[i]) == 1 && spin[j] == 0) {
          e1rforce = 0.0;

          cuda_ElecNucElec(q[j], rc, eradius[i], &ecoul, &fpair, &e1rforce);

          e1rforce = spline * qqrd2e * e1rforce;
          erforce[i] += e1rforce;

          // Radial electron virial, iff flexible pressure flag set
          if (evflag && pressure_with_evirials_flag) {
            e1rvirial = eradius[i] * e1rforce;
            cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                              vflag_either, vflag_global, vflag_atom, i, i,
                              spin[i], spin[i], nlocal, newton_pair, 0.0,
                              e1rvirial, eng_coul, eatom, vatom, virial);
          }
        }

        // electron (i) - electron (j) interactions

        else if (abs(spin[i]) == 1 && abs(spin[j]) == 1) {
          e1rforce = e2rforce = 0.0;
          s_e1rforce = s_e2rforce = 0.0;

          cuda_ElecElecElec(rc, eradius[i], eradius[j], &ecoul, &fpair,
                            &e1rforce, &e2rforce);
          cuda_PauliElecElec(spin[i] == spin[j], rc, eradius[i], eradius[j],
                             &epauli, &s_fpair, &s_e1rforce, &s_e2rforce);

          // Apply conversion factor
          epauli *= hhmss2e;
          s_fpair *= hhmss2e;

          e1rforce = spline * (qqrd2e * e1rforce + hhmss2e * s_e1rforce);
          erforce[i] += e1rforce;
          e2rforce = spline * (qqrd2e * e2rforce + hhmss2e * s_e2rforce);
          erforce[j] += e2rforce;

          // Radial electron virial, iff flexible pressure flag set
          if (evflag && pressure_with_evirials_flag) {
            e1rvirial = eradius[i] * e1rforce;
            e2rvirial = eradius[j] * e2rforce;
            cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                              vflag_either, vflag_global, vflag_atom, i, j,
                              spin[i], spin[j], nlocal, newton_pair, 0.0,
                              e1rvirial + e2rvirial, eng_coul, eatom, vatom,
                              virial);
          }
        }

        // fixed-core (i) - electron (j) interactions

        else if (spin[i] == 2 && abs(spin[j]) == 1) {
          e1rforce = e2rforce = 0.0;
          s_e1rforce = s_e2rforce = 0.0;

          cuda_ElecNucElec(q[i], rc, eradius[j], &ecoul, &fpair, &e2rforce);
          cuda_ElecElecElec(rc, eradius[i], eradius[j], &ecoul, &fpair,
                            &e1rforce, &e2rforce);
          cuda_ElecElecElec(rc, eradius[i], eradius[j], &ecoul, &fpair,
                            &e1rforce, &e2rforce);
          cuda_PauliElecElec(0, rc, eradius[i], eradius[j], &epauli, &s_fpair,
                             &s_e1rforce, &s_e2rforce);
          cuda_PauliElecElec(1, rc, eradius[i], eradius[j], &epauli, &s_fpair,
                             &s_e1rforce, &s_e2rforce);

          // Apply conversion factor
          epauli *= hhmss2e;
          s_fpair *= hhmss2e;

          // only update virial for j electron
          e2rforce = spline * (qqrd2e * e2rforce + hhmss2e * s_e2rforce);
          erforce[j] += e2rforce;

          // Radial electron virial, iff flexible pressure flag set
          if (evflag && pressure_with_evirials_flag) {
            e2rvirial = eradius[j] * e2rforce;
            cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                              vflag_either, vflag_global, vflag_atom, j, j,
                              spin[j], spin[j], nlocal, newton_pair, 0.0,
                              e2rvirial, eng_coul, eatom, vatom, virial);
          }
        }

        // electron (i) - fixed-core (j) interactions

        else if (abs(spin[i]) == 1 && spin[j] == 2) {
          e1rforce = e2rforce = 0.0;
          s_e1rforce = s_e2rforce = 0.0;

          cuda_ElecNucElec(q[j], rc, eradius[i], &ecoul, &fpair, &e2rforce);
          cuda_ElecElecElec(rc, eradius[j], eradius[i], &ecoul, &fpair,
                            &e1rforce, &e2rforce);
          cuda_ElecElecElec(rc, eradius[j], eradius[i], &ecoul, &fpair,
                            &e1rforce, &e2rforce);

          cuda_PauliElecElec(0, rc, eradius[j], eradius[i], &epauli, &s_fpair,
                             &s_e1rforce, &s_e2rforce);
          cuda_PauliElecElec(1, rc, eradius[j], eradius[i], &epauli, &s_fpair,
                             &s_e1rforce, &s_e2rforce);

          // Apply conversion factor
          epauli *= hhmss2e;
          s_fpair *= hhmss2e;

          // only update virial for i electron
          e2rforce = spline * (qqrd2e * e2rforce + hhmss2e * s_e2rforce);
          erforce[i] += e2rforce;

          // add radial atomic virial, iff flexible pressure flag set
          if (evflag && pressure_with_evirials_flag) {
            e2rvirial = eradius[i] * e2rforce;
            cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                              vflag_either, vflag_global, vflag_atom, i, i,
                              spin[i], spin[i], nlocal, newton_pair, 0.0,
                              e2rvirial, eng_coul, eatom, vatom, virial);
          }
        }

        // fixed-core (i) - fixed-core (j) interactions

        else if (spin[i] == 2 && spin[j] == 2) {
          e1rforce = e2rforce = 0.0;
          s_e1rforce = s_e2rforce = 0.0;
          double qxq = q[i] * q[j];

          cuda_ElecNucNuc(qxq, rc, &ecoul, &fpair);
          cuda_ElecNucElec(q[i], rc, eradius[j], &ecoul, &fpair, &e1rforce);
          cuda_ElecNucElec(q[i], rc, eradius[j], &ecoul, &fpair, &e1rforce);
          cuda_ElecNucElec(q[j], rc, eradius[i], &ecoul, &fpair, &e1rforce);
          cuda_ElecNucElec(q[j], rc, eradius[i], &ecoul, &fpair, &e1rforce);
          cuda_ElecElecElec(rc, eradius[i], eradius[j], &ecoul, &fpair,
                            &e1rforce, &e2rforce);
          cuda_ElecElecElec(rc, eradius[i], eradius[j], &ecoul, &fpair,
                            &e1rforce, &e2rforce);
          cuda_ElecElecElec(rc, eradius[i], eradius[j], &ecoul, &fpair,
                            &e1rforce, &e2rforce);
          cuda_ElecElecElec(rc, eradius[i], eradius[j], &ecoul, &fpair,
                            &e1rforce, &e2rforce);

          cuda_PauliElecElec(0, rc, eradius[i], eradius[j], &epauli, &s_fpair,
                             &s_e1rforce, &s_e2rforce);
          cuda_PauliElecElec(1, rc, eradius[i], eradius[j], &epauli, &s_fpair,
                             &s_e1rforce, &s_e2rforce);
          epauli *= 2;
          s_fpair *= 2;

          // Apply conversion factor
          epauli *= hhmss2e;
          s_fpair *= hhmss2e;
        }

        // pseudo-core (i) - electron/fixed-core electrons (j) interactions

        else if (spin[i] == 3 && (abs(spin[j]) == 1 || spin[j] == 2)) {
          e2rforce = ecp_e2rforce = 0.0;

          if (((PAULI_CORE_D[ecp_type[itype]]) == 0.0) &&
              ((PAULI_CORE_E[ecp_type[itype]]) == 0.0)) {
            if (abs(spin[j]) == 1) {
              cuda_ElecCoreElec(q[i], rc, eradius[i], eradius[j], &ecoul,
                                &fpair, &e2rforce);
              cuda_PauliCoreElec(rc, eradius[j], &ecp_epauli, &ecp_fpair,
                                 &ecp_e2rforce, PAULI_CORE_A[ecp_type[itype]],
                                 PAULI_CORE_B[ecp_type[itype]],
                                 PAULI_CORE_C[ecp_type[itype]]);
            } else { // add second s electron contribution from fixed-core
              double qxq = q[i] * q[j];
              cuda_ElecCoreNuc(qxq, rc, eradius[j], &ecoul, &fpair);
              cuda_ElecCoreElec(q[i], rc, eradius[i], eradius[j], &ecoul,
                                &fpair, &e2rforce);
              cuda_ElecCoreElec(q[i], rc, eradius[i], eradius[j], &ecoul,
                                &fpair, &e2rforce);
              cuda_PauliCoreElec(rc, eradius[j], &ecp_epauli, &ecp_fpair,
                                 &ecp_e2rforce, PAULI_CORE_A[ecp_type[itype]],
                                 PAULI_CORE_B[ecp_type[itype]],
                                 PAULI_CORE_C[ecp_type[itype]]);
              cuda_PauliCoreElec(rc, eradius[j], &ecp_epauli, &ecp_fpair,
                                 &ecp_e2rforce, PAULI_CORE_A[ecp_type[itype]],
                                 PAULI_CORE_B[ecp_type[itype]],
                                 PAULI_CORE_C[ecp_type[itype]]);
            }
          } else {
            if (abs(spin[j]) == 1) {
              cuda_ElecCoreElec(q[i], rc, eradius[i], eradius[j], &ecoul,
                                &fpair, &e2rforce);
              cuda_PauliCorePElec(
                  rc, eradius[j], &ecp_epauli, &ecp_fpair, &ecp_e2rforce,
                  PAULI_CORE_A[ecp_type[itype]], PAULI_CORE_B[ecp_type[itype]],
                  PAULI_CORE_C[ecp_type[itype]], PAULI_CORE_D[ecp_type[itype]],
                  PAULI_CORE_E[ecp_type[itype]]);
            } else { // add second s electron contribution from fixed-core
              double qxq = q[i] * q[j];
              cuda_ElecCoreNuc(qxq, rc, eradius[j], &ecoul, &fpair);
              cuda_ElecCoreElec(q[i], rc, eradius[i], eradius[j], &ecoul,
                                &fpair, &e2rforce);
              cuda_ElecCoreElec(q[i], rc, eradius[i], eradius[j], &ecoul,
                                &fpair, &e2rforce);
              cuda_PauliCorePElec(
                  rc, eradius[j], &ecp_epauli, &ecp_fpair, &ecp_e2rforce,
                  PAULI_CORE_A[ecp_type[itype]], PAULI_CORE_B[ecp_type[itype]],
                  PAULI_CORE_C[ecp_type[itype]], PAULI_CORE_D[ecp_type[itype]],
                  PAULI_CORE_E[ecp_type[itype]]);
              cuda_PauliCorePElec(
                  rc, eradius[j], &ecp_epauli, &ecp_fpair, &ecp_e2rforce,
                  PAULI_CORE_A[ecp_type[itype]], PAULI_CORE_B[ecp_type[itype]],
                  PAULI_CORE_C[ecp_type[itype]], PAULI_CORE_D[ecp_type[itype]],
                  PAULI_CORE_E[ecp_type[itype]]);
            }
          }

          // Apply conversion factor from Hartree to kcal/mol
          ecp_epauli *= h2e;
          ecp_fpair *= h2e;

          // only update virial for j electron
          e2rforce = spline * (qqrd2e * e2rforce + h2e * ecp_e2rforce);
          erforce[j] += e2rforce;

          // add radial atomic virial, iff flexible pressure flag set
          if (evflag && pressure_with_evirials_flag) {
            e2rvirial = eradius[j] * e2rforce;
            cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                              vflag_either, vflag_global, vflag_atom, j, j,
                              spin[j], spin[j], nlocal, newton_pair, 0.0,
                              e2rvirial, eng_coul, eatom, vatom, virial);
          }
        }

        // electron/fixed-core electrons (i) - pseudo-core (j) interactions

        else if ((abs(spin[i]) == 1 || spin[i] == 2) && spin[j] == 3) {
          e1rforce = ecp_e1rforce = 0.0;

          if (((PAULI_CORE_D[ecp_type[jtype]]) == 0.0) &&
              ((PAULI_CORE_E[ecp_type[jtype]]) == 0.0)) {
            if (abs(spin[i]) == 1) {
              cuda_ElecCoreElec(q[j], rc, eradius[j], eradius[i], &ecoul,
                                &fpair, &e1rforce);
              cuda_PauliCoreElec(rc, eradius[i], &ecp_epauli, &ecp_fpair,
                                 &ecp_e1rforce, PAULI_CORE_A[ecp_type[jtype]],
                                 PAULI_CORE_B[ecp_type[jtype]],
                                 PAULI_CORE_C[ecp_type[jtype]]);
            } else {
              double qxq = q[i] * q[j];
              cuda_ElecCoreNuc(qxq, rc, eradius[i], &ecoul, &fpair);
              cuda_ElecCoreElec(q[j], rc, eradius[j], eradius[i], &ecoul,
                                &fpair, &e1rforce);
              cuda_ElecCoreElec(q[j], rc, eradius[j], eradius[i], &ecoul,
                                &fpair, &e1rforce);
              cuda_PauliCoreElec(rc, eradius[i], &ecp_epauli, &ecp_fpair,
                                 &ecp_e1rforce, PAULI_CORE_A[ecp_type[jtype]],
                                 PAULI_CORE_B[ecp_type[jtype]],
                                 PAULI_CORE_C[ecp_type[jtype]]);
              cuda_PauliCoreElec(rc, eradius[i], &ecp_epauli, &ecp_fpair,
                                 &ecp_e1rforce, PAULI_CORE_A[ecp_type[jtype]],
                                 PAULI_CORE_B[ecp_type[jtype]],
                                 PAULI_CORE_C[ecp_type[jtype]]);
            }
          } else {
            if (abs(spin[i]) == 1) {
              cuda_ElecCoreElec(q[j], rc, eradius[j], eradius[i], &ecoul,
                                &fpair, &e1rforce);
              cuda_PauliCorePElec(
                  rc, eradius[i], &ecp_epauli, &ecp_fpair, &ecp_e1rforce,
                  PAULI_CORE_A[ecp_type[jtype]], PAULI_CORE_B[ecp_type[jtype]],
                  PAULI_CORE_C[ecp_type[jtype]], PAULI_CORE_D[ecp_type[jtype]],
                  PAULI_CORE_E[ecp_type[jtype]]);
            } else {
              double qxq = q[i] * q[j];
              cuda_ElecCoreNuc(qxq, rc, eradius[i], &ecoul, &fpair);
              cuda_ElecCoreElec(q[j], rc, eradius[j], eradius[i], &ecoul,
                                &fpair, &e1rforce);
              cuda_ElecCoreElec(q[j], rc, eradius[j], eradius[i], &ecoul,
                                &fpair, &e1rforce);
              cuda_PauliCorePElec(
                  rc, eradius[i], &ecp_epauli, &ecp_fpair, &ecp_e1rforce,
                  PAULI_CORE_A[ecp_type[jtype]], PAULI_CORE_B[ecp_type[jtype]],
                  PAULI_CORE_C[ecp_type[jtype]], PAULI_CORE_D[ecp_type[jtype]],
                  PAULI_CORE_E[ecp_type[jtype]]);
              cuda_PauliCorePElec(
                  rc, eradius[i], &ecp_epauli, &ecp_fpair, &ecp_e1rforce,
                  PAULI_CORE_A[ecp_type[jtype]], PAULI_CORE_B[ecp_type[jtype]],
                  PAULI_CORE_C[ecp_type[jtype]], PAULI_CORE_D[ecp_type[jtype]],
                  PAULI_CORE_E[ecp_type[jtype]]);
            }
          }

          // Apply conversion factor from Hartree to kcal/mol
          ecp_epauli *= h2e;
          ecp_fpair *= h2e;

          // only update virial for j electron
          e1rforce = spline * (qqrd2e * e1rforce + h2e * ecp_e1rforce);
          erforce[i] += e1rforce;

          // add radial atomic virial, iff flexible pressure flag set
          if (evflag && pressure_with_evirials_flag) {
            e1rvirial = eradius[i] * e1rforce;
            cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                              vflag_either, vflag_global, vflag_atom, i, i,
                              spin[i], spin[i], nlocal, newton_pair, 0.0,
                              e1rvirial, eng_coul, eatom, vatom, virial);
          }
        }

        // pseudo-core (i) - pseudo-core (j) interactions

        else if (spin[i] == 3 && spin[j] == 3) {
          double qxq = q[i] * q[j];

          cuda_ElecCoreCore(qxq, rc, eradius[i], eradius[j], &ecoul, &fpair);
        }

        // Apply Coulomb conversion factor for all cases
        ecoul *= qqrd2e;
        fpair *= qqrd2e;

        // Sum up energy and force contributions
        epauli += ecp_epauli;
        energy = ecoul + epauli;
        fpair = fpair + s_fpair + ecp_fpair;

        // Apply cutoff spline
        fpair = fpair * spline - energy * dspline;
        energy = spline * energy;

        // Tally cartesian forces
        cuda_SmallRForce(delx, dely, delz, rc, fpair, &fx, &fy, &fz);
        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        if (newton_pair || j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
        }

        // Tally energy (in ecoul) and compute normal pressure virials
        if (evflag)
          cuda_ev_tally_xyz(eflag_either, eflag_global, eflag_atom,
                            vflag_either, vflag_global, vflag_atom, i, j,
                            nlocal, newton_pair, 0.0, energy, fx, fy, fz, delx,
                            dely, delz, eng_vdwl, eng_coul, eatom, virial,
                            vatom);
        if (eflag_global) {
          if (newton_pair) {
            pvector[1] += spline * epauli;
            pvector[2] += spline * ecoul;
          } else {
            halfpauli = 0.5 * spline * epauli;
            halfcoul = 0.5 * spline * ecoul;
            if (i < nlocal) {
              pvector[1] += halfpauli;
              pvector[2] += halfcoul;
            }
            if (j < nlocal) {
              pvector[1] += halfpauli;
              pvector[2] += halfcoul;
            }
          }
        }
      }
    }

    // limit electron stifness (size) for periodic systems, to max=half-box-size

    if (abs(spin[i]) == 1 && limit_eradius_flag) {
      double half_box_length = 0, dr, kfactor = hhmss2e * 1.0;
      e1rforce = errestrain = 0.0;

      if (domain_xperiodic == 1 || domain_yperiodic == 1 ||
          domain_zperiodic == 1) {
        // delx = domain->boxhi[0]-dom  ain->boxlo[0];
        // dely = domain->boxhi[1]-domain->boxlo[1];
        // delz = domain->boxhi[2]-domain->boxlo[2];
        delx = domain_delx;
        dely = domain_dely;
        delz = domain_delz;
        half_box_length = 0.5 * min(delx, min(dely, delz));
        if (eradius[i] > half_box_length) {
          dr = eradius[i] - half_box_length;
          errestrain = 0.5 * kfactor * dr * dr;
          e1rforce = -kfactor * dr;
          if (eflag_global)
            pvector[3] += errestrain;

          erforce[i] += e1rforce;

          // Tally radial restrain energy and add radial restrain virial
          if (evflag) {
            cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                              vflag_either, vflag_global, vflag_atom, i, i,
                              spin[i], spin[i], nlocal, newton_pair, errestrain,
                              0.0, eng_coul, eatom, vatom, virial);
            if (pressure_with_evirials_flag) // flexible electron pressure
              cuda_ev_tally_eff(eflag_either, eflag_global, eflag_atom,
                                vflag_either, vflag_global, vflag_atom, i, i,
                                spin[i], spin[i], nlocal, newton_pair, 0.0,
                                eradius[i] * e1rforce, eng_coul, eatom, vatom,
                                virial);
          }
        }
      }
    }
  }
}

void cuda_data_test(struct EFF_GPU &eff_gpu) {
  int index;
  printf("GPU data\n");
  printf("x:\n");
  for (index = 0; index < 10; index++) {
    printf("%lf ", (eff_gpu.x_gpu[index][0]));
  }
  printf("\n");
  printf("f\n");
  for (index = 0; index < 10; index++) {
    printf("%lf ", (eff_gpu.f_gpu[index][0]));
  }
  printf("\n");
  printf("\n");
}

__host__ void cuda_eff_test(struct EFF_GPU &eff_gpu, int eflag, int vflag) {
  // cuda_compute_kernel<<<1, 256>>>(eff_gpu, eflag, vflag);
  cuda_data_test(eff_gpu);
  cudaDeviceSynchronize();
}
