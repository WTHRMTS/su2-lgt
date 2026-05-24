#pragma once
#include <vector>

#include "su2lgt/lattice.hpp"
#include "su2lgt/su2.hpp"

namespace su2lgt {

// ---- Wilson loop -------------------------------------------------------
//
// Half-trace of a single rectangular R x T Wilson loop in the (mu, nu)
// plane with starting corner `site`.  The loop is traversed
//   x  --(+mu, R times)-->  --(+nu, T times)-->  --(-mu, R times)-->
//      --(-nu, T times)-->  back to x.
//
// For SU(2) the result is a real number in [-1, 1].  Cold lattice gives 1.
double wilson_loop(const Lattice& lat, int site, int mu, int nu,
                   int R, int T);

// Average Wilson loop W(R, T) over (a) all starting sites, (b) the three
// spatial directions for mu (the leg of length R), and (c) the temporal
// direction for nu (the leg of length T).  This is the standard
// "spatial-temporal" Wilson loop used to extract the static-quark
// potential V(R) from the area-law decay
//
//      W(R, T)  ~  exp(-V(R) * T)   for large T.
double mean_spatial_temporal_wilson_loop(const Lattice& lat, int R, int T);


// ---- Polyakov-loop pair correlator -------------------------------------
//
// At finite T the pair correlator
//      P(r) = < l(x) * l(x + r) >
// gives the static potential via
//      P(r)  ~  exp(-V(r) / T_phys)
// (for r large enough that excited-state contributions have died off).
//
// We return the correlator averaged over (a) starting spatial site x,
// and (b) the three spatial axial directions, evaluated at integer
// separations r = 0, 1, ..., r_max.
//
// The local Polyakov-loop field is computed once internally (O(V_s * L_t))
// and the pair correlator falls out at O(V_s * r_max).
std::vector<double> polyakov_pair_correlator(const Lattice& lat, int r_max);


// ---- (Internal helper, exposed for testing) ----------------------------
//
// Local Polyakov loop l(x) at every spatial site, returned as a flat
// array of size L_x * L_y * L_z indexed by  (z * L_y + y) * L_x + x.
std::vector<double> polyakov_local_field(const Lattice& lat);

}  // namespace su2lgt
