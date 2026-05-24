#pragma once
#include <vector>

#include "su2lgt/lattice.hpp"

namespace su2lgt {

// Spatial position-space gauge-boson propagator on a Landau-gauge-fixed
// lattice.  Returns G[r] for r in [0, r_max] with
//
//   G[r] = (1 / (3 V_s)) * sum_{x_s} sum_{a, mu} (
//           Abar^a_mu(x_s) * Abar^a_mu(x_s + r * n_hat) )
//   averaged over the three spatial axes  n_hat in {x, y, z}.
//
// The "static" t-average (zero Matsubara projection)
//
//     Abar^a_mu(x_s) = (1 / N_t) sum_t A^a_mu(x_s, t)
//
// is taken first.  This is the static-sector propagator, which is the
// quantity that couples to the Cooper-pair gap function in the BCS
// integral equation.
//
// G[0] equals  (1 / V_s) sum_{x_s} sum_{a, mu} Abar^a_mu(x_s)^2
// and is positive (= the squared algebra-norm density).  For massive
// boson channels G[r] decays approximately as exp(-m r) at long range.
//
// IMPORTANT: the propagator is gauge-dependent.  The caller is
// responsible for applying landau_gauge_fix(lat) before calling this
// function.  On a gauge-fixed configuration the result is the
// canonical Landau-gauge propagator (modulo the Gribov-copy
// ambiguity inherited from the gauge fixer).
std::vector<double> static_propagator(const Lattice& lat, int r_max);

}  // namespace su2lgt
