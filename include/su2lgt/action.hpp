#pragma once
#include "su2lgt/lattice.hpp"
#include "su2lgt/su2.hpp"

namespace su2lgt {

// ----------------------------------------------------------------------
//  Wilson plaquette action for pure SU(2) lattice gauge theory.
//
//      S = beta * sum_p ( 1 - (1/2) Re Tr U_p )
//
//  with the plaquette
//
//      U_{mu nu}(x) = U_mu(x) U_nu(x + mu) U_mu^dag(x + nu) U_nu^dag(x).
//
//  All measurements are in 4D Euclidean lattice with periodic BCs.
// ----------------------------------------------------------------------

// Number of distinct plaquettes on the lattice.  6 planes (mu < nu) per
// site, periodic in every direction.
inline int num_plaquettes(const Lattice& lat) {
    return 6 * lat.volume();
}

// Plaquette product as an SU(2) element (unit norm to roundoff).
SU2 plaquette(const Lattice& lat, int site, int mu, int nu);

// Sum over all plaquettes of (1/2) Re Tr U_p = sum of plaquette.a0.
// On a cold lattice this equals num_plaquettes; on a fully disordered
// lattice it fluctuates around 0.
double total_plaquette_trace(const Lattice& lat);

// Wilson action S = beta * (Np - sum_p (Up).a0).
inline double plaquette_action(const Lattice& lat, double beta) {
    return beta * (num_plaquettes(lat) - total_plaquette_trace(lat));
}

// Mean plaquette in [-1, 1]: < (1/2) Re Tr U_p >.
inline double mean_plaquette(const Lattice& lat) {
    return total_plaquette_trace(lat) / num_plaquettes(lat);
}

// Staple sum A_mu(x) for the link U_mu(x):
//
//   A_mu(x) =  sum_{nu != mu} [
//                U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)            (forward)
//              + U_nu^dag(x+mu-nu) U_mu^dag(x-nu) U_nu(x-nu)      (backward)
//              ]
//
// Stored in an SU2 struct purely as a 4-tuple: it is a sum of SU(2)
// matrices and therefore NOT unit norm, but the multiplication formula
// in `operator*` is bilinear and works correctly for it.
SU2 staple_sum(const Lattice& lat, int site, int mu);

// Local change in action when the link at (site, mu) is changed
// from U_old to U_new and nothing else changes.  Uses only the
// precomputed staple:
//
//      Delta S = -beta * ( (U_new * staple).a0 - (U_old * staple).a0 ).
//
// O(1) work, independent of lattice volume.
double delta_action_from_staple(double beta,
                                const SU2& U_old,
                                const SU2& U_new,
                                const SU2& staple);

}  // namespace su2lgt
