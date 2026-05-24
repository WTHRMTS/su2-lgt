#pragma once
#include <vector>

#include "su2lgt/lattice.hpp"
#include "su2lgt/su2.hpp"

namespace su2lgt {

// Result of a Landau gauge-fixing run.
struct GaugeFixResult {
    int    iterations    = 0;
    double F_initial     = 0.0;     // pre-fixing functional value
    double F_final       = 0.0;     // post-fixing functional value
    double F_max         = 0.0;     // theoretical upper bound = 4 * volume
    double theta_initial = 0.0;     // pre-fixing residual
    double theta_final   = 0.0;     // post-fixing residual
    bool   converged     = false;
};

// Maximise the Landau functional
//
//     F[U] = sum_{x, mu} (1/2) Re Tr U_mu(x)  =  sum a0(U_mu(x))
//
// by Cabibbo-Marinari site-by-site optimisation.  At each site the
// locally optimal gauge transformation g(x) is applied to the eight
// links touching x:
//
//     U_mu(x)        ->  g(x) U_mu(x)              (4 outgoing)
//     U_mu(x - mu)   ->  U_mu(x - mu) g^dagger(x)  (4 incoming)
//
// F is monotonically non-decreasing and bounded above by 4 V, so the
// iteration converges.  Stop when |F_new - F_old| / F_new < tolerance,
// or after `max_iter` full sweeps over the lattice.
GaugeFixResult landau_gauge_fix(Lattice& lat,
                                double tolerance = 1e-9,
                                int    max_iter  = 5000);

// F[U] = sum_{x,mu} a0(U_mu(x)).
double landau_F(const Lattice& lat);

// Lattice divergence of the algebra-valued field A_mu(x) at every site,
// summed in quadrature over colour and divided by 4 V.  In exact Landau
// gauge this vanishes; we use it as a residual to monitor convergence.
//
//     theta = (1 / 4 V) sum_x sum_a [ sum_mu (A^a_mu(x) - A^a_mu(x - mu)) ]^2
double landau_theta(const Lattice& lat);

// Apply a site-local gauge transformation to every link:
//
//     U_mu(x) -> g(x) U_mu(x) g^dagger(x + mu)
//
// `g` must have lat.volume() entries indexed by the lattice's flat
// site index.  The transformation is gauge-invariant by construction.
void apply_gauge_transformation(Lattice& lat, const std::vector<SU2>& g);

}  // namespace su2lgt
