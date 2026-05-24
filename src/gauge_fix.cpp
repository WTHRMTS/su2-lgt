#include "su2lgt/gauge_fix.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace su2lgt {

double landau_F(const Lattice& lat) {
    double F = 0.0;
    const int V = lat.volume();
    for (int s = 0; s < V; ++s)
        for (int mu = 0; mu < 4; ++mu)
            F += lat.U(s, mu).a0;
    return F;
}

double landau_theta(const Lattice& lat) {
    // Residual of the Landau gauge condition:
    //
    //   For each site x and colour a in {1, 2, 3}:
    //     d^a(x) = sum_mu [ A^a_mu(x) - A^a_mu(x - mu) ]
    //   theta = (1 / (4 V)) sum_x [ d^1(x)^2 + d^2(x)^2 + d^3(x)^2 ]
    //
    // We approximate A^a_mu(x) = a^a(U_mu(x)).  This linearisation is
    // exact in the continuum limit and a good monitor here.
    double theta = 0.0;
    const int V = lat.volume();
    for (int s = 0; s < V; ++s) {
        double d1 = 0.0, d2 = 0.0, d3 = 0.0;
        for (int mu = 0; mu < 4; ++mu) {
            const SU2& U_fwd = lat.U(s, mu);
            const SU2& U_bwd = lat.U(lat.shift(s, mu, -1), mu);
            d1 += U_fwd.a1 - U_bwd.a1;
            d2 += U_fwd.a2 - U_bwd.a2;
            d3 += U_fwd.a3 - U_bwd.a3;
        }
        theta += d1 * d1 + d2 * d2 + d3 * d3;
    }
    return theta / (4.0 * V);
}

namespace {

// K(x) = sum_mu [ U_mu(x) + U_mu^dag(x - mu) ]
//
// Treated as a 4-tuple (sum of SU(2) elements; not unit norm).
SU2 compute_K(const Lattice& lat, int s) {
    SU2 K{0.0, 0.0, 0.0, 0.0};
    for (int mu = 0; mu < 4; ++mu) {
        K = K + lat.U(s, mu);
        const int s_minus = lat.shift(s, mu, -1);
        K = K + lat.U(s_minus, mu).dagger();
    }
    return K;
}

void apply_g_at_site(Lattice& lat, int s, const SU2& g) {
    const SU2 g_dag = g.dagger();
    for (int mu = 0; mu < 4; ++mu) {
        lat.U(s, mu) = g * lat.U(s, mu);
        const int s_minus = lat.shift(s, mu, -1);
        lat.U(s_minus, mu) = lat.U(s_minus, mu) * g_dag;
    }
}

}  // namespace

GaugeFixResult landau_gauge_fix(Lattice& lat, double tolerance, int max_iter) {
    GaugeFixResult res;
    res.F_max         = 4.0 * lat.volume();
    res.F_initial     = landau_F(lat);
    res.theta_initial = landau_theta(lat);

    double F_prev = res.F_initial;
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        for (int s = 0; s < lat.volume(); ++s) {
            const SU2 K = compute_K(lat, s);
            const double n = std::sqrt(K.norm_sq());
            if (n < 1e-15) continue;        // degenerate, skip
            // Locally optimal g aligns Re Tr(g K) = max:
            //     g_opt = K^dagger / |K|.
            const SU2 g_opt{ K.a0 / n,
                            -K.a1 / n,
                            -K.a2 / n,
                            -K.a3 / n };
            apply_g_at_site(lat, s, g_opt);
        }

        const double F_now = landau_F(lat);
        const double dF    = F_now - F_prev;

        // Convergence: relative change below tolerance.
        if (std::abs(dF) < tolerance * std::max(1.0, std::abs(F_now))) {
            res.iterations  = iter + 1;
            res.F_final     = F_now;
            res.theta_final = landau_theta(lat);
            res.converged   = true;
            return res;
        }
        F_prev = F_now;
    }

    res.iterations  = max_iter;
    res.F_final     = F_prev;
    res.theta_final = landau_theta(lat);
    res.converged   = false;
    return res;
}

void apply_gauge_transformation(Lattice& lat, const std::vector<SU2>& g) {
    // Read all old links into a buffer first, then write back, so that
    // the in-place update of one link doesn't bias the computation of
    // its neighbours' transformations.
    const int V = lat.volume();
    std::vector<SU2> new_links(static_cast<std::size_t>(4) * V);
    for (int s = 0; s < V; ++s) {
        for (int mu = 0; mu < 4; ++mu) {
            const int s_mu = lat.shift(s, mu, +1);
            new_links[4 * s + mu] = g[s] * lat.U(s, mu) * g[s_mu].dagger();
        }
    }
    for (int s = 0; s < V; ++s)
        for (int mu = 0; mu < 4; ++mu)
            lat.U(s, mu) = new_links[4 * s + mu];
}

}  // namespace su2lgt
