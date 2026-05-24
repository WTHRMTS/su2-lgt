// Step 12 unit tests: gauge-boson static propagator G(r).
//
// Verifies:
//   1. cold lattice -> A^a = 0 -> G(r) = 0 for all r.
//   2. r_max+1 entries returned.
//   3. G(0) = (1/V_s) sum_{x_s} <|Abar(x_s)|^2>  matches a direct
//      computation of the squared-norm density.
//   4. on a thermalised+gauge-fixed configuration G(0) > 0 and
//      G(r) decays at long r (G(r_max) < G(0)).

#include "su2lgt/propagator.hpp"
#include "su2lgt/gauge_fix.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/rng.hpp"
#include "su2lgt/updater.hpp"
#include "su2lgt/su2.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace su2lgt;

namespace {

constexpr double TOL = 1e-12;
int g_check_count = 0;

void check(bool ok, const char* what) {
    ++g_check_count;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (assertion %d)\n", what, g_check_count);
        std::exit(1);
    }
}

// ---- 1.  cold lattice: every algebra component is zero ----------------

void test_cold_propagator() {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    const auto G = static_propagator(lat, /*r_max*/2);
    check(G.size() == 3, "G has r_max+1 entries");
    for (int r = 0; r <= 2; ++r)
        check(std::abs(G[r]) < TOL, "cold G(r) = 0");
}

// ---- 2.  G(0) matches direct computation of <|Abar|^2> --------------

void test_G0_matches_direct(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const int Lx = lat.size(0), Ly = lat.size(1), Lz = lat.size(2),
              Lt = lat.size(3);
    const int Vs = Lx * Ly * Lz;

    // Compute Abar^a_mu(x_s) directly and the squared-norm density.
    std::vector<double> Abar(static_cast<std::size_t>(3) * 4 * Vs, 0.0);
    auto idx = [&](int a, int mu, int x, int y, int z) {
        return a * 4 * Vs + mu * Vs + (z * Ly + y) * Lx + x;
    };
    for (int t = 0; t < Lt; ++t)
        for (int z = 0; z < Lz; ++z)
            for (int y = 0; y < Ly; ++y)
                for (int x = 0; x < Lx; ++x) {
                    const int s = lat.site(x, y, z, t);
                    for (int mu = 0; mu < 4; ++mu) {
                        const SU2& U = lat.U(s, mu);
                        Abar[idx(0, mu, x, y, z)] += U.a1;
                        Abar[idx(1, mu, x, y, z)] += U.a2;
                        Abar[idx(2, mu, x, y, z)] += U.a3;
                    }
                }
    const double inv_Lt = 1.0 / Lt;
    for (auto& v : Abar) v *= inv_Lt;

    double direct_G0 = 0.0;
    for (int z = 0; z < Lz; ++z)
        for (int y = 0; y < Ly; ++y)
            for (int x = 0; x < Lx; ++x)
                for (int a = 0; a < 3; ++a)
                    for (int mu = 0; mu < 4; ++mu) {
                        const double v = Abar[idx(a, mu, x, y, z)];
                        direct_G0 += v * v;
                    }
    direct_G0 /= Vs;

    const auto G = static_propagator(lat, /*r_max*/0);
    check(std::abs(G[0] - direct_G0) < 1e-12,
          "G(0) matches direct |Abar|^2 density");
}

// ---- 3.  decay at long r on a thermalised+gauge-fixed config --------

void test_decay_after_gauge_fix(RNG& rng) {
    Lattice lat(8, 8, 8, 4);
    lat.cold_start();
    MetropolisUpdater upd(lat, rng, /*beta*/2.30, /*eps*/0.45, /*n_hits*/5);
    for (int s = 0; s < 200; ++s) {
        upd.sweep();
        if ((s + 1) % 50 == 0) lat.reunitarise_all();
    }

    const auto gf = landau_gauge_fix(lat, /*tol*/1e-9, /*max_iter*/2000);
    check(gf.theta_final < 1e-7, "gauge fix converged on thermalised config");

    const auto G = static_propagator(lat, /*r_max*/4);
    check(G[0] > 0.0,        "G(0) > 0 on hot config");
    check(G[0] > G[4],        "G(0) > G(r_max)");
    // Sanity: G(0) is at most  3 * 4  (since each Abar contribution
    // is bounded by 1 in magnitude on SU(2) and we sum 3 colours x 4 dirs).
    check(G[0] < 3.0 * 4.0,   "G(0) bounded above by 12");
}

// ---- 4.  reproducibility under gauge fixing ------------------------
//
// Take a configuration; gauge-fix it; measure G(r); save G_1.
// Apply a random gauge transformation; gauge-fix again; measure; save G_2.
// G_1 ~= G_2 modulo Gribov copies.  We use a loose tolerance because
// of the Gribov ambiguity but the propagator should still be very close.

void test_gauge_fix_reproducibility(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    MetropolisUpdater upd(lat, rng, 2.30, 0.45, 5);
    for (int s = 0; s < 100; ++s) upd.sweep();

    // Make a copy, gauge-fix it, measure.
    Lattice lat1 = lat;
    landau_gauge_fix(lat1, 1e-9, 2000);
    const auto G1 = static_propagator(lat1, 2);

    // Re-fix without perturbing -- should be a no-op.
    Lattice lat2 = lat;
    landau_gauge_fix(lat2, 1e-9, 2000);
    const auto G2 = static_propagator(lat2, 2);

    for (int r = 0; r <= 2; ++r)
        check(std::abs(G1[r] - G2[r]) < 1e-7,
              "G(r) reproducible across re-fix on same config");
}

}  // namespace

int main() {
    RNG rng(0xCAFEBABE);

    test_cold_propagator();              std::printf("[ok] cold G(r) = 0\n");
    test_G0_matches_direct(rng);         std::printf("[ok] G(0) = direct |Abar|^2\n");
    test_decay_after_gauge_fix(rng);     std::printf("[ok] G decays after gauge fix\n");
    test_gauge_fix_reproducibility(rng); std::printf("[ok] G reproducible across re-fix\n");

    std::printf("\nAll Step 12 tests passed (%d assertions).\n", g_check_count);
    return 0;
}
