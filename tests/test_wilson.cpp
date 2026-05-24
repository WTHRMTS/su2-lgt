// Step 10 unit tests: Wilson loop, mean Wilson loop, Polyakov pair
// correlator.

#include "su2lgt/wilson.hpp"
#include "su2lgt/observables.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/rng.hpp"
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

// ---- cold lattice: every Wilson loop is identity, half-trace = 1 ------

void test_cold_wilson() {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    for (int R = 1; R <= 2; ++R)
        for (int T = 1; T <= 2; ++T)
            for (int mu = 0; mu < 3; ++mu)
                check(std::abs(wilson_loop(lat, 0, mu, 3, R, T) - 1.0) < TOL,
                      "cold Wilson loop = 1");

    for (int R = 1; R <= 2; ++R)
        for (int T = 1; T <= 2; ++T)
            check(std::abs(mean_spatial_temporal_wilson_loop(lat, R, T) - 1.0)
                      < TOL,
                  "cold mean Wilson loop = 1");
}

// ---- gauge invariance: random g(x) leaves W(R, T) unchanged ----------

void test_wilson_gauge_invariance(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    // Save a baseline measurement
    const double W_before = mean_spatial_temporal_wilson_loop(lat, 2, 2);

    // Apply random g(x) on every site:  U_mu(x) -> g(x) U_mu(x) g^dag(x+mu)
    Lattice lat2(4, 4, 4, 4);
    std::vector<SU2> g(lat.volume());
    for (auto& gx : g) gx = rng.haar_su2();
    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu) {
            const int s_mu = lat.shift(s, mu, +1);
            lat2.U(s, mu) = g[s] * lat.U(s, mu) * g[s_mu].dagger();
        }

    const double W_after = mean_spatial_temporal_wilson_loop(lat2, 2, 2);
    check(std::abs(W_before - W_after) < 1e-10,
          "Wilson loop is gauge-invariant under random g(x)");
}

// ---- Wilson loop bounded in [-1, 1] for any unitary configuration ----

void test_wilson_bounded(RNG& rng) {
    Lattice lat(3, 3, 3, 4);
    lat.hot_start(rng);
    for (int R = 1; R <= 2; ++R)
        for (int T = 1; T <= 2; ++T)
            for (int mu = 0; mu < 3; ++mu) {
                const double w = wilson_loop(lat, 0, mu, 3, R, T);
                check(w >= -1.0 - 1e-12 && w <= 1.0 + 1e-12,
                      "Wilson loop in [-1, 1]");
            }
}

// ---- Polyakov pair correlator: cold lattice gives all 1s -------------

void test_polyakov_pair_cold() {
    Lattice lat(6, 6, 6, 4);
    lat.cold_start();
    const auto C = polyakov_pair_correlator(lat, /*r_max*/3);
    check(C.size() == 4, "pair correlator has r_max+1 entries");
    for (int r = 0; r <= 3; ++r)
        check(std::abs(C[r] - 1.0) < TOL,
              "cold pair correlator = 1 at all r");
}

// ---- Polyakov pair correlator: r=0 reproduces <L^2> in another way ---

void test_polyakov_pair_r0(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);
    const auto C = polyakov_pair_correlator(lat, /*r_max*/2);

    // r=0: average of l(x)^2 over spatial sites.
    const auto f = polyakov_local_field(lat);
    double sumsq = 0.0;
    for (double v : f) sumsq += v * v;
    const double mean_l2 = sumsq / f.size();

    check(std::abs(C[0] - mean_l2) < 1e-12,
          "P(0) = <l(x)^2> directly");
}

// ---- Centre transformation flips l(x), so l(x)*l(y) is INVARIANT -----

void test_pair_invariance_under_centre(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const auto C_before = polyakov_pair_correlator(lat, 2);

    // Centre transformation: U_t(x, t=0) -> -U_t(x, t=0) for every spatial x.
    const int t_slice = 0;
    for (int z = 0; z < lat.size(2); ++z)
        for (int y = 0; y < lat.size(1); ++y)
            for (int x = 0; x < lat.size(0); ++x) {
                const int s = lat.site(x, y, z, t_slice);
                SU2& U = lat.U(s, 3);
                U = SU2{-U.a0, -U.a1, -U.a2, -U.a3};
            }

    const auto C_after = polyakov_pair_correlator(lat, 2);
    for (int r = 0; r <= 2; ++r)
        check(std::abs(C_before[r] - C_after[r]) < 1e-12,
              "pair correlator invariant under centre transformation");
}

}  // namespace

int main() {
    RNG rng(0xBADBABE);

    test_cold_wilson();              std::printf("[ok] cold Wilson loops = 1\n");
    test_wilson_gauge_invariance(rng);
                                     std::printf("[ok] Wilson loop gauge-invariant\n");
    test_wilson_bounded(rng);        std::printf("[ok] |Wilson loop| <= 1\n");
    test_polyakov_pair_cold();       std::printf("[ok] cold pair correlator = 1\n");
    test_polyakov_pair_r0(rng);      std::printf("[ok] P(0) = <l^2>\n");
    test_pair_invariance_under_centre(rng);
                                     std::printf("[ok] pair correlator centre-invariant\n");

    std::printf("\nAll Step 10 tests passed (%d assertions).\n", g_check_count);
    return 0;
}
