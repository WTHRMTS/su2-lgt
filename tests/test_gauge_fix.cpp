// Step 11 unit tests: Landau gauge fixing.
//
// Verifies:
//   1. cold lattice is already at F = 4 V; gauge-fix leaves it unchanged.
//   2. apply_gauge_transformation preserves all gauge-invariants
//      (action, plaquette, Polyakov loop).
//   3. starting from a gauge-equivalent perturbation of cold,
//      gauge-fix recovers F ~ 4 V (modulo Gribov copies).
//   4. monotonicity:  F never decreases during the iteration.
//   5. idempotency:   re-fixing an already-fixed configuration is a no-op.
//   6. theta decreases substantially during fixing.

#include "su2lgt/gauge_fix.hpp"
#include "su2lgt/action.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/observables.hpp"
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

// ---- 1. cold lattice is already maximised ------------------------------

void test_cold_at_max() {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    const double F0 = landau_F(lat);
    check(std::abs(F0 - 4.0 * lat.volume()) < TOL,
          "cold F = 4 V");

    const auto res = landau_gauge_fix(lat, 1e-12, 10);
    check(std::abs(res.F_final - 4.0 * lat.volume()) < 1e-9,
          "cold lattice stays at F = 4 V after fix");
    check(res.iterations <= 2,
          "cold lattice converges in <= 2 sweeps");
}

// ---- 2. apply_gauge_transformation preserves gauge invariants ---------

void test_gauge_transformation_invariance(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const double S_before = plaquette_action(lat, 1.7);
    const double L_before = polyakov_global(lat);

    std::vector<SU2> g(lat.volume());
    for (auto& gx : g) gx = rng.haar_su2();
    apply_gauge_transformation(lat, g);

    const double S_after = plaquette_action(lat, 1.7);
    const double L_after = polyakov_global(lat);

    check(std::abs(S_before - S_after) < 1e-9,
          "action is invariant under apply_gauge_transformation");
    check(std::abs(L_before - L_after) < 1e-9,
          "Polyakov loop is invariant under apply_gauge_transformation");
}

// ---- 3. gauge-equivalent perturbation of cold -> recovers F = 4 V ----

void test_gauge_equivalent_cold(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();

    // Random gauge transformation drives F < 4 V.
    std::vector<SU2> g(lat.volume());
    for (auto& gx : g) gx = rng.haar_su2();
    apply_gauge_transformation(lat, g);
    const double F_perturbed = landau_F(lat);
    check(F_perturbed < 4.0 * lat.volume() - 0.1,
          "random gauge transformation lowers F below max");

    const auto res = landau_gauge_fix(lat, 1e-10, 5000);
    // Recovery isn't exact -- there is a Gribov-copy ambiguity -- but
    // for a small lattice it should be within ~0.1% of the maximum.
    check(res.F_final > 4.0 * lat.volume() - 1.0,
          "gauge-fixing recovers F close to 4 V from a gauged-cold start");
    check(res.iterations <= 5000,
          "gauge fixing terminates within max_iter");
}

// ---- 4. monotonicity:  F never decreases per sweep --------------------

void test_monotonicity(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    double F_prev = landau_F(lat);
    for (int sweep = 0; sweep < 20; ++sweep) {
        const auto res = landau_gauge_fix(lat, /*tol*/0.0, /*max_iter*/1);
        const double F_now = res.F_final;
        check(F_now >= F_prev - 1e-12,
              "F is monotonically non-decreasing per sweep");
        F_prev = F_now;
    }
}

// ---- 5. idempotency:  re-fixing changes nothing further ---------------

void test_idempotency(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    landau_gauge_fix(lat, 1e-10, 5000);
    const double F1     = landau_F(lat);
    const double theta1 = landau_theta(lat);

    const auto res = landau_gauge_fix(lat, 1e-10, 5000);
    check(std::abs(res.F_final - F1) < 1e-7,
          "re-fixing does not move F beyond noise");
    check(std::abs(res.theta_final - theta1) < 1e-7,
          "re-fixing does not move theta beyond noise");
}

// ---- 6. theta decreases significantly ---------------------------------

void test_theta_decreases(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const auto res = landau_gauge_fix(lat, 1e-10, 5000);
    check(res.theta_final < res.theta_initial,
          "theta decreases through gauge fixing");
    // For random hot start, theta_initial is O(1).  After a non-trivial
    // gauge fix it should drop by at least a factor of 100.
    check(res.theta_final * 100.0 < res.theta_initial,
          "theta drops by factor >= 100");
    check(res.F_final > res.F_initial,
          "F increases through gauge fixing");
}

// ---- 7. plaquette action is gauge-invariant under fixing --------------
//
// The whole point of gauge fixing is that it shouldn't change the
// physics.  Verify by measuring the plaquette action before and after.

void test_action_invariant_under_fix(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const double S_before = plaquette_action(lat, 2.3);
    landau_gauge_fix(lat, 1e-10, 5000);
    const double S_after  = plaquette_action(lat, 2.3);

    check(std::abs(S_before - S_after) < 1e-7,
          "plaquette action invariant under gauge fixing");
}

}  // namespace

int main() {
    RNG rng(0xF1E2D3);

    test_cold_at_max();                      std::printf("[ok] cold at F = 4V\n");
    test_gauge_transformation_invariance(rng); std::printf("[ok] gauge transformation preserves S, L\n");
    test_gauge_equivalent_cold(rng);         std::printf("[ok] gauged-cold recovers F = 4V\n");
    test_monotonicity(rng);                  std::printf("[ok] F monotone in iteration\n");
    test_idempotency(rng);                   std::printf("[ok] re-fixing is a no-op\n");
    test_theta_decreases(rng);               std::printf("[ok] theta drops through fixing\n");
    test_action_invariant_under_fix(rng);    std::printf("[ok] action invariant under fixing\n");

    std::printf("\nAll Step 11 tests passed (%d assertions).\n", g_check_count);
    return 0;
}
