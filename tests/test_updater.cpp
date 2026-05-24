// Unit tests for Step 6: the Metropolis updater.
//
// These tests run actual short Markov chains, which makes them slower
// than the algebra tests (a few seconds total).  They cover:
//   - sweep arithmetic (proposed = 4 * V * n_hits)
//   - acceptance monotonicity in the proposal width eps
//   - link unitarity preservation through hundreds of sweeps
//   - strong-coupling limit (beta=0): mean plaquette -> 0
//   - weak-coupling limit (large beta): mean plaquette -> 1
//   - ergodicity: hot- and cold-start chains converge to the same value

#include "su2lgt/updater.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/rng.hpp"
#include "su2lgt/action.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace su2lgt;

namespace {

int g_check_count = 0;

void check(bool ok, const char* what) {
    ++g_check_count;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (assertion %d)\n", what, g_check_count);
        std::exit(1);
    }
}

// Helper: thermalise on a 4^4 lattice and return mean plaquette
// averaged over the measurement phase.
double run_chain(bool hot_start,
                 double beta,
                 double eps,
                 std::uint64_t seed,
                 int n_therm,
                 int n_meas) {
    Lattice lat(4, 4, 4, 4);
    RNG rng(seed);
    if (hot_start) lat.hot_start(rng);
    else           lat.cold_start();

    MetropolisUpdater upd(lat, rng, beta, eps, /*n_hits*/5);

    for (int s = 0; s < n_therm; ++s) {
        upd.sweep();
        if ((s + 1) % 100 == 0) lat.reunitarise_all();
    }

    double sum = 0.0;
    int    n   = 0;
    for (int s = 0; s < n_meas; ++s) {
        upd.sweep();
        if ((s + 1) % 100 == 0) lat.reunitarise_all();
        sum += mean_plaquette(lat);
        ++n;
    }
    return sum / n;
}

void test_sweep_counts() {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    RNG rng(0xCAFEBABE);
    const int n_hits = 3;
    MetropolisUpdater upd(lat, rng, /*beta*/2.30, /*eps*/0.3, n_hits);

    upd.sweep();
    const long long expected = 4LL * lat.volume() * n_hits;
    check(upd.stats().proposed == expected,
          "one sweep -> 4 * V * n_hits proposals");
    check(upd.stats().accepted >= 0
          && upd.stats().accepted <= upd.stats().proposed,
          "accepted in [0, proposed]");

    upd.sweep();
    check(upd.stats().proposed == 2 * expected,
          "two sweeps -> 2 * 4 * V * n_hits proposals");
}

void test_eps_monotonic() {
    auto run_acc = [](double eps, std::uint64_t seed) {
        Lattice lat(4, 4, 4, 4);
        lat.cold_start();
        RNG rng(seed);
        MetropolisUpdater upd(lat, rng, 2.30, eps, 5);
        for (int s = 0; s < 100; ++s) upd.sweep();
        return upd.stats().acceptance_rate();
    };
    const double r_small = run_acc(0.05, 0xFEED);
    const double r_large = run_acc(0.50, 0xFEED);
    check(r_small > r_large + 0.1,
          "smaller proposal width -> higher acceptance");
}

void test_unitarity_preserved_during_run() {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    RNG rng(0x123);
    MetropolisUpdater upd(lat, rng, 2.30, 0.3, 5);

    // 200 sweeps with no reunitarisation.  Drift should be negligible.
    for (int s = 0; s < 200; ++s) upd.sweep();

    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu) {
            const double n = lat.U(s, mu).norm_sq();
            check(std::abs(n - 1.0) < 1e-8,
                  "link stays unitary across 200 sweeps without reunitarisation");
        }
}

void test_strong_coupling_limit() {
    // beta = 0: action is constant, every proposal accepts (apart from
    // measure-zero exact-equality cases), and the equilibrium measure
    // is uniform Haar.  Expectation: mean plaquette -> 0.
    const double P = run_chain(/*hot*/false, /*beta*/0.0, /*eps*/0.5,
                               0x55AA55, /*n_therm*/300, /*n_meas*/100);
    check(std::abs(P) < 0.05, "beta=0 mean plaquette ~ 0");
}

void test_weak_coupling_limit() {
    // Large beta: SU(2) leading-order weak-coupling expansion gives
    //   <P> ~ 1 - 3/(4 beta).
    // At beta = 8 this is ~0.906.  We allow [0.85, 0.95] to absorb
    // finite-volume + finite-stats noise on a 4^4 box.
    const double P = run_chain(/*hot*/false, /*beta*/8.0, /*eps*/0.15,
                               0xBEEF, /*n_therm*/300, /*n_meas*/200);
    check(P > 0.85 && P < 0.95,
          "beta=8 mean plaquette in [0.85, 0.95]");
}

void test_hot_cold_agree() {
    // Run hot- and cold-start chains at moderate (sub-critical) beta;
    // their thermalised mean plaquettes should agree to within the
    // statistical error.
    const double P_hot  = run_chain(/*hot*/true,  /*beta*/1.8, /*eps*/0.30,
                                     0x1111, /*n_therm*/500, /*n_meas*/300);
    const double P_cold = run_chain(/*hot*/false, /*beta*/1.8, /*eps*/0.30,
                                     0x2222, /*n_therm*/500, /*n_meas*/300);
    check(std::abs(P_hot - P_cold) < 0.03,
          "hot- and cold-start chains converge to same mean plaquette");
}

void test_zero_eps_no_motion() {
    // eps -> 0 makes proposals essentially the identity; we expect
    // very high acceptance and almost no change in mean plaquette.
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    RNG rng(0xEEEE);
    MetropolisUpdater upd(lat, rng, /*beta*/2.30, /*eps*/1e-6, /*n_hits*/5);

    for (int s = 0; s < 50; ++s) upd.sweep();
    check(upd.stats().acceptance_rate() > 0.99,
          "tiny eps -> >99% acceptance");
    check(std::abs(mean_plaquette(lat) - 1.0) < 1e-6,
          "tiny eps from cold -> stays near identity (mean P ~ 1)");
}

}  // namespace

int main() {
    test_sweep_counts();
    std::printf("[ok] sweep counts\n");

    test_eps_monotonic();
    std::printf("[ok] acceptance monotonic in eps\n");

    test_unitarity_preserved_during_run();
    std::printf("[ok] unitarity preserved across run\n");

    test_zero_eps_no_motion();
    std::printf("[ok] tiny eps -> near-identity, high acceptance\n");

    test_strong_coupling_limit();
    std::printf("[ok] beta=0 -> mean plaquette ~ 0\n");

    test_weak_coupling_limit();
    std::printf("[ok] beta=8 -> mean plaquette ~ 0.9\n");

    test_hot_cold_agree();
    std::printf("[ok] hot/cold ergodicity\n");

    std::printf("\nAll Step 6 tests passed (%d assertions).\n", g_check_count);
    return 0;
}
