// Unit tests for Step 7: Polyakov loop, per-configuration measurement
// record, and the running aggregator that produces the Binder cumulant
// and susceptibility from a sequence of records.

#include "su2lgt/observables.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/rng.hpp"
#include "su2lgt/updater.hpp"
#include "su2lgt/su2.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

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

// ---- cold lattice: every link is identity, every Polyakov loop = 1 ----

void test_cold_polyakov() {
    Lattice lat(4, 4, 4, 3);  // intentionally Lt != Ls
    lat.cold_start();

    for (int z = 0; z < lat.size(2); ++z)
        for (int y = 0; y < lat.size(1); ++y)
            for (int x = 0; x < lat.size(0); ++x)
                check(std::abs(polyakov_local(lat, x, y, z) - 1.0) < TOL,
                      "cold Polyakov loop = 1 at every spatial site");

    check(std::abs(polyakov_global(lat) - 1.0) < TOL,
          "cold global Polyakov = 1");

    const ObsRecord r = measure(lat, /*sweep*/42);
    check(r.sweep == 42, "sweep index passed through");
    check(std::abs(r.L    - 1.0) < TOL, "cold L  = 1");
    check(std::abs(r.absL - 1.0) < TOL, "cold |L|= 1");
    check(std::abs(r.L2   - 1.0) < TOL, "cold L^2 = 1");
    check(std::abs(r.L4   - 1.0) < TOL, "cold L^4 = 1");
    check(std::abs(r.plaq - 1.0) < TOL, "cold plaq = 1");
}

// ---- centre transformation: flip every t-link in slice t=0 -----------

void test_centre_flips_polyakov(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    // Record local Polyakov loops at every site.
    std::vector<double> before(lat.size(0) * lat.size(1) * lat.size(2));
    int idx = 0;
    for (int z = 0; z < lat.size(2); ++z)
        for (int y = 0; y < lat.size(1); ++y)
            for (int x = 0; x < lat.size(0); ++x)
                before[idx++] = polyakov_local(lat, x, y, z);

    const double L_before = polyakov_global(lat);

    // Centre transformation: U_0(x, t=0) -> -U_0(x, t=0) for every spatial x.
    const int t_slice = 0;
    for (int z = 0; z < lat.size(2); ++z)
        for (int y = 0; y < lat.size(1); ++y)
            for (int x = 0; x < lat.size(0); ++x) {
                const int s = lat.site(x, y, z, t_slice);
                SU2& U = lat.U(s, 3);
                U = SU2{-U.a0, -U.a1, -U.a2, -U.a3};
            }

    // Each l(x) must have flipped sign; the global L too.
    idx = 0;
    for (int z = 0; z < lat.size(2); ++z)
        for (int y = 0; y < lat.size(1); ++y)
            for (int x = 0; x < lat.size(0); ++x) {
                const double after = polyakov_local(lat, x, y, z);
                check(std::abs(after + before[idx]) < 1e-12,
                      "centre flip negates local Polyakov loop at every site");
                ++idx;
            }

    const double L_after = polyakov_global(lat);
    check(std::abs(L_after + L_before) < 1e-12,
          "centre flip negates global Polyakov loop");
}

// ---- l(x) is bounded in [-1, 1] for any unitary configuration --------

void test_polyakov_bounded(RNG& rng) {
    Lattice lat(3, 3, 3, 4);
    lat.hot_start(rng);

    for (int z = 0; z < lat.size(2); ++z)
        for (int y = 0; y < lat.size(1); ++y)
            for (int x = 0; x < lat.size(0); ++x) {
                const double l = polyakov_local(lat, x, y, z);
                check(l >= -1.0 - 1e-12 && l <= 1.0 + 1e-12,
                      "local Polyakov loop in [-1, 1]");
            }

    const double L = polyakov_global(lat);
    check(L >= -1.0 - 1e-12 && L <= 1.0 + 1e-12,
          "global Polyakov in [-1, 1]");
}

// ---- accumulator arithmetic: cold record stream gives U4 = 2/3 -------

void test_accumulator_cold() {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    const int Vs = lat.size(0) * lat.size(1) * lat.size(2);

    ObsAccumulator acc(Vs);
    for (int s = 0; s < 50; ++s) acc.push(measure(lat, s));

    check(acc.n() == 50, "accumulator counts 50 records");
    check(std::abs(acc.mean_L()    - 1.0) < TOL, "cold mean L  = 1");
    check(std::abs(acc.mean_absL() - 1.0) < TOL, "cold mean |L|= 1");
    check(std::abs(acc.mean_L2()   - 1.0) < TOL, "cold mean L^2 = 1");
    check(std::abs(acc.mean_L4()   - 1.0) < TOL, "cold mean L^4 = 1");
    check(std::abs(acc.mean_plaq() - 1.0) < TOL, "cold mean plaq = 1");

    // Cold lattice has L = 1 at every measurement so:
    //   U_4 = 1 - <L^4> / (3 <L^2>^2) = 1 - 1/3 = 2/3.
    check(std::abs(acc.binder_U4() - 2.0/3.0) < TOL,
          "cold Binder cumulant = 2/3");

    // chi = Vs * (<L^2> - <|L|>^2) = Vs * (1 - 1) = 0 when no fluctuations.
    check(std::abs(acc.susceptibility()) < TOL,
          "cold susceptibility = 0 (no fluctuations)");
}

// ---- accumulator on a Gaussian-distributed L (synthetic) -------------
//
// Drive ObsRecords with L drawn from N(0, sigma^2) so we know the
// theoretical limits.  For a Gaussian:
//   <L^4> = 3 <L^2>^2  =>  Binder U_4 -> 0   (symmetric phase value).

void test_accumulator_gaussian() {
    ObsAccumulator acc(/*Vs=*/64);
    RNG rng(0xACC);
    constexpr int N = 200000;
    for (int i = 0; i < N; ++i) {
        ObsRecord r;
        const double L = 0.05 * rng.normal();   // sigma = 0.05
        r.L     = L;
        r.absL  = std::abs(L);
        r.L2    = L * L;
        r.L4    = r.L2 * r.L2;
        r.plaq  = 0.0;
        acc.push(r);
    }

    // With this many samples, U4 should be very close to 0.
    check(std::abs(acc.binder_U4()) < 0.02,
          "Gaussian-L synthetic stream gives Binder ~ 0");
    // <L> -> 0 and <|L|> -> sigma * sqrt(2/pi) ~ 0.0399 for sigma=0.05.
    check(std::abs(acc.mean_L()) < 0.005,
          "<L> -> 0 for symmetric Gaussian");
}

// ---- accumulator on a delta-function L (synthetic) -------------------
//
// All records have L = +0.4 exactly.  Then:
//   <L^4> = <L^2>^2   =>   Binder = 1 - 1/3 = 2/3   (broken-phase value).

void test_accumulator_delta() {
    ObsAccumulator acc(64);
    constexpr double L0 = 0.4;
    for (int i = 0; i < 1000; ++i) {
        ObsRecord r;
        r.L = L0; r.absL = std::abs(L0);
        r.L2 = L0 * L0; r.L4 = r.L2 * r.L2;
        acc.push(r);
    }
    check(std::abs(acc.binder_U4() - 2.0/3.0) < 1e-12,
          "delta-L stream gives Binder = 2/3");
    check(std::abs(acc.susceptibility()) < 1e-12,
          "delta-L stream gives chi = 0");
}

// ---- consistency with a real MC chain -------------------------------
//
// At very small beta the system is essentially uniform Haar.  The
// global Polyakov loop fluctuates symmetrically around 0, so <L> ~ 0
// and Binder is close to (but not exactly) 0 due to finite N.  This is
// a sanity check that the full pipeline measure -> push -> aggregate
// produces sensible numbers.

void test_short_mc_chain() {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();
    RNG rng(0xC0C0);
    MetropolisUpdater upd(lat, rng, /*beta*/0.5, /*eps*/0.4, /*n_hits*/3);

    // Quick thermalisation.
    for (int s = 0; s < 200; ++s) upd.sweep();

    const int Vs = lat.size(0) * lat.size(1) * lat.size(2);
    ObsAccumulator acc(Vs);
    for (int s = 0; s < 500; ++s) {
        upd.sweep();
        acc.push(measure(lat, s));
    }

    // <L> should be small (order 1/sqrt(N) of the typical |L| amplitude).
    check(std::abs(acc.mean_L()) < 0.1,
          "<L> small in symmetric phase");
    // Binder should be in the symmetric range [-0.5, 0.5] (well below 2/3).
    check(acc.binder_U4() < 0.55,
          "Binder below broken-phase limit at small beta");
}

}  // namespace

int main() {
    RNG rng(0xBEEFCAFE);

    test_cold_polyakov();              std::printf("[ok] cold Polyakov loop\n");
    test_centre_flips_polyakov(rng);   std::printf("[ok] centre transformation flips L\n");
    test_polyakov_bounded(rng);        std::printf("[ok] |Polyakov| <= 1\n");
    test_accumulator_cold();           std::printf("[ok] accumulator on cold stream\n");
    test_accumulator_gaussian();       std::printf("[ok] accumulator on Gaussian stream\n");
    test_accumulator_delta();          std::printf("[ok] accumulator on delta stream\n");
    test_short_mc_chain();             std::printf("[ok] accumulator on real MC chain\n");

    std::printf("\nAll Step 7 tests passed (%d assertions).\n", g_check_count);
    return 0;
}
