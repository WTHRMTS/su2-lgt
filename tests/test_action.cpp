// Unit tests for Step 5: plaquette, total action, staple sum, local
// energy difference.  No physics yet -- this is a numerical-correctness
// test of the action machinery the Step-6 updater will lean on.

#include "su2lgt/action.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/rng.hpp"
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

// ---- cold lattice: every plaquette is the identity ----------------------

void test_cold_lattice() {
    Lattice lat(4, 4, 4, 4);
    lat.cold_start();

    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu)
            for (int nu = mu + 1; nu < 4; ++nu) {
                const SU2 P = plaquette(lat, s, mu, nu);
                check(std::abs(P.half_trace() - 1.0) < TOL,
                      "cold plaquette has half_trace = 1");
                check(std::abs(P.norm_sq() - 1.0) < 1e-12,
                      "cold plaquette is unitary");
            }

    const int Np = num_plaquettes(lat);
    check(std::abs(total_plaquette_trace(lat) - Np) < 1e-9,
          "cold total plaquette trace = Np");
    check(std::abs(plaquette_action(lat, 2.30)) < 1e-9,
          "cold action = 0");
    check(std::abs(mean_plaquette(lat) - 1.0) < 1e-12,
          "cold mean plaquette = 1");
}

// ---- hot lattice: mean plaquette near zero ------------------------------

void test_hot_lattice(RNG& rng) {
    Lattice lat(6, 6, 6, 6);
    lat.hot_start(rng);
    const double mp = mean_plaquette(lat);
    check(std::abs(mp) < 0.05,
          "hot start mean plaquette ~ 0 (within 0.05)");
    // Plaquettes should still all be unitary -- they are products of SU(2)s.
    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu)
            for (int nu = mu + 1; nu < 4; ++nu)
                check(std::abs(plaquette(lat, s, mu, nu).norm_sq() - 1.0)
                          < 1e-10,
                      "plaquette of unitary links is unitary");
}

// ---- Z_2 centre symmetry of the action ---------------------------------

void test_centre_symmetry(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const double S0 = plaquette_action(lat, 1.0);

    // Flip every t-direction link in a single time slice.  Each spatial
    // plaquette is unaffected; each (mu, t) plaquette either has none or
    // both of its t-direction links flipped -> sign cancels.
    const int t_slice = 1;
    for (int z = 0; z < lat.size(2); ++z)
    for (int y = 0; y < lat.size(1); ++y)
    for (int x = 0; x < lat.size(0); ++x) {
        const int s = lat.site(x, y, z, t_slice);
        SU2& U = lat.U(s, 3);
        U = SU2{-U.a0, -U.a1, -U.a2, -U.a3};
    }

    const double S1 = plaquette_action(lat, 1.0);
    check(std::abs(S0 - S1) < 1e-9,
          "centre flip preserves action");
}

// ---- staple matches direct plaquette sum at one link --------------------

void test_staple_per_link(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const int site = 13;
    const int mu   = 1;
    const SU2 U = lat.U(site, mu);
    const SU2 A = staple_sum(lat, site, mu);

    // Direct sum: this link participates in 6 plaquettes.  For each
    // nu != mu, there is the forward plaquette starting at x and the
    // backward plaquette starting at x - nu.
    double direct = 0.0;
    for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) continue;
        direct += plaquette(lat, site, mu, nu).half_trace();
        direct += plaquette(lat, lat.shift(site, nu, -1), mu, nu).half_trace();
    }

    check(std::abs((U * A).a0 - direct) < 1e-12,
          "(U_link * staple).a0 = sum of 6 plaquette half-traces");
}

// ---- global staple/plaquette accounting ---------------------------------

void test_global_staple_accounting(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const double sum_plaq = total_plaquette_trace(lat);

    double sum_link = 0.0;
    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu) {
            const SU2 A = staple_sum(lat, s, mu);
            sum_link += (lat.U(s, mu) * A).a0;
        }

    // Each plaquette is in the staple of 4 of its 4 links.
    check(std::abs(sum_link - 4.0 * sum_plaq) < 1e-9,
          "Sum_links (U.staple).a0 = 4 * Sum_plaq (Up).a0");
}

// ---- Delta S consistency: full recompute vs staple formula --------------

void test_delta_action_consistency(RNG& rng) {
    Lattice lat(5, 5, 5, 5);
    lat.hot_start(rng);

    const double beta = 2.20;

    // Try at a few different links; modifications accumulate but each
    // step recomputes its own S_before.
    const int test_sites[]      = { 17,   73,   4,  101 };
    const int test_directions[] = {  2,    0,   3,    1 };

    for (int k = 0; k < 4; ++k) {
        const int site = test_sites[k];
        const int mu   = test_directions[k];

        const SU2  U_old   = lat.U(site, mu);
        const SU2  staple  = staple_sum(lat, site, mu);
        const double S_before = plaquette_action(lat, beta);

        const SU2  U_new   = rng.haar_su2();
        lat.U(site, mu) = U_new;

        const double S_after  = plaquette_action(lat, beta);
        const double dS_full  = S_after - S_before;
        const double dS_local = delta_action_from_staple(
                                    beta, U_old, U_new, staple);

        check(std::abs(dS_full - dS_local) < 1e-9,
              "Delta S full == Delta S from staple");
    }
}

// ---- Delta S = 0 for trivial change ------------------------------------

void test_delta_action_trivial(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const int site = 5, mu = 1;
    const SU2 U = lat.U(site, mu);
    const SU2 A = staple_sum(lat, site, mu);
    const double dS = delta_action_from_staple(2.30, U, U, A);
    check(std::abs(dS) < 1e-15, "Delta S = 0 when U' = U");
}

// ---- Action is gauge-invariant ----------------------------------------
//
// Apply a random gauge transformation g(x) at every site and verify
// that the Wilson action is unchanged.  Under a gauge transformation,
//     U_mu(x) -> g(x) U_mu(x) g^dag(x + mu),
// every plaquette becomes  g(x) U_p(x) g^dag(x), so its trace is
// invariant.  This is the deepest single test of the action.

void test_gauge_invariance(RNG& rng) {
    Lattice lat(4, 4, 4, 4);
    lat.hot_start(rng);

    const double beta = 1.7;
    const double S_before = plaquette_action(lat, beta);

    // Random g(x) at every site.
    std::vector<SU2> g(lat.volume());
    for (auto& gx : g) gx = rng.haar_su2();

    Lattice lat2(4, 4, 4, 4);
    for (int s = 0; s < lat.volume(); ++s) {
        for (int mu = 0; mu < 4; ++mu) {
            const int s_mu = lat.shift(s, mu, +1);
            lat2.U(s, mu) = g[s] * lat.U(s, mu) * g[s_mu].dagger();
        }
    }

    const double S_after = plaquette_action(lat2, beta);
    check(std::abs(S_before - S_after) < 1e-9,
          "action is gauge-invariant under random g(x)");
}

}  // namespace

int main() {
    RNG rng(0xBADCAFE);

    test_cold_lattice();             std::printf("[ok] cold lattice action\n");
    test_hot_lattice(rng);           std::printf("[ok] hot lattice mean plaquette\n");
    test_centre_symmetry(rng);       std::printf("[ok] centre symmetry of action\n");
    test_staple_per_link(rng);       std::printf("[ok] staple matches 6-plaquette sum\n");
    test_global_staple_accounting(rng);
                                     std::printf("[ok] global staple accounting (factor of 4)\n");
    test_delta_action_consistency(rng);
                                     std::printf("[ok] Delta S full vs staple\n");
    test_delta_action_trivial(rng);  std::printf("[ok] Delta S = 0 for trivial change\n");
    test_gauge_invariance(rng);      std::printf("[ok] action is gauge invariant\n");

    std::printf("\nAll Step 5 tests passed (%d assertions).\n", g_check_count);
    return 0;
}
