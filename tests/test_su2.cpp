// Unit tests for the Step-4 scaffolding: SU(2) algebra, RNG, and the
// 4D lattice container with periodic boundary conditions.  No physics
// here -- the action and updater arrive in Steps 5 and 6.
//
// Run with `ctest` or directly: ./test_su2.

#include "su2lgt/su2.hpp"
#include "su2lgt/rng.hpp"
#include "su2lgt/lattice.hpp"

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

double dist(const SU2& U, const SU2& V) {
    const double d0 = U.a0 - V.a0;
    const double d1 = U.a1 - V.a1;
    const double d2 = U.a2 - V.a2;
    const double d3 = U.a3 - V.a3;
    return std::sqrt(d0*d0 + d1*d1 + d2*d2 + d3*d3);
}

bool approx_equal(const SU2& U, const SU2& V, double tol = TOL) {
    return dist(U, V) < tol;
}

// ---- algebra ------------------------------------------------------------

void test_identity() {
    const SU2 I = SU2::identity();
    check(I.a0 == 1.0 && I.a1 == 0.0 && I.a2 == 0.0 && I.a3 == 0.0,
          "identity stored as (1,0,0,0)");
    check(approx_equal(I * I, I), "I * I = I");
    check(std::abs(I.half_trace() - 1.0) < TOL, "half_trace(I) = 1");
}

void test_known_product() {
    // exp(i pi/2 sigma_x) * exp(i pi/2 sigma_y) = -i sigma_z
    // As quaternions:  (0,1,0,0) * (0,0,1,0) = (0,0,0,-1)
    const SU2 U{0.0, 1.0, 0.0, 0.0};
    const SU2 V{0.0, 0.0, 1.0, 0.0};
    const SU2 W = U * V;
    check(approx_equal(W, SU2{0.0, 0.0, 0.0, -1.0}, 1e-15),
          "(i sigma_x)(i sigma_y) = -i sigma_z");
}

void test_unitary(RNG& rng) {
    for (int i = 0; i < 1000; ++i) {
        const SU2 U = rng.haar_su2();
        check(std::abs(U.norm_sq() - 1.0) < 1e-12, "Haar element is unit");
        check(approx_equal(U * U.dagger(), SU2::identity(), 1e-12),
              "U U^dagger = I");
        check(approx_equal(U.dagger() * U, SU2::identity(), 1e-12),
              "U^dagger U = I");
    }
}

void test_associativity(RNG& rng) {
    for (int i = 0; i < 200; ++i) {
        const SU2 A = rng.haar_su2();
        const SU2 B = rng.haar_su2();
        const SU2 C = rng.haar_su2();
        check(approx_equal((A * B) * C, A * (B * C), 1e-11),
              "(AB)C = A(BC)");
    }
}

void test_trace_cyclic(RNG& rng) {
    for (int i = 0; i < 200; ++i) {
        const SU2 A = rng.haar_su2();
        const SU2 B = rng.haar_su2();
        const SU2 C = rng.haar_su2();
        const double tABC = (A * B * C).half_trace();
        const double tBCA = (B * C * A).half_trace();
        const double tCAB = (C * A * B).half_trace();
        check(std::abs(tABC - tBCA) < 1e-12, "Tr(ABC) = Tr(BCA)");
        check(std::abs(tABC - tCAB) < 1e-12, "Tr(ABC) = Tr(CAB)");
    }
}

void test_near_identity(RNG& rng) {
    const double eps = 0.05;
    for (int i = 0; i < 1000; ++i) {
        const SU2 X = rng.random_near_identity(eps);
        check(std::abs(X.norm_sq() - 1.0) < 1e-12,
              "near-identity proposal is unit");
        check(X.a0 >= std::sqrt(1.0 - eps*eps) - 1e-12,
              "a0 of near-identity proposal stays in expected band");
    }
}

void test_reunitarise(RNG& rng) {
    SU2 U = rng.haar_su2();
    // Multiply many times to accumulate roundoff drift, then re-normalise.
    for (int i = 0; i < 1000; ++i) U = U * rng.haar_su2();
    U.reunitarise();
    check(std::abs(U.norm_sq() - 1.0) < 1e-15,
          "reunitarise restores unit norm");
}

// ---- lattice ------------------------------------------------------------

void test_lattice_indexing() {
    Lattice lat(4, 5, 6, 3);   // distinct extents in every direction
    check(lat.volume() == 4 * 5 * 6 * 3, "volume = product of extents");
    check(lat.num_links() == 4 * lat.volume(), "num_links = 4 * volume");

    // Round-trip site index <-> coordinates.
    for (int s = 0; s < lat.volume(); ++s) {
        int x, y, z, t;
        lat.coords(s, x, y, z, t);
        check(lat.site(x, y, z, t) == s, "site/coords round-trip");
    }

    // Forward periodic shifts wrap correctly.
    int s = lat.site(3, 4, 5, 2);
    check(lat.shift(s, 0, +1) == lat.site(0, 4, 5, 2), "+x wraps to 0");
    check(lat.shift(s, 1, +1) == lat.site(3, 0, 5, 2), "+y wraps to 0");
    check(lat.shift(s, 2, +1) == lat.site(3, 4, 0, 2), "+z wraps to 0");
    check(lat.shift(s, 3, +1) == lat.site(3, 4, 5, 0), "+t wraps to 0");

    // Backward shifts.
    int s0 = lat.site(0, 0, 0, 0);
    check(lat.shift(s0, 0, -1) == lat.site(3, 0, 0, 0), "-x wraps to Lx-1");
    check(lat.shift(s0, 1, -1) == lat.site(0, 4, 0, 0), "-y wraps to Ly-1");
    check(lat.shift(s0, 2, -1) == lat.site(0, 0, 5, 0), "-z wraps to Lz-1");
    check(lat.shift(s0, 3, -1) == lat.site(0, 0, 0, 2), "-t wraps to Lt-1");

    // Round-trip: +mu followed by -mu returns to start.
    for (int mu = 0; mu < 4; ++mu) {
        const int sp = lat.shift(s, mu, +1);
        check(lat.shift(sp, mu, -1) == s, "+mu then -mu returns to origin");
    }
}

void test_lattice_starts(RNG& rng) {
    Lattice lat(4, 4, 4, 2);

    lat.cold_start();
    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu)
            check(approx_equal(lat.U(s, mu), SU2::identity()),
                  "cold-start link is identity");

    lat.hot_start(rng);
    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu)
            check(std::abs(lat.U(s, mu).norm_sq() - 1.0) < 1e-12,
                  "hot-start link is unitary");

    lat.reunitarise_all();
    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu)
            check(std::abs(lat.U(s, mu).norm_sq() - 1.0) < 1e-15,
                  "reunitarise_all preserves unitarity");
}

void test_centre_symmetry_skeleton() {
    // Spot-check the data-structure operation that the (Step 7) Polyakov-
    // loop centre transformation will need: we should be able to flip
    // the sign of every t-direction link in a single time slice and still
    // have unitary links.
    Lattice lat(3, 3, 3, 4);
    lat.cold_start();
    const int t_slice = 1;
    for (int z = 0; z < lat.size(2); ++z)
    for (int y = 0; y < lat.size(1); ++y)
    for (int x = 0; x < lat.size(0); ++x) {
        const int s = lat.site(x, y, z, t_slice);
        SU2& U = lat.U(s, 3);
        U = SU2{-U.a0, -U.a1, -U.a2, -U.a3};
    }
    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu)
            check(std::abs(lat.U(s, mu).norm_sq() - 1.0) < 1e-15,
                  "centre flip preserves unit norm");
}

}  // namespace

int main() {
    RNG rng(0xDEADBEEFULL);

    test_identity();                    std::printf("[ok] identity\n");
    test_known_product();               std::printf("[ok] known product\n");
    test_unitary(rng);                  std::printf("[ok] unitarity (1000 random)\n");
    test_associativity(rng);            std::printf("[ok] associativity (200 triples)\n");
    test_trace_cyclic(rng);             std::printf("[ok] trace cyclicity (200 triples)\n");
    test_near_identity(rng);            std::printf("[ok] near-identity proposal (1000)\n");
    test_reunitarise(rng);              std::printf("[ok] reunitarise after drift\n");
    test_lattice_indexing();            std::printf("[ok] lattice indexing & PBC\n");
    test_lattice_starts(rng);           std::printf("[ok] cold / hot starts\n");
    test_centre_symmetry_skeleton();    std::printf("[ok] centre-flip preserves norms\n");

    std::printf("\nAll Step 4 tests passed (%d assertions).\n", g_check_count);
    return 0;
}
