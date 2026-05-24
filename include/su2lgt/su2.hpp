#pragma once
#include <cmath>

namespace su2lgt {

// Every U in SU(2) can be written
//
//     U = a0 * I + i * (a1*sigma_x + a2*sigma_y + a3*sigma_z)
//
// with a0^2 + a1^2 + a2^2 + a3^2 = 1.  We store it as a unit 4-vector
// (a0, a1, a2, a3) -- equivalently a unit quaternion.  All algebra
// reduces to real arithmetic, the trace is trivial (Tr U = 2 a0), and
// the dagger is a sign flip on the spatial part.
struct SU2 {
    double a0, a1, a2, a3;

    static constexpr SU2 identity() { return {1.0, 0.0, 0.0, 0.0}; }

    constexpr SU2 dagger() const { return {a0, -a1, -a2, -a3}; }

    // (1/2) Re Tr U  =  a0
    constexpr double half_trace() const { return a0; }

    constexpr double norm_sq() const {
        return a0*a0 + a1*a1 + a2*a2 + a3*a3;
    }

    // Drift-correction: divide out accumulated norm error so the
    // element stays exactly on S^3 after many multiplications.
    void reunitarise() {
        const double n = std::sqrt(norm_sq());
        a0 /= n; a1 /= n; a2 /= n; a3 /= n;
    }
};

// Quaternion product:
//   (UV).0 = U.0 V.0 - U.a . V.a
//   (UV).a = U.0 V.a + V.0 U.a - U.a x V.a
// where the cross product gets a minus sign because (a.sigma)(b.sigma)
// = (a.b) I + i (a x b).sigma and the prefactor i^2 = -1 flips it.
constexpr SU2 operator*(const SU2& U, const SU2& V) {
    return {
        U.a0*V.a0 - U.a1*V.a1 - U.a2*V.a2 - U.a3*V.a3,
        U.a0*V.a1 + U.a1*V.a0 - U.a2*V.a3 + U.a3*V.a2,
        U.a0*V.a2 + U.a2*V.a0 - U.a3*V.a1 + U.a1*V.a3,
        U.a0*V.a3 + U.a3*V.a0 - U.a1*V.a2 + U.a2*V.a1
    };
}

// Sum and scalar multiplication.  These do *not* preserve unitarity --
// they are used for staple sums (sums of products of links around an
// open path) which appear in the local-energy calculation in Step 5.
constexpr SU2 operator+(const SU2& U, const SU2& V) {
    return {U.a0 + V.a0, U.a1 + V.a1, U.a2 + V.a2, U.a3 + V.a3};
}

constexpr SU2 operator*(double s, const SU2& U) {
    return {s*U.a0, s*U.a1, s*U.a2, s*U.a3};
}

}  // namespace su2lgt
