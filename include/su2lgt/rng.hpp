#pragma once
#include <cstdint>
#include <random>
#include <cmath>
#include <algorithm>

#include "su2lgt/su2.hpp"

namespace su2lgt {

class RNG {
    std::mt19937_64 gen_;
public:
    explicit RNG(std::uint64_t seed) : gen_(seed) {}

    double uniform() {
        std::uniform_real_distribution<double> d(0.0, 1.0);
        return d(gen_);
    }

    double normal() {
        std::normal_distribution<double> d(0.0, 1.0);
        return d(gen_);
    }

    // Uniformly distributed on SU(2) (which is S^3) in the Haar measure.
    // Implementation: four iid Gaussians normalised to unit length.
    SU2 haar_su2() {
        SU2 U;
        double n;
        do {
            U.a0 = normal();
            U.a1 = normal();
            U.a2 = normal();
            U.a3 = normal();
            n = U.norm_sq();
        } while (n < 1e-30);
        const double inv = 1.0 / std::sqrt(n);
        U.a0 *= inv; U.a1 *= inv; U.a2 *= inv; U.a3 *= inv;
        return U;
    }

    // SU(2) element close to the identity, used as a Metropolis proposal:
    //   X = a0 I + i (a . sigma)
    // with a0 in [sqrt(1 - eps^2), 1] and a uniform on the 2-sphere of
    // radius sqrt(1 - a0^2).  The proposal is symmetric under
    // X -> X^dagger because that maps a -> -a and the sphere is
    // negation-invariant.  Ergodicity over SU(2) follows from composing
    // many small steps over the chain.
    SU2 random_near_identity(double eps) {
        const double r  = uniform();
        const double a0 = std::sqrt(std::max(0.0, 1.0 - eps*eps * r*r));
        const double s  = std::sqrt(std::max(0.0, 1.0 - a0*a0));

        // Marsaglia method for a uniform unit vector on S^2.
        double x, y, q;
        do {
            x = 2.0*uniform() - 1.0;
            y = 2.0*uniform() - 1.0;
            q = x*x + y*y;
        } while (q >= 1.0 || q == 0.0);
        const double f  = 2.0 * std::sqrt(1.0 - q);
        const double n1 = x * f;
        const double n2 = y * f;
        const double n3 = 1.0 - 2.0*q;

        return SU2{a0, s*n1, s*n2, s*n3};
    }
};

}  // namespace su2lgt
