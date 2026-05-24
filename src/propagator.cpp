#include "su2lgt/propagator.hpp"

#include <algorithm>
#include <cstddef>

namespace su2lgt {

std::vector<double> static_propagator(const Lattice& lat, int r_max) {
    const int Lx = lat.size(0);
    const int Ly = lat.size(1);
    const int Lz = lat.size(2);
    const int Lt = lat.size(3);
    const int Vs = Lx * Ly * Lz;

    // Layout: A_avg has shape [colour=3][mu=4][V_s].  Stored as a flat
    // array, indexed as a*4*Vs + mu*Vs + spatial_idx.
    auto idx_aux = [&](int a, int mu, int x, int y, int z) {
        return a * 4 * Vs + mu * Vs
             + (z * Ly + y) * Lx + x;
    };

    std::vector<double> A_avg(static_cast<std::size_t>(3) * 4 * Vs, 0.0);

    // Step 1: t-average each link's algebra components.
    for (int t = 0; t < Lt; ++t) {
        for (int z = 0; z < Lz; ++z)
            for (int y = 0; y < Ly; ++y)
                for (int x = 0; x < Lx; ++x) {
                    const int s = lat.site(x, y, z, t);
                    for (int mu = 0; mu < 4; ++mu) {
                        const SU2& U = lat.U(s, mu);
                        A_avg[idx_aux(0, mu, x, y, z)] += U.a1;
                        A_avg[idx_aux(1, mu, x, y, z)] += U.a2;
                        A_avg[idx_aux(2, mu, x, y, z)] += U.a3;
                    }
                }
    }
    const double inv_Lt = 1.0 / static_cast<double>(Lt);
    for (auto& v : A_avg) v *= inv_Lt;

    // Step 2: position-space correlator, averaged over the three
    // spatial axes and over starting site.
    std::vector<double> G(r_max + 1, 0.0);
    long long count_per_r = 0;

    for (int z = 0; z < Lz; ++z)
        for (int y = 0; y < Ly; ++y)
            for (int x = 0; x < Lx; ++x) {
                ++count_per_r;       // counts starting points; same for all r
                for (int r = 0; r <= r_max; ++r) {
                    const int xr = (x + r) % Lx;
                    const int yr = (y + r) % Ly;
                    const int zr = (z + r) % Lz;

                    double sum = 0.0;
                    for (int a = 0; a < 3; ++a)
                        for (int mu = 0; mu < 4; ++mu) {
                            const double v = A_avg[idx_aux(a, mu, x, y, z)];
                            sum += v * A_avg[idx_aux(a, mu, xr, y,  z )];
                            sum += v * A_avg[idx_aux(a, mu, x,  yr, z )];
                            sum += v * A_avg[idx_aux(a, mu, x,  y,  zr)];
                        }
                    G[r] += sum;
                }
            }

    // Normalisation: each r had count_per_r starting points,
    // each contributing 3 axial measurements.
    const double norm = 1.0 / static_cast<double>(3 * count_per_r);
    for (auto& g : G) g *= norm;

    return G;
}

}  // namespace su2lgt
