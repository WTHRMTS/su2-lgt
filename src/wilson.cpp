#include "su2lgt/wilson.hpp"

#include "su2lgt/observables.hpp"  // for polyakov_local

namespace su2lgt {

double wilson_loop(const Lattice& lat, int site, int mu, int nu,
                   int R, int T) {
    SU2 P = SU2::identity();
    int s = site;

    // +mu, R links
    for (int i = 0; i < R; ++i) {
        P = P * lat.U(s, mu);
        s = lat.shift(s, mu, +1);
    }
    // +nu, T links
    for (int i = 0; i < T; ++i) {
        P = P * lat.U(s, nu);
        s = lat.shift(s, nu, +1);
    }
    // -mu, R links (use dagger of the link from the lower-coordinate end)
    for (int i = 0; i < R; ++i) {
        s = lat.shift(s, mu, -1);
        P = P * lat.U(s, mu).dagger();
    }
    // -nu, T links
    for (int i = 0; i < T; ++i) {
        s = lat.shift(s, nu, -1);
        P = P * lat.U(s, nu).dagger();
    }
    return P.half_trace();
}

double mean_spatial_temporal_wilson_loop(const Lattice& lat, int R, int T) {
    const int t_dir = 3;
    double sum = 0.0;
    long long count = 0;
    const int V = lat.volume();
    for (int s = 0; s < V; ++s) {
        for (int mu = 0; mu < 3; ++mu) {     // mu spatial only
            sum += wilson_loop(lat, s, mu, t_dir, R, T);
            ++count;
        }
    }
    return sum / static_cast<double>(count);
}

std::vector<double> polyakov_local_field(const Lattice& lat) {
    const int Lx = lat.size(0);
    const int Ly = lat.size(1);
    const int Lz = lat.size(2);
    std::vector<double> f(static_cast<std::size_t>(Lx) * Ly * Lz);
    for (int z = 0; z < Lz; ++z)
        for (int y = 0; y < Ly; ++y)
            for (int x = 0; x < Lx; ++x)
                f[(static_cast<std::size_t>(z) * Ly + y) * Lx + x] =
                    polyakov_local(lat, x, y, z);
    return f;
}

std::vector<double> polyakov_pair_correlator(const Lattice& lat, int r_max) {
    const int Lx = lat.size(0);
    const int Ly = lat.size(1);
    const int Lz = lat.size(2);

    const auto f = polyakov_local_field(lat);

    std::vector<double> C(r_max + 1, 0.0);
    std::vector<long long> count(r_max + 1, 0);

    auto idx = [&](int x, int y, int z) {
        return (static_cast<std::size_t>(z) * Ly + y) * Lx + x;
    };

    for (int z = 0; z < Lz; ++z)
        for (int y = 0; y < Ly; ++y)
            for (int x = 0; x < Lx; ++x) {
                const double l1 = f[idx(x, y, z)];
                for (int r = 0; r <= r_max; ++r) {
                    // Average over the three positive axial directions.
                    const double cx = f[idx((x + r) % Lx, y, z)];
                    const double cy = f[idx(x, (y + r) % Ly, z)];
                    const double cz = f[idx(x, y, (z + r) % Lz)];
                    C[r]    += l1 * (cx + cy + cz);
                    count[r] += 3;
                }
            }

    for (int r = 0; r <= r_max; ++r) C[r] /= static_cast<double>(count[r]);
    return C;
}

}  // namespace su2lgt
