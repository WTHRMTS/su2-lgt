#include "su2lgt/observables.hpp"

namespace su2lgt {

double polyakov_local(const Lattice& lat, int x, int y, int z) {
    SU2 P = SU2::identity();
    const int Lt = lat.size(3);
    for (int t = 0; t < Lt; ++t) {
        P = P * lat.U(lat.site(x, y, z, t), 3);
    }
    return P.half_trace();
}

double polyakov_global(const Lattice& lat) {
    const int Lx = lat.size(0);
    const int Ly = lat.size(1);
    const int Lz = lat.size(2);
    double sum = 0.0;
    for (int z = 0; z < Lz; ++z)
        for (int y = 0; y < Ly; ++y)
            for (int x = 0; x < Lx; ++x)
                sum += polyakov_local(lat, x, y, z);
    return sum / static_cast<double>(Lx * Ly * Lz);
}

ObsRecord measure(const Lattice& lat, long long sweep_index) {
    ObsRecord r;
    r.sweep = sweep_index;
    r.L     = polyakov_global(lat);
    r.absL  = std::abs(r.L);
    r.L2    = r.L * r.L;
    r.L4    = r.L2 * r.L2;
    r.plaq  = mean_plaquette(lat);
    return r;
}

}  // namespace su2lgt
