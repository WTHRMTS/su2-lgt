#include "su2lgt/lattice.hpp"
#include "su2lgt/rng.hpp"

#include <algorithm>
#include <stdexcept>

namespace su2lgt {

Lattice::Lattice(int Lx, int Ly, int Lz, int Lt) {
    if (Lx <= 0 || Ly <= 0 || Lz <= 0 || Lt <= 0)
        throw std::invalid_argument("Lattice extents must be positive");
    L_[0] = Lx; L_[1] = Ly; L_[2] = Lz; L_[3] = Lt;
    volume_ = Lx * Ly * Lz * Lt;
    links_.assign(4 * volume_, SU2::identity());
}

int Lattice::site(int x, int y, int z, int t) const {
    // x is the fastest-varying coordinate in the flat index.
    return ((t * L_[2] + z) * L_[1] + y) * L_[0] + x;
}

void Lattice::coords(int s, int& x, int& y, int& z, int& t) const {
    x = s % L_[0];   s /= L_[0];
    y = s % L_[1];   s /= L_[1];
    z = s % L_[2];   s /= L_[2];
    t = s;
}

int Lattice::shift(int s, int mu, int sign) const {
    int x, y, z, t;
    coords(s, x, y, z, t);
    int* coord[4] = {&x, &y, &z, &t};
    int& c = *coord[mu];
    if (sign > 0) c = (c + 1) % L_[mu];
    else          c = (c + L_[mu] - 1) % L_[mu];
    return site(x, y, z, t);
}

void Lattice::cold_start() {
    std::fill(links_.begin(), links_.end(), SU2::identity());
}

void Lattice::hot_start(RNG& rng) {
    for (auto& U : links_) U = rng.haar_su2();
}

void Lattice::reunitarise_all() {
    for (auto& U : links_) U.reunitarise();
}

}  // namespace su2lgt
