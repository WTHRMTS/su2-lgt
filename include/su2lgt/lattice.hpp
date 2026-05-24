#pragma once
#include <vector>

#include "su2lgt/su2.hpp"

namespace su2lgt {

class RNG;

// 4D Euclidean lattice with periodic boundary conditions in every
// direction.  Direction convention: 0=x, 1=y, 2=z, 3=t.  The site
// indexing has x as the fastest-varying coordinate, which matters
// because the inner Metropolis loop will iterate over neighbours in x,
// and we want those to be cache-adjacent.
//
// Each site carries four link variables U_mu(x), one per direction.
// Links are stored interleaved by direction: link_index = 4*site + mu.
class Lattice {
public:
    Lattice(int Lx, int Ly, int Lz, int Lt);

    int  size(int mu) const  { return L_[mu]; }
    int  volume()    const   { return volume_; }
    int  num_links() const   { return 4 * volume_; }

    // (x,y,z,t) -> flat site index.
    int site(int x, int y, int z, int t) const;

    // Flat site index -> coordinates.
    void coords(int site_idx, int& x, int& y, int& z, int& t) const;

    // Site obtained by shifting `site_idx` along direction mu by +1
    // (sign=+1) or -1 (sign=-1), with periodic wrap-around.
    int shift(int site_idx, int mu, int sign) const;

    // Link variable accessors.  No bounds-checking in Release builds.
    SU2&       U(int site_idx, int mu)       { return links_[4*site_idx + mu]; }
    const SU2& U(int site_idx, int mu) const { return links_[4*site_idx + mu]; }

    // Initialisation strategies.
    void cold_start();              // every link = identity
    void hot_start(RNG& rng);       // every link = Haar-random SU(2)

    // Restore unit norm on every link (call periodically during long runs).
    void reunitarise_all();

private:
    int L_[4];
    int volume_;
    std::vector<SU2> links_;
};

}  // namespace su2lgt
