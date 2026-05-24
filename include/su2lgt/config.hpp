#pragma once
#include <cstdint>
#include <iosfwd>
#include <string>

namespace su2lgt {

// Run-time parameters for one Monte Carlo run at a single (lattice, beta)
// point.  Sweeps over beta are driven externally (a shell script invokes
// the binary repeatedly).  A run dumps one CSV file in `out_dir`.
struct RunConfig {
    // Lattice extents.  --Ls sets Lx=Ly=Lz at once for the standard
    // Ns^3 x Nt geometry; individual extents can also be overridden.
    int Lx = 8;
    int Ly = 8;
    int Lz = 8;
    int Lt = 4;

    // Inverse coupling.  beta = 4 / g^2 in the Wilson convention.
    double beta = 2.30;

    // Sweep counts.
    int n_therm    = 2000;
    int n_meas     = 100000;
    int meas_every = 1;

    // Metropolis proposal width (radians scale of near-identity step)
    // and number of update attempts per link per sweep.
    double prop_eps = 0.3;
    int    n_hits   = 5;

    // RNG seed.
    std::uint64_t seed = 0xC0FFEEBABEF00DULL;

    // Output.
    std::string out_dir = "data";
    std::string label;        // optional user tag; if empty, name auto

    // Print to stream as CSV-style comment lines.
    void print(std::ostream& os) const;

    // Auto-generated CSV filename: e.g. Ls8_Lt4_beta2.300_<label>.csv
    std::string auto_filename() const;
};

}  // namespace su2lgt
