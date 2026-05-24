#pragma once
#include <cmath>

#include "su2lgt/action.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/su2.hpp"

namespace su2lgt {

// ---- Polyakov loop ----------------------------------------------------
//
// Local Polyakov loop at spatial site (x, y, z):
//
//     l(x) = (1/2) Re Tr [ U_0(x, t=0) U_0(x, t=1) ... U_0(x, t=Lt-1) ]
//
// where the product is taken in temporal order along the periodic time
// direction.  For SU(2), the trace is real and l(x) lies in [-1, 1].
double polyakov_local(const Lattice& lat, int x, int y, int z);

// Global Polyakov loop = spatial average of l(x):
//
//     L = (1 / Ns^3) sum_x l(x)
//
// This is the order parameter for the Z_2 centre-symmetry-breaking
// (deconfinement) phase transition.
double polyakov_global(const Lattice& lat);

// ---- Per-configuration measurement record ----------------------------
//
// One record per measurement sweep.  L^2 and L^4 are kept separately
// because they (and only they) feed the Binder cumulant
//
//     U_4 = 1 - <L^4> / (3 <L^2>^2),
//
// and |L| is kept because <|L|> is the practical order-parameter
// estimator on a finite lattice (where <L> = 0 by symmetry).
struct ObsRecord {
    long long sweep = 0;
    double L     = 0.0;     // global Polyakov loop
    double absL  = 0.0;     // |L|
    double L2    = 0.0;     // L^2
    double L4    = 0.0;     // L^4
    double plaq  = 0.0;     // mean plaquette  (1/2) <Re Tr U_p>
};

// One-shot per-configuration measurement.
ObsRecord measure(const Lattice& lat, long long sweep_index);

// ---- Running aggregator ----------------------------------------------
//
// Accumulates first/second/fourth moments of L and first moments of
// |L| and the plaquette across many records.  Emits the standard
// derived quantities at the end of a run:
//
//     <L>, <|L|>, <L^2>, <L^4>, <plaq>,
//     Binder cumulant   U_4 = 1 - <L^4> / (3 <L^2>^2),
//     susceptibility    chi = N_s^3 * (<L^2> - <|L|>^2).
//
// These are the headline outputs of the J-sweep at Step 8.
class ObsAccumulator {
public:
    explicit ObsAccumulator(int spatial_volume) : Vs_(spatial_volume) {}

    void push(const ObsRecord& r) {
        ++n_;
        sumL_    += r.L;
        sumAbsL_ += r.absL;
        sumL2_   += r.L2;
        sumL4_   += r.L4;
        sumP_    += r.plaq;
    }

    long long n() const { return n_; }

    double mean_L()    const { return n_ ? sumL_    / n_ : 0.0; }
    double mean_absL() const { return n_ ? sumAbsL_ / n_ : 0.0; }
    double mean_L2()   const { return n_ ? sumL2_   / n_ : 0.0; }
    double mean_L4()   const { return n_ ? sumL4_   / n_ : 0.0; }
    double mean_plaq() const { return n_ ? sumP_    / n_ : 0.0; }

    // Binder cumulant of the global Polyakov loop.
    double binder_U4() const {
        const double L2 = mean_L2();
        if (L2 <= 0.0) return 0.0;
        return 1.0 - mean_L4() / (3.0 * L2 * L2);
    }

    // Polyakov-loop susceptibility (spatial-volume normalisation).
    double susceptibility() const {
        const double aL = mean_absL();
        return static_cast<double>(Vs_) * (mean_L2() - aL * aL);
    }

private:
    int       Vs_;
    long long n_ = 0;
    double    sumL_    = 0.0;
    double    sumAbsL_ = 0.0;
    double    sumL2_   = 0.0;
    double    sumL4_   = 0.0;
    double    sumP_    = 0.0;
};

}  // namespace su2lgt
