#pragma once
#include <cstdint>

namespace su2lgt {

class Lattice;
class RNG;

// Cumulative counts of proposed / accepted Metropolis hits.
struct UpdateStats {
    long long proposed = 0;
    long long accepted = 0;
    double acceptance_rate() const {
        return proposed > 0
             ? static_cast<double>(accepted) / static_cast<double>(proposed)
             : 0.0;
    }
};

// Single-site Metropolis updater for pure SU(2) Wilson gauge theory.
//
// One sweep visits every link in lexicographic order (site index, mu).
// At each link the staple A is computed once and `n_hits` Metropolis
// proposals U -> X U are tried in sequence, with X drawn from
// `RNG::random_near_identity(eps)`.  The proposal is symmetric so the
// acceptance criterion is the plain min(1, exp(-Delta S)).
//
// The updater holds references to the lattice and RNG; no copies are made.
class MetropolisUpdater {
public:
    MetropolisUpdater(Lattice& lat,
                      RNG& rng,
                      double beta,
                      double eps,
                      int n_hits);

    // Run one full sweep.  Returns the number of accepted hits in
    // this sweep (also accumulates into `stats`).
    long long sweep();

    UpdateStats stats() const { return stats_; }
    void reset_stats() { stats_ = UpdateStats{}; }

    // Allow runtime tuning of beta and proposal width without
    // reconstructing the updater.
    void   set_beta(double beta) { beta_ = beta; }
    double beta() const { return beta_; }

    void   set_eps(double eps) { eps_ = eps; }
    double eps() const { return eps_; }

    int n_hits() const { return n_hits_; }

private:
    Lattice& lat_;
    RNG&     rng_;
    double   beta_;
    double   eps_;
    int      n_hits_;
    UpdateStats stats_{};
};

}  // namespace su2lgt
