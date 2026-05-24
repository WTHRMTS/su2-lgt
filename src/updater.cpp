#include "su2lgt/updater.hpp"

#include "su2lgt/action.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/rng.hpp"
#include "su2lgt/su2.hpp"

#include <cmath>

namespace su2lgt {

MetropolisUpdater::MetropolisUpdater(Lattice& lat,
                                     RNG& rng,
                                     double beta,
                                     double eps,
                                     int n_hits)
    : lat_(lat), rng_(rng), beta_(beta), eps_(eps), n_hits_(n_hits) {}

long long MetropolisUpdater::sweep() {
    long long acc_this_sweep = 0;
    const int V = lat_.volume();

    for (int s = 0; s < V; ++s) {
        for (int mu = 0; mu < 4; ++mu) {

            // Staple once per link: shared across all `n_hits` proposals.
            const SU2 A = staple_sum(lat_, s, mu);

            SU2& U = lat_.U(s, mu);

            // Cache (U * A).a0 so that each accepted hit only needs to
            // recompute the new value.  This saves one quaternion product
            // per hit when proposals are rejected (the common case).
            double UA_a0 = (U * A).a0;

            for (int h = 0; h < n_hits_; ++h) {
                const SU2 X     = rng_.random_near_identity(eps_);
                const SU2 U_new = X * U;
                const double UA_a0_new = (U_new * A).a0;
                const double dS = -beta_ * (UA_a0_new - UA_a0);

                ++stats_.proposed;
                if (dS <= 0.0 || rng_.uniform() < std::exp(-dS)) {
                    U = U_new;
                    UA_a0 = UA_a0_new;
                    ++stats_.accepted;
                    ++acc_this_sweep;
                }
            }
        }
    }
    return acc_this_sweep;
}

}  // namespace su2lgt
