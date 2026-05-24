#include "su2lgt/action.hpp"

namespace su2lgt {

SU2 plaquette(const Lattice& lat, int site, int mu, int nu) {
    const int s_mu = lat.shift(site, mu, +1);   // x + mu
    const int s_nu = lat.shift(site, nu, +1);   // x + nu
    return lat.U(site, mu)
         * lat.U(s_mu, nu)
         * lat.U(s_nu, mu).dagger()
         * lat.U(site, nu).dagger();
}

double total_plaquette_trace(const Lattice& lat) {
    double sum = 0.0;
    const int V = lat.volume();
    for (int s = 0; s < V; ++s) {
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = mu + 1; nu < 4; ++nu) {
                sum += plaquette(lat, s, mu, nu).half_trace();
            }
        }
    }
    return sum;
}

SU2 staple_sum(const Lattice& lat, int site, int mu) {
    SU2 A{0.0, 0.0, 0.0, 0.0};

    const int s_mu = lat.shift(site, mu, +1);   // x + mu

    for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) continue;

        const int s_nu          = lat.shift(site, nu, +1);  // x + nu
        const int s_minus_nu    = lat.shift(site, nu, -1);  // x - nu
        const int s_mu_minus_nu = lat.shift(s_mu, nu, -1);  // x + mu - nu

        // Forward staple: completes the (mu, +nu) plaquette starting at x.
        const SU2 forward = lat.U(s_mu, nu)
                          * lat.U(s_nu, mu).dagger()
                          * lat.U(site,  nu).dagger();

        // Backward staple: completes the (mu, -nu) plaquette starting at x.
        const SU2 backward = lat.U(s_mu_minus_nu, nu).dagger()
                           * lat.U(s_minus_nu,    mu).dagger()
                           * lat.U(s_minus_nu,    nu);

        A = A + forward + backward;
    }
    return A;
}

double delta_action_from_staple(double beta,
                                const SU2& U_old,
                                const SU2& U_new,
                                const SU2& staple) {
    return -beta * ((U_new * staple).a0 - (U_old * staple).a0);
}

}  // namespace su2lgt
