#include "su2lgt/action.hpp"
#include "su2lgt/config.hpp"
#include "su2lgt/gauge_fix.hpp"
#include "su2lgt/lattice.hpp"
#include "su2lgt/observables.hpp"
#include "su2lgt/propagator.hpp"
#include "su2lgt/rng.hpp"
#include "su2lgt/updater.hpp"
#include "su2lgt/wilson.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace su2lgt;
namespace fs = std::filesystem;

namespace {

void print_usage() {
    std::cout <<
        "Usage: su2lgt [options]\n"
        "  --Ls N                 spatial extent (sets Lx=Ly=Lz=N)\n"
        "  --Lx N / --Ly N / --Lz N   override individual extents\n"
        "  --Lt N                 temporal extent (default 4)\n"
        "  --beta B               inverse coupling beta = 4/g^2\n"
        "  --n_therm N            thermalisation sweeps\n"
        "  --n_meas N             measurement sweeps\n"
        "  --meas_every k         measure every k-th sweep\n"
        "  --prop_eps e           Metropolis proposal width\n"
        "  --n_hits N             Metropolis hits per link per sweep\n"
        "  --seed S               RNG seed (decimal or 0x...)\n"
        "  --out_dir DIR          CSV output directory (default: data)\n"
        "  --label TAG            optional run label appended to filename\n"
        "  --start cold|hot       initial configuration\n"
        "  --no_csv               do not write CSV, only print summary\n"
        "  --no_wilson            skip Wilson-loop / pair-correlator measurements\n"
        "  --prop_every K         measure boson propagator every K-th meas (0=off)\n"
        "  --prop_max_iter N      gauge-fix iteration cap (default 2000)\n"
        "  --help, -h             this message\n";
}

double dbl(const char* s)        { return std::strtod(s, nullptr); }
int    integer(const char* s)    { return std::atoi(s); }
std::uint64_t u64(const char* s) { return std::strtoull(s, nullptr, 0); }

void write_csv_header(std::ostream& os, const RunConfig& cfg,
                      int r_max, int R_max, int T_max,
                      bool wilson_enabled,
                      bool prop_enabled) {
    os << "# Lx=" << cfg.Lx
       << " Ly=" << cfg.Ly
       << " Lz=" << cfg.Lz
       << " Lt=" << cfg.Lt
       << " beta=" << cfg.beta
       << " n_therm=" << cfg.n_therm
       << " n_meas=" << cfg.n_meas
       << " meas_every=" << cfg.meas_every
       << " prop_eps=" << cfg.prop_eps
       << " n_hits=" << cfg.n_hits
       << " seed=" << cfg.seed
       << " r_max=" << r_max
       << " R_max=" << R_max
       << " T_max=" << T_max
       << " wilson=" << (wilson_enabled ? 1 : 0)
       << " propagator=" << (prop_enabled ? 1 : 0)
       << "\n";

    os << "sweep,L,absL,L2,L4,plaq";
    if (wilson_enabled) {
        for (int r = 0; r <= r_max; ++r)
            os << ",Pcorr_r" << r;
        for (int R = 1; R <= R_max; ++R)
            for (int T = 1; T <= T_max; ++T)
                os << ",W_R" << R << "_T" << T;
    }
    if (prop_enabled) {
        for (int r = 0; r <= r_max; ++r)
            os << ",Gprop_r" << r;
    }
    os << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    RunConfig cfg;
    std::string start = "cold";
    bool no_csv      = false;
    bool no_wilson   = false;
    int  prop_every  = 0;
    int  prop_max_iter = 2000;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](int k) {
            if (i + k >= argc) {
                std::cerr << "Missing argument after " << a << "\n";
                print_usage();
                std::exit(1);
            }
        };
        if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else if (a == "--Ls") {
            need(1); const int v = integer(argv[++i]);
            cfg.Lx = cfg.Ly = cfg.Lz = v;
        }
        else if (a == "--Lx")        { need(1); cfg.Lx = integer(argv[++i]); }
        else if (a == "--Ly")        { need(1); cfg.Ly = integer(argv[++i]); }
        else if (a == "--Lz")        { need(1); cfg.Lz = integer(argv[++i]); }
        else if (a == "--Lt")        { need(1); cfg.Lt = integer(argv[++i]); }
        else if (a == "--beta")      { need(1); cfg.beta = dbl(argv[++i]); }
        else if (a == "--n_therm")   { need(1); cfg.n_therm = integer(argv[++i]); }
        else if (a == "--n_meas")    { need(1); cfg.n_meas  = integer(argv[++i]); }
        else if (a == "--meas_every"){ need(1); cfg.meas_every = integer(argv[++i]); }
        else if (a == "--prop_eps")  { need(1); cfg.prop_eps = dbl(argv[++i]); }
        else if (a == "--n_hits")    { need(1); cfg.n_hits = integer(argv[++i]); }
        else if (a == "--seed")      { need(1); cfg.seed = u64(argv[++i]); }
        else if (a == "--out_dir")   { need(1); cfg.out_dir = argv[++i]; }
        else if (a == "--label")     { need(1); cfg.label = argv[++i]; }
        else if (a == "--start")     { need(1); start = argv[++i]; }
        else if (a == "--no_csv")    { no_csv = true; }
        else if (a == "--no_wilson") { no_wilson = true; }
        else if (a == "--prop_every"){ need(1); prop_every = integer(argv[++i]); }
        else if (a == "--prop_max_iter"){ need(1); prop_max_iter = integer(argv[++i]); }
        else { std::cerr << "Unknown argument: " << a << "\n"; print_usage(); return 1; }
    }

    cfg.print(std::cout);

    Lattice lat(cfg.Lx, cfg.Ly, cfg.Lz, cfg.Lt);
    RNG rng(cfg.seed);

    if (start == "cold")     lat.cold_start();
    else if (start == "hot") lat.hot_start(rng);
    else { std::cerr << "Bad --start: " << start << "\n"; return 1; }

    const int Vs = cfg.Lx * cfg.Ly * cfg.Lz;
    const int r_max = std::min({cfg.Lx, cfg.Ly, cfg.Lz}) / 2;
    const int R_max = std::min({cfg.Lx, cfg.Ly, cfg.Lz}) / 2;
    const int T_max = std::max(1, cfg.Lt - 1);
    const bool prop_enabled = (prop_every > 0);

    std::cout
        << "# Lattice initialised: "
        << cfg.Lx << "x" << cfg.Ly << "x" << cfg.Lz << "x" << cfg.Lt
        << " (" << lat.volume() << " sites, "
        << lat.num_links() << " links, Vs=" << Vs
        << "), start=" << start << "\n"
        << "# Wilson grid: R in [1," << R_max << "], T in [1," << T_max
        << "], pair-corr r in [0," << r_max << "]"
        << (no_wilson ? " (DISABLED)" : "") << "\n"
        << "# Propagator: " << (prop_enabled
              ? ("every " + std::to_string(prop_every) + " meas, max "
                 + std::to_string(prop_max_iter) + " gauge-fix iter")
              : std::string("DISABLED")) << "\n";

    MetropolisUpdater upd(lat, rng, cfg.beta, cfg.prop_eps, cfg.n_hits);
    constexpr int reunit_every = 100;

    auto log_line = [&](const char* phase, int n) {
        std::printf("# %-7s %6d   <plaq>=%.5f   L=%+.5f   acc=%.4f\n",
                    phase, n,
                    mean_plaquette(lat),
                    polyakov_global(lat),
                    upd.stats().acceptance_rate());
    };

    // ---- Thermalisation ----
    std::cout << "#\n# === Thermalisation (" << cfg.n_therm << " sweeps) ===\n";
    const int log_every_therm = std::max(1, cfg.n_therm / 10);
    for (int s = 0; s < cfg.n_therm; ++s) {
        upd.sweep();
        if ((s + 1) % reunit_every == 0) lat.reunitarise_all();
        if ((s + 1) % log_every_therm == 0 || s + 1 == cfg.n_therm)
            log_line("therm", s + 1);
    }
    upd.reset_stats();

    // ---- Measurement ----
    std::cout << "#\n# === Measurement (" << cfg.n_meas << " sweeps) ===\n";
    const int log_every_meas = std::max(1, cfg.n_meas / 10);

    std::ofstream csv;
    std::string csv_path;
    if (!no_csv && !cfg.out_dir.empty()) {
        std::error_code ec;
        fs::create_directories(cfg.out_dir, ec);
        csv_path = cfg.out_dir + "/" + cfg.auto_filename();
        csv.open(csv_path);
        if (!csv) {
            std::cerr << "WARN: cannot open " << csv_path << "\n";
        } else {
            write_csv_header(csv, cfg, r_max, R_max, T_max,
                             !no_wilson, prop_enabled);
            std::cout << "# CSV output: " << csv_path << "\n";
        }
    }

    ObsAccumulator acc(Vs);

    std::vector<double> Pcorr_sum(r_max + 1, 0.0);
    std::vector<std::vector<double>> W_sum(R_max + 1,
                                           std::vector<double>(T_max + 1, 0.0));
    long long w_n = 0;

    std::vector<double> Gprop_sum(r_max + 1, 0.0);
    long long g_n = 0;
    int meas_index = 0;     // counts measurements actually pushed

    for (int s = 0; s < cfg.n_meas; ++s) {
        upd.sweep();
        if ((s + 1) % reunit_every == 0) lat.reunitarise_all();

        if (cfg.meas_every <= 1 || s % cfg.meas_every == 0) {
            const ObsRecord r = measure(lat, s);
            acc.push(r);

            std::vector<double> Pcorr;
            std::vector<double> W_flat;
            if (!no_wilson) {
                Pcorr = polyakov_pair_correlator(lat, r_max);
                W_flat.reserve(static_cast<std::size_t>(R_max) * T_max);
                for (int R = 1; R <= R_max; ++R)
                    for (int T = 1; T <= T_max; ++T) {
                        const double w = mean_spatial_temporal_wilson_loop(
                                            lat, R, T);
                        W_flat.push_back(w);
                        W_sum[R][T] += w;
                    }
                for (int rr = 0; rr <= r_max; ++rr) Pcorr_sum[rr] += Pcorr[rr];
                ++w_n;
            }

            // Boson propagator on a gauge-fixed copy, every prop_every-th meas.
            std::vector<double> Gprop;
            const bool measure_prop = prop_enabled
                                  && (meas_index % prop_every == 0);
            if (measure_prop) {
                Lattice copy = lat;
                copy.reunitarise_all();
                landau_gauge_fix(copy, /*tol*/1e-9, prop_max_iter);
                Gprop = static_propagator(copy, r_max);
                for (int rr = 0; rr <= r_max; ++rr) Gprop_sum[rr] += Gprop[rr];
                ++g_n;
            }

            if (csv) {
                csv << r.sweep << ',' << r.L << ',' << r.absL << ','
                    << r.L2   << ',' << r.L4 << ',' << r.plaq;
                if (!no_wilson) {
                    for (double v : Pcorr)  csv << ',' << v;
                    for (double v : W_flat) csv << ',' << v;
                }
                if (prop_enabled) {
                    if (measure_prop)
                        for (double v : Gprop) csv << ',' << v;
                    else
                        for (int rr = 0; rr <= r_max; ++rr) csv << ",NaN";
                }
                csv << '\n';
            }
            ++meas_index;
        }
        if ((s + 1) % log_every_meas == 0 || s + 1 == cfg.n_meas)
            log_line("meas", s + 1);
    }
    if (csv) csv.close();

    // ---- Summary ----
    std::cout << "#\n# === Run summary ===\n";
    std::printf("# beta              = %.4f\n",  cfg.beta);
    std::printf("# measurements      = %lld\n",  acc.n());
    std::printf("# acceptance rate   = %.4f\n",  upd.stats().acceptance_rate());
    std::printf("# <plaq>            = %.6f\n",  acc.mean_plaq());
    std::printf("# <L>               = %+.6f\n", acc.mean_L());
    std::printf("# <|L|>             = %.6f\n",  acc.mean_absL());
    std::printf("# <L^2>             = %.6f\n",  acc.mean_L2());
    std::printf("# <L^4>             = %.6f\n",  acc.mean_L4());
    std::printf("# Binder U4         = %.6f   (0 symmetric, 2/3 broken)\n",
                acc.binder_U4());
    std::printf("# susceptibility    = %.6f   (Vs=%d)\n",
                acc.susceptibility(), Vs);

    if (!no_wilson && w_n > 0) {
        std::cout << "#\n# Polyakov pair correlator (averaged over "
                  << w_n << " measurements):\n";
        for (int rr = 0; rr <= r_max; ++rr)
            std::printf("#   P(r=%d) = %+.6f\n", rr,
                        Pcorr_sum[rr] / static_cast<double>(w_n));

        std::cout << "#\n# Mean Wilson loops W(R, T) (averaged):\n";
        std::cout << "#         T=";
        for (int T = 1; T <= T_max; ++T) std::printf(" %8d  ", T);
        std::cout << "\n";
        for (int R = 1; R <= R_max; ++R) {
            std::printf("#   R=%2d  ", R);
            for (int T = 1; T <= T_max; ++T)
                std::printf(" %8.5f  ", W_sum[R][T] / static_cast<double>(w_n));
            std::cout << "\n";
        }
    }

    if (prop_enabled && g_n > 0) {
        std::cout << "#\n# Boson propagator G(r) (Landau gauge, averaged over "
                  << g_n << " measurements):\n";
        for (int rr = 0; rr <= r_max; ++rr)
            std::printf("#   G(r=%d) = %.6e\n", rr,
                        Gprop_sum[rr] / static_cast<double>(g_n));
    }

    if (!csv_path.empty())
        std::cout << "# CSV written to    : " << csv_path << "\n";
    return 0;
}
