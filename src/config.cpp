#include "su2lgt/config.hpp"

#include <ostream>
#include <sstream>
#include <iomanip>

namespace su2lgt {

void RunConfig::print(std::ostream& os) const {
    os << "# RunConfig:\n"
       << "#   geom    Lx=" << Lx << " Ly=" << Ly
       << " Lz=" << Lz << " Lt=" << Lt << "\n"
       << "#   beta    " << beta << "\n"
       << "#   sweeps  n_therm=" << n_therm
       << " n_meas=" << n_meas
       << " meas_every=" << meas_every << "\n"
       << "#   updater prop_eps=" << prop_eps
       << " n_hits=" << n_hits << "\n"
       << "#   seed    0x" << std::hex << seed << std::dec << "\n"
       << "#   out_dir " << out_dir
       << "  label=" << (label.empty() ? "(auto)" : label) << "\n";
}

std::string RunConfig::auto_filename() const {
    std::ostringstream oss;
    oss << "Ls" << Lx
        << "_Lt" << Lt
        << "_beta" << std::fixed << std::setprecision(4) << beta;
    if (!label.empty()) oss << "_" << label;
    oss << ".csv";
    return oss.str();
}

}  // namespace su2lgt
