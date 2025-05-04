#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstring>

// Expose everything in the pybind11 namespace as "py"
namespace py = pybind11;

/**
 * Simulate time evolution for a 3-state Potts model, returning a full trajectory
 * in a (steps, num_spins) array of complex128.
 *
 * @param T               Total simulation time.
 * @param dt              Time step.
 * @param num_spins       Number of spins in the system.
 * @param alpha_rate      Rate parameter for alpha evolution.
 * @param gamma           Gamma parameter (coupling to conj(x)^2).
 * @param r_target        Target radius parameter.
 * @param connection_list Adjacency list of spin connections (outer size = num_spins).
 * @param initial_alpha_list Initial alpha values for each spin.
 * @param noise_factor    Scaling factor for stochastic noise.
 * @return A NumPy array of shape (steps, num_spins) with dtype complex128,
 *         containing the spin amplitudes at each time step.
 */
py::array_t<std::complex<double>> simulate(
    double T, double dt, int num_spins,
    double alpha_rate, double gamma, double r_target,
    const std::vector<std::vector<int>>& connection_list,
    const std::vector<double>& initial_alpha_list,
    double noise_factor)
{
    // Validate input
    if (connection_list.size() != static_cast<size_t>(num_spins)) {
        throw std::invalid_argument("connection_list length must equal num_spins");
    }
    if (initial_alpha_list.size() != static_cast<size_t>(num_spins)) {
        throw std::invalid_argument("initial_alpha_list length must equal num_spins");
    }
    if (T < 0.0 || dt <= 0.0) {
        throw std::invalid_argument("T must be non-negative and dt must be positive");
    }

    // Number of steps
    int steps = static_cast<int>(T / dt);
    if (steps < 1) {
        // Return an empty array or single row if T/dt < 1
        // For consistency, let's return (0, num_spins).
        return py::array_t<std::complex<double>>(
            py::array::ShapeContainer({0, num_spins})
        );
    }

    //----------------------------------------------------------------
    // Flatten adjacency for more cache-friendly iteration over edges
    //----------------------------------------------------------------
    std::vector<int> sources;
    std::vector<int> targets;
    size_t total_neighbors = 0;
    for (int i = 0; i < num_spins; ++i) {
        total_neighbors += connection_list[i].size();
    }
    sources.reserve(total_neighbors);
    targets.reserve(total_neighbors);

    for (int i = 0; i < num_spins; ++i) {
        for (auto nbr : connection_list[i]) {
            sources.push_back(i);
            targets.push_back(nbr);
        }
    }

    //----------------------------------------------------------------
    // Allocate data structures
    //----------------------------------------------------------------
    // We'll store spin amplitudes in a vector of length num_spins.
    // Start at 0.0, or if you have a different initialization, adapt here.
    std::vector<std::complex<double>> x(num_spins, {0.0, 0.0});

    // alpha values from user
    std::vector<double> alpha_list = initial_alpha_list;

    // Temporary arrays used each iteration
    std::vector<std::complex<double>> coupling_term(num_spins);
    std::vector<double> abs_x(num_spins);
    std::vector<double> abs_x_sq(num_spins);
    std::vector<std::complex<double>> conj_x(num_spins);
    std::vector<std::complex<double>> noise_values(num_spins);

    // Random number generator for Gaussian noise
    std::mt19937 gen(1);  // fixed seed
    std::normal_distribution<double> dist(0.0, 1.0);
    double noise_scale = noise_factor * std::sqrt(dt / 2.0);

    // We'll need to store (steps, num_spins) states in a NumPy array
    // shape = (steps, num_spins), dtype=complex128
    py::array_t<std::complex<double>> result(
        py::array::ShapeContainer({static_cast<size_t>(steps), 
                                  static_cast<size_t>(num_spins)})
    );
    // Get raw pointer access to the result buffer
    auto buf_info = result.request();
    auto result_ptr = static_cast<std::complex<double>*>(buf_info.ptr);

    //----------------------------------------------------------------
    // Time-stepping loop
    //----------------------------------------------------------------
    double sqrt_r_target = std::sqrt(r_target);

    for (int step = 0; step < steps; ++step) {
        // 1) Clear coupling_term
        std::fill(coupling_term.begin(), coupling_term.end(), std::complex<double>(0.0, 0.0));

        // 2) Precompute magnitudes, conj(x), noise
        for (int i = 0; i < num_spins; ++i) {
            double magnitude = std::abs(x[i]);
            abs_x[i] = magnitude;
            abs_x_sq[i] = magnitude * magnitude;
            conj_x[i] = std::conj(x[i]);
            // Noise
            double nr = dist(gen);
            double ni = dist(gen);
            noise_values[i] = std::complex<double>(nr, ni) * noise_scale;
        }

        // 3) Update alpha
        for (int i = 0; i < num_spins; ++i) {
            double delta_alpha = alpha_rate * (sqrt_r_target - std::sqrt(abs_x[i]));
            alpha_list[i] += dt * delta_alpha;
        }

        // 4) Compute coupling
        for (size_t e = 0; e < sources.size(); ++e) {
            int idx = sources[e];
            int nbr = targets[e];
            coupling_term[idx] -= x[nbr];  // negative sign or your chosen sign convention
        }

        // 5) Update x
        // For 3-state model, conj(x)^2 = conj_x[i] * conj_x[i]
        for (int i = 0; i < num_spins; ++i) {
            // conj^2
            std::complex<double> conj_sq = conj_x[i] * conj_x[i];

            // derivative = alpha * x - |x|^2 * x + gamma * conj^2 + coupling_term
            std::complex<double> derivative =
                alpha_list[i] * x[i]
                - abs_x_sq[i] * x[i]
                + gamma * conj_sq
                + coupling_term[i];

            x[i] += derivative * dt + noise_values[i];
        }

        // 6) Store this step's data into the output array
        //    row index = step, so offset is step * num_spins
        std::memcpy(result_ptr + static_cast<size_t>(step) * num_spins,
                    x.data(),
                    static_cast<size_t>(num_spins) * sizeof(std::complex<double>));
    }

    // Return the full trajectory array
    return result;
}

//-----------------------------------------------------------
// Pybind11 module definition
//-----------------------------------------------------------
PYBIND11_MODULE(potts_sim, m) {
    m.doc() = R"pbdoc(
        Single-threaded 3-state Potts simulation returning full trajectory.
        Uses flattened adjacency and conj(x)^2 unrolling for speed.
    )pbdoc";

    m.def("simulate", &simulate,
          py::arg("T"),
          py::arg("dt"),
          py::arg("num_spins"),
          py::arg("alpha_rate"),
          py::arg("gamma"),
          py::arg("r_target"),
          py::arg("connection_list"),
          py::arg("initial_alpha_list"),
          py::arg("noise_factor"),
          R"pbdoc(
              Simulate the 3-state Potts model from t=0 to t=T in increments of dt.
              Returns a (steps, num_spins) array of complex128 with the spin amplitudes
              at each time step.

              steps = floor(T/dt).
          )pbdoc");
}
