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
 * Simulate time evolution for a Potts model with given parameters.
 *
 * @param T Total simulation time.
 * @param dt Time step.
 * @param num_spins Number of spins in the system.
 * @param alpha_rate Rate parameter for alpha evolution.
 * @param gamma Gamma parameter (coupling to conjugate term).
 * @param r_target Target radius parameter.
 * @param connection_list Adjacency list of spin connections (length num_spins, each a list of neighbor indices).
 * @param initial_alpha_list Initial alpha values (real) for each spin.
 * @param num_states Number of spin states in the Potts model.
 * @param noise_factor Scaling factor for the stochastic noise.
 * @return A NumPy array of shape (num_steps, num_spins) with dtype complex128 containing the complex amplitudes at each time step.
 */
py::array_t<std::complex<double>> simulate(double T, double dt, int num_spins,
                                             double alpha_rate, double gamma, double r_target,
                                             const std::vector<std::vector<int>>& connection_list,
                                             const std::vector<double>& initial_alpha_list,
                                             int num_states, double noise_factor) {
    // Validate input sizes
    if (connection_list.size() != static_cast<size_t>(num_spins)) {
        throw std::invalid_argument("connection_list length must equal num_spins");
    }
    if (initial_alpha_list.size() != static_cast<size_t>(num_spins)) {
        throw std::invalid_argument("initial_alpha_list length must equal num_spins");
    }
    if (T < 0 || dt <= 0) {
        throw std::invalid_argument("T must be non-negative and dt must be positive");
    }
    // Determine number of integration steps (using floor of T/dt)
    int steps = static_cast<int>(T / dt);
    if (steps < 0) {
        steps = 0;
    }
    // Prepare output NumPy array of shape (steps, num_spins) with complex128 dtype
    py::array_t<std::complex<double>> result({ steps, num_spins });
    // If no steps to simulate, return empty result
    if (steps == 0) {
        return result;
    }

    // Initialize state arrays
    std::vector<std::complex<double>> x(num_spins, std::complex<double>(0.0, 0.0));
    std::vector<double> alpha_list = initial_alpha_list;

    std::vector<std::complex<double>> coupling_term(num_spins);
    std::vector<double> abs_x(num_spins);
    std::vector<double> abs_x_sq(num_spins);
    std::vector<std::complex<double>> conj_x(num_spins);
    std::vector<std::complex<double>> noise_values(num_spins);

    // Random number generator for Gaussian noise
    std::mt19937 gen(1);  // fixed seed for reproducibility
    std::normal_distribution<double> dist(0.0, 1.0);
    double noise_scale = noise_factor * std::sqrt(dt / 2.0);

    // Pointer to output array data
    std::complex<double>* result_ptr = result.mutable_data();

    // Main simulation loop over time steps
    for (int step = 0; step < steps; ++step) {
        // Clear coupling terms for this step
        std::fill(coupling_term.begin(), coupling_term.end(), std::complex<double>(0.0, 0.0));

        // Pre-compute magnitude, squared magnitude, conjugate of x, and generate noise for each spin
        for (int i = 0; i < num_spins; ++i) {
            double mag = std::abs(x[i]);
            abs_x[i] = mag;
            abs_x_sq[i] = mag * mag;
            conj_x[i] = std::conj(x[i]);
            noise_values[i] = std::complex<double>(dist(gen), dist(gen)) * noise_scale;
        }

        // Update alpha for each spin
        double sqrt_r_target = std::sqrt(r_target);
        for (int i = 0; i < num_spins; ++i) {
            // d(alpha)/dt = alpha_rate * (sqrt(r_target) - sqrt(|x_i|))
            alpha_list[i] += dt * alpha_rate * (sqrt_r_target - std::sqrt(abs_x[i]));
        }

        // Compute coupling term: sum of neighbor contributions
        for (int i = 0; i < num_spins; ++i) {
            const std::vector<int>& neighbors = connection_list[i];
            for (int j_index = 0; j_index < static_cast<int>(neighbors.size()); ++j_index) {
                int j = neighbors[j_index];
                coupling_term[i] -= x[j];
            }
        }

        // Update each spin's complex amplitude
        for (int i = 0; i < num_spins; ++i) {
            // Compute conj(x[i])^(num_states - 1)
            std::complex<double> conj_power(1.0, 0.0);
            for (int p = 0; p < num_states - 1; ++p) {
                conj_power *= conj_x[i];
            }
            std::complex<double> derivative = alpha_list[i] * x[i]
                                              - abs_x_sq[i] * x[i]
                                              + gamma * conj_power
                                              + coupling_term[i];
            x[i] += derivative * dt + noise_values[i];
        }

        // Store state
        std::memcpy(result_ptr + static_cast<size_t>(step) * num_spins,
                    x.data(),
                    num_spins * sizeof(std::complex<double>));
    }

    return result;
}

// Pybind11 module definition
PYBIND11_MODULE(potts_sim, m) {
    m.doc() = "Potts model simulation module (single-threaded)";
    m.def("simulate", &simulate,
          py::arg("T"), py::arg("dt"), py::arg("num_spins"),
          py::arg("alpha_rate"), py::arg("gamma"), py::arg("r_target"),
          py::arg("connection_list"), py::arg("initial_alpha_list"),
          py::arg("num_states"), py::arg("noise_factor"),
          "Simulate the Potts model dynamics and return complex amplitudes at each time step");
}
