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
 * Discrete time evolution of a q-state Potts machine model.
 *
 * @param T                         Total simulation time.
 * @param dt                        Time step.
 * @param num_spins                 Number of spins in the system.
 * @param num_states                Number of states (q) for the Potts model.
 * @param alpha_rate                Rate parameter for alpha evolution.
 * @param gamma                     Gamma parameter (coupling to conj(x)^2).
 * @param r_target                  Target amplitude for alpha evolution. 
 * @param connections               Array of shape (2, num_connections) containing the source and target
 *                                  indices for the connections in the system (bidirectional pairs).
 * @param initial_alpha_arr         Initial alpha values for each spin.
 * @param noise_factor              Scaling factor for stochastic noise.
 * @param seed                      Random seed for noise generation.
 * @param return_continuous_states  If true, return a history of complex spin amplitudes.
 * @param return_discrete_states    If true, return a history of rounded Potts states.
 * @param return_energy             If true, return a history of the system energy.
 * @return                          A dictionary with keys "continuous_states", "discrete_states", and "energy",
 *                                  each mapped to the corresponding array or None if not requested.
 *                                  The arrays are of shape (num_steps, num_spins) for continuous and discrete states,
 *                                  and (num_steps,) for energy.
 */
py::object run(
    double T, double dt, int num_spins, int num_states,
    double alpha_rate, double gamma, double r_target,
    const py::array_t<int>& connections,
    const std::vector<double>& initial_alpha_arr,
    double noise_factor, int seed = 1,
    bool return_continuous_states = true,
    bool return_discrete_states = false,
    bool return_energy = false)
{

    //----------------------------------------------------------------
    // Validate and process input parameters
    //----------------------------------------------------------------

    // Extract source and target indices from the connections array assuming shape (2, num_connections)
    auto conn = connections.unchecked<2>();
    size_t num_connections = conn.shape(1);
    if (conn.shape(0) != 2) {
        throw std::invalid_argument("connections array must have shape (2, num_connections)");
    }
    std::vector<int> sources(num_connections);
    std::vector<int> targets(num_connections);
    for (size_t i = 0; i < num_connections; ++i) {
        sources[i] = conn(0, i);
        targets[i] = conn(1, i);
    }

    double sqrt_r_target = std::sqrt(r_target); // sqrt of target amplitude
    
    if (initial_alpha_arr.size() != static_cast<size_t>(num_spins)) {
        throw std::invalid_argument("initial_alpha_arr length must equal num_spins");
    }
    
    if (T < 0.0 || dt <= 0.0) {
        throw std::invalid_argument("T must be non-negative and dt must be positive");
    }
    
    // Calculate number of num_steps
    int num_steps = static_cast<int>(std::floor(T / dt));
    if (num_steps < 1) {
        throw std::invalid_argument("Number of steps must be at least 1");
    }
    
    //----------------------------------------------------------------
    // Allocate arrays for spin states, coupling terms, noise, etc.
    //----------------------------------------------------------------
    
    std::vector<std::complex<double>> x(num_spins, {0.0, 0.0}); // initialize spin states to zero
    std::vector<double> alpha_arr = initial_alpha_arr; // copy initial alpha values
    
    std::complex<double> ref_state_conj[num_states]; // reference state conjugates
    for (int j = 0; j < num_states; ++j) {
        double theta = -M_PI + (2 * j + 1) * M_PI / num_states; // reference angle
        ref_state_conj[j] = std::polar(1.0, -theta); // reference state vector
    }

    // Allocate partial results for update rule
    std::vector<std::complex<double>> coupling_arr(num_spins); 
    std::vector<double> abs_x(num_spins);
    std::vector<double> abs_x_sqrt(num_spins);
    std::vector<double> abs_x_sq(num_spins);
    std::vector<std::complex<double>> conj_x(num_spins);
    std::vector<std::complex<double>> conj_pow(num_spins);
    std::vector<std::complex<double>> noise_arr(num_spins);
    std::vector<int> rounded_state(num_spins); // to store the rounded states
    double energy; // energy of the system (standard Potts Hamiltonian)

    // Initialize random number generator for Gaussian noise
    std::mt19937 gen(seed);  // fixed seed
    std::normal_distribution<double> dist(0.0, 1.0); // mean=0, stddev=1
    double noise_scale = noise_factor * std::sqrt(dt / 2.0); // account for dt scaling

    // Allocate continuous state (complex amplitudes) history if requested
    py::array_t<std::complex<double>> continuous_state_history;
    std::complex<double>* ptr_cont = nullptr;
    if (return_continuous_states) {
        continuous_state_history = py::array_t<std::complex<double>>(
            { size_t(num_steps), size_t(num_spins) }
        );
        auto bi = continuous_state_history.request();
        ptr_cont = static_cast<std::complex<double>*>(bi.ptr);
    }

    // Allocate discrete state (rounded Potts state) history if requested
    py::array_t<int> discrete_state_history;
    int* ptr_disc = nullptr;
    if (return_discrete_states) {
        discrete_state_history = py::array_t<int>(
            { size_t(num_steps), size_t(num_spins) }
        );
        auto bi = discrete_state_history.request();
        ptr_disc = static_cast<int*>(bi.ptr);
    }

    // Allocate energy (Hamiltonian) history if requested
    py::array_t<double> energy_history;
    double* ptr_e = nullptr;
    if (return_energy) {
        energy_history = py::array_t<double>(num_steps);
        auto bi = energy_history.request();
        ptr_e = static_cast<double*>(bi.ptr);
    }
 
    //----------------------------------------------------------------
    // Time-stepping loop
    //----------------------------------------------------------------

    for (int step = 0; step < num_steps; ++step) {

        // Compute spin-specific values
        for (int i = 0; i < num_spins; ++i) {
            // Compute magnitudes and conjugate
            double magnitude = std::abs(x[i]);
            abs_x[i] = magnitude; // |x_i|
            abs_x_sqrt[i] = std::sqrt(magnitude); // sqrt(|x_i|)
            abs_x_sq[i] = magnitude * magnitude; // |x_i|^2
            conj_x[i] = std::conj(x[i]); // conjugate of x_i
            conj_pow[i] = std::pow(conj_x[i], num_states - 1); // conj(x_i)^(q-1)
            
            // Generate noise values
            double nr = dist(gen);
            double ni = dist(gen);
            noise_arr[i] = std::complex<double>(nr, ni) * noise_scale;

            // Update alpha: 
            // alpha(t+dt) = alpha(t) + dt * alpha_rate * (sqrt(r_target) - sqrt(|x|))
            alpha_arr[i] += dt * alpha_rate * (sqrt_r_target - abs_x_sqrt[i]);
        }

        // Compute coupling between spins
        //      coupling[i] = sum_j (J_ij x[j]) for all j connected to i,
        //      where we use J_ij = -1 for antiferromagnetic coupling
        std::fill(coupling_arr.begin(), coupling_arr.end(), std::complex<double>(0.0, 0.0)); // reset coupling term
        for (size_t conn_idx = 0; conn_idx < sources.size(); ++conn_idx) {
            int s_idx = sources[conn_idx];
            int t_idx = targets[conn_idx];
            coupling_arr[s_idx] -= x[t_idx]; 
            coupling_arr[t_idx] -= x[s_idx];
        }

        // Update spin states using the update rule:
        //      x(t+dt) = x(t) + dt * (alpha * x - |x|^2 * x + gamma * conj(x)^(q-1) + coupling) + noise
        for (int i = 0; i < num_spins; ++i) {
            x[i] += dt * (alpha_arr[i] * x[i] - abs_x_sq[i] * x[i] + gamma * conj_pow[i] + coupling_arr[i]) + noise_arr[i];
        }

        // Round to nearest Potts state
        for (int i = 0; i < num_spins; ++i) {
            double max_dot = -1e300;
            int closest_state = 0;
            for (int j = 0; j < num_states; ++j) {
                double dot = std::real(ref_state_conj[j] * x[i]); // dot product with reference state
                if (dot > max_dot) {
                    max_dot = dot;
                    closest_state = j;
                }
            }
            rounded_state[i] = closest_state; // Store the rounded state
        }

        // Calculate energy of rounded states using the standard Potts Hamiltonian
        //      H = -sum_i<j (J_ij * delta(s_i, s_j) for all connected pairs (i, j)
        //      where delta is the Kronecker delta function and J_ij = -1
        energy = 0.0; // reset energy
        for (size_t conn_idx = 0; conn_idx < sources.size(); ++conn_idx) {
            int s_idx = sources[conn_idx];
            int t_idx = targets[conn_idx];
            if (rounded_state[s_idx] == rounded_state[t_idx]) {
                energy += 1.0; // add to energy for same state
            }
        }

        // Store states and energy if requested
        if (return_continuous_states) {
            std::memcpy(ptr_cont + static_cast<size_t>(step) * num_spins,
                x.data(),
                static_cast<size_t>(num_spins) * sizeof(std::complex<double>));
        }
        if (return_discrete_states) {
            std::memcpy(ptr_disc + step * num_spins, 
                rounded_state.data(), 
                num_spins * sizeof(int));
        }
        if (return_energy) ptr_e[step] = energy;
    }

    // Return results as dictionary
    py::dict out;
    out["continuous_states"] = return_continuous_states ? py::object(continuous_state_history) : py::none();
    out["discrete_states"] = return_discrete_states ? py::object(discrete_state_history) : py::none();
    out["energy"] = return_energy ? py::object(energy_history) : py::none();
    return out;
    
}

//-----------------------------------------------------------
// Pybind11 module definition
//-----------------------------------------------------------
PYBIND11_MODULE(potts_sim, m) {
    m.doc() = R"pbdoc(
        Discrete time evolution of a q-state Potts machine model.
    )pbdoc";

    m.def("run", &run,
          py::arg("T"),
          py::arg("dt"),
          py::arg("num_spins"),
          py::arg("num_states"),
          py::arg("alpha_rate"),
          py::arg("gamma"),
          py::arg("r_target"),
          py::arg("connections"),
          py::arg("initial_alpha_arr"),
          py::arg("noise_factor"),
          py::arg("seed") = 1,
          py::arg("return_continuous_states") = true,
          py::arg("return_discrete_states") = false,
          py::arg("return_energy") = false,
          R"pbdoc(
              Run q-state Potts model from t=0 to t=T in increments of dt.
              
              Parameters
              ----------
              T : float
                  Total simulation time.
              dt : float
                  Time step.
              num_spins : int
                  Number of spins in the system.
              num_states : int
                  Number of Potts states (q).
              alpha_rate : float
                  Rate parameter for alpha evolution.
              gamma : float
                  Coupling parameter.
              r_target : float
                  Target amplitude for alpha evolution.
              connections : array_like, shape (2, num_connections)
                  Source and target indices for spin couplings.
              initial_alpha_arr : sequence of float, length num_spins
                  Initial alpha values.
              noise_factor : float
                  Scaling factor for Gaussian noise.
              seed : int, optional
                  Random seed (default: 1).
              return_continuous_states : bool, optional
                  If True, include complex state history (default: True).
              return_discrete_states : bool, optional
                  If True, include discrete state history (default: False).
              return_energy : bool, optional
                  If True, include energy history (default: False).
              
              Returns
              -------
              dict
                  Dictionary with keys "continuous_states", "discrete_states", and "energy".
          )pbdoc");
}
