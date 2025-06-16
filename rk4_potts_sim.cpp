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

// Enumeration for Potts model types
enum class ModelType {
    POLYNOMIAL,
    NEC,
    SIGMOID,
    FIXED_AMPLITUDE
};

// Helper functions for RK4 integration

// Compute coupling for all spins based on the current state
void compute_coupling(
    const std::vector<std::complex<double>>& x,
    std::vector<std::complex<double>>& coupling_arr,
    const std::vector<int>& sources,
    const std::vector<int>& targets) 
{
    std::fill(coupling_arr.begin(), coupling_arr.end(), std::complex<double>(0.0, 0.0));
    for (size_t conn_idx = 0; conn_idx < sources.size(); ++conn_idx) {
        int s_idx = sources[conn_idx];
        int t_idx = targets[conn_idx];
        coupling_arr[s_idx] -= x[t_idx]; 
        coupling_arr[t_idx] -= x[s_idx];
    }
}

// Compute derivatives for POLYNOMIAL model
void compute_polynomial_derivative(
    const std::vector<std::complex<double>>& x,
    const std::vector<std::complex<double>>& coupling_arr,
    std::vector<std::complex<double>>& dx_dt,
    int num_states,
    int polynomial_order,
    double beta,
    double gamma)
{
    for (size_t i = 0; i < x.size(); ++i) {
        double abs_x = std::abs(x[i]);
        double abs_x_pow = std::pow(abs_x, polynomial_order - 1);
        std::complex<double> conj_x = std::conj(x[i]);
        std::complex<double> conj_pow = std::pow(conj_x, num_states - 1);
        
        dx_dt[i] = -x[i] - abs_x_pow * x[i] + gamma * conj_pow + beta * coupling_arr[i];
    }
}

// Compute derivatives for NEC model
void compute_nec_derivative(
    const std::vector<std::complex<double>>& x,
    const std::vector<std::complex<double>>& coupling_arr,
    std::vector<std::complex<double>>& dx_dt,
    int num_states,
    const std::vector<double>& alpha_arr,
    double gamma)
{
    for (size_t i = 0; i < x.size(); ++i) {
        double abs_x = std::abs(x[i]);
        double abs_x_sq = abs_x * abs_x;
        std::complex<double> conj_x = std::conj(x[i]);
        std::complex<double> conj_pow = std::pow(conj_x, num_states - 1);
        
        dx_dt[i] = alpha_arr[i] * x[i] - abs_x_sq * x[i] + gamma * conj_pow + coupling_arr[i];
    }
}

// Compute derivatives for SIGMOID model
void compute_sigmoid_derivative(
    const std::vector<std::complex<double>>& x,
    const std::vector<std::complex<double>>& coupling_arr,
    std::vector<std::complex<double>>& dx_dt,
    int num_states,
    double alpha,
    double beta,
    double gamma)
{
    for (size_t i = 0; i < x.size(); ++i) {
        double abs_x = std::abs(x[i]);
        double abs_x_pow1 = std::pow(abs_x, num_states - 1);
        double abs_x_pow2 = std::pow(abs_x, num_states - 2);
        double theta = std::arg(x[i]);
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        std::complex<double> rotated_coupling = coupling_arr[i] * std::polar(1.0, -theta);
        
        double dr_dt = -abs_x + std::tanh(
                      alpha * abs_x
                      + gamma * abs_x_pow1 * std::cos(num_states * theta)
                      + beta * std::real(rotated_coupling));
                      
        double dtheta_dt = -gamma * abs_x_pow2 * std::sin(num_states * theta)
                          + beta / (abs_x + 1e-10) * std::imag(rotated_coupling);
                          
        dx_dt[i] = std::complex<double>(
                  dr_dt * cos_theta - abs_x * dtheta_dt * sin_theta,
                  dr_dt * sin_theta + abs_x * dtheta_dt * cos_theta);
    }
}

// Compute derivatives for FIXED_AMPLITUDE model
void compute_fixed_amplitude_derivative(
    const std::vector<std::complex<double>>& x,
    const std::vector<std::complex<double>>& coupling_arr,
    std::vector<double>& dtheta_dt,
    int num_states,
    double gamma)
{
    for (size_t i = 0; i < x.size(); ++i) {
        double theta = std::arg(x[i]);
        double sin_q_theta = std::sin(num_states * theta);
        double phase_coupling = std::imag(coupling_arr[i] * std::polar(1.0, -theta));
        
        dtheta_dt[i] = -gamma * sin_q_theta + phase_coupling;
    }
}

/**
 * Discrete time evolution of a q‐state Potts machine model.
 *
 * @param T                       Total simulation time.
 * @param dt                      Time step.
 * @param num_spins               Number of spins in the system.
 * @param num_states              Number of Potts states (q).
 * @param edges                   Array of shape (2, num_edges) containing source and target indices.
 * @param noise_factor            Scaling factor for Gaussian noise.
 * @param seed                    Random seed for noise generation.
 * @param model_type              Variant of Potts model to run (POLYNOMIAL, NEC, SIGMOID, FIXED_AMPLITUDE).
 * @param polynomial_order        Order of polynomial nonlinearity (POLYNOMIAL).
 * @param alpha                   Alpha parameter (SIGMOID).
 * @param alpha_rate              Rate parameter (NEC).
 * @param r_target                Target amplitude (NEC).
 * @param initial_alpha_arr       Initial alpha values per spin (NEC).
 * @param beta_schedule           Time‐dependent beta (POLYNOMIAL, SIGMOID).
 * @param gamma_schedule          Time‐dependent gamma (POLYNOMIAL, NEC, SIGMOID, FIXED_AMPLITUDE).
 * @param return_continuous_states If true, include complex‐state history.
 * @param return_discrete_states  If true, include discrete‐state history.
 * @param return_energy           If true, include energy history.
 * @param return_cut_value        If true, include cut value history.
 * @param return_best_only        If true, only return states and values for best solution found.
 * @return                        A dict with keys "continuous_states", "discrete_states", "energy", "cut_value", and "step".
 */
py::object run(
    double T, double dt, int num_spins, int num_states,
    const py::array_t<int>& edges,
    double noise_factor, int seed,
    ModelType model_type,
    int polynomial_order,
    double alpha,
    double alpha_rate,
    double r_target,
    const std::vector<double>& initial_alpha_arr,
    const std::vector<double>& beta_schedule,
    const std::vector<double>& gamma_schedule,
    bool return_continuous_states,
    bool return_discrete_states,
    bool return_energy,
    bool return_cut_value,
    bool return_best_only
)
{

    //----------------------------------------------------------------
    // Process and validate input parameters
    //----------------------------------------------------------------

    // FOR ALL MODELS

    // Extract source and target indices from the edges array assuming shape (2, num_edges)
    auto conn = edges.unchecked<2>();
    size_t num_edges = conn.shape(1);
    if (conn.shape(0) != 2) {
        throw std::invalid_argument("edges array must have shape (2, num_edges)");
    }
    std::vector<int> sources(num_edges);
    std::vector<int> targets(num_edges);
    for (size_t i = 0; i < num_edges; ++i) {
        sources[i] = conn(0, i);
        targets[i] = conn(1, i);
    }

    if (T < 0.0 || dt <= 0.0) {
        throw std::invalid_argument("T must be non-negative and dt must be positive");
    }
    
    // Calculate number of num_steps
    int num_steps = static_cast<int>(std::floor(T / dt));
    if (num_steps < 1) {
        throw std::invalid_argument("Number of steps must be at least 1");
    }

    // MODEL-SPECIFIC VALIDATIONS

    // For NEC, check initial_alpha_arr length and assign it to alpha_arr
    std::vector<double> alpha_arr;
    if (model_type == ModelType::NEC) {
        if (initial_alpha_arr.size() != static_cast<size_t>(num_spins)) {
            throw std::invalid_argument("initial_alpha_arr length must equal num_spins");
        }
        alpha_arr = initial_alpha_arr; // use initial alpha array as the const/evolving alpha array
    }

    // For POLYNOMIAL and SIGMOID, check beta_schedule length
    if (model_type == ModelType::POLYNOMIAL || model_type == ModelType::SIGMOID) {
        if (beta_schedule.size() != static_cast<size_t>(num_steps)) {
            throw std::invalid_argument("beta_schedule length must equal num_steps");
        }
    }

    // For POLYNOMIAL, NEC, SIGMOID, and FIXED_AMPLITUDE, check gamma_schedule length
    if (model_type == ModelType::POLYNOMIAL || 
        model_type == ModelType::SIGMOID || 
        model_type == ModelType::NEC ||
        model_type == ModelType::FIXED_AMPLITUDE) {
        if (gamma_schedule.size() != static_cast<size_t>(num_steps)) {
            throw std::invalid_argument("gamma_schedule length must equal num_steps");
        }
    }

    // For NEC, check r_target and set sqrt_r_target
    double sqrt_r_target = 0.0;
    if (model_type == ModelType::NEC) {
        if (r_target < 0.0) {
            throw std::invalid_argument("r_target must be non-negative for NEC model");
        }
        sqrt_r_target = std::sqrt(r_target);
    }
    
    //----------------------------------------------------------------
    // Allocate arrays for spin states, coupling terms, noise, etc.
    //----------------------------------------------------------------
    
    // Allocate arrays for spin states, coupling terms, noise, etc.
    std::vector<std::complex<double>> x(num_spins, {0.0, 0.0}); // initialize spin states to zero
    std::vector<std::complex<double>> coupling_arr(num_spins); 
    std::vector<int> rounded_state(num_spins); // to store the rounded states
    
    // Reference state vectors for rounding
    //      ref_state_conj[j] = exp(-i * theta_j) for j=0,1,...,q-1
    //      where theta_j = 2*pi/q * (j + floor(q/2))
    std::vector<std::complex<double>> ref_state_conj(num_states);
    for (int j = 0; j < num_states; ++j) {
        double theta_j = 2.0 * M_PI / num_states * (j + std::floor(num_states / 2.0));
        ref_state_conj[j] = std::polar(1.0, -theta_j);
    }

    // Allocate partial results for update rules
    std::complex<double> noise;
    double phase_noise;
    double abs_x;
    double abs_x_sqrt;
    double energy = 0.0;
    double cut_value = 0.0;

    // Initialize random number generator for Gaussian noise
    std::mt19937 gen(seed);  // fixed seed
    std::normal_distribution<double> dist(0.0, 1.0); // mean=0, stddev=1
    double noise_scale = noise_factor * std::sqrt(dt / 2.0); // account for dt scaling

    // Variables for tracking best solution
    int best_step = 0;
    double best_energy = std::numeric_limits<double>::infinity();
    double best_cut_value = -std::numeric_limits<double>::infinity();
    std::vector<std::complex<double>> best_continuous_state(num_spins);
    std::vector<int> best_discrete_state(num_spins);

    // For FIXED_AMPLITUDE model only, initialize x with amplitude 1 and random phase
    if (model_type == ModelType::FIXED_AMPLITUDE) {
        for (int i = 0; i < num_spins; ++i) {
            double theta = dist(gen) * M_PI; // random phase
            x[i] = std::polar(1.0, theta); // set amplitude to 1 and random phase
        }
    }

    // Allocate arrays for RK4 integration
    std::vector<std::complex<double>> k1(num_spins);
    std::vector<std::complex<double>> k2(num_spins);
    std::vector<std::complex<double>> k3(num_spins);
    std::vector<std::complex<double>> k4(num_spins);
    std::vector<std::complex<double>> x_temp(num_spins);
    std::vector<std::complex<double>> coupling_temp(num_spins);
    std::vector<double> dtheta_dt_vec(num_spins); // For FIXED_AMPLITUDE model

    // Allocate continuous state (complex amplitudes) history if requested
    py::array_t<std::complex<double>> continuous_state_history;
    std::complex<double>* ptr_continuous_states = nullptr;
    if (return_continuous_states && !return_best_only) {
        continuous_state_history = py::array_t<std::complex<double>>(
            { size_t(num_steps), size_t(num_spins) }
        );
        auto bi = continuous_state_history.request();
        ptr_continuous_states = static_cast<std::complex<double>*>(bi.ptr);
    }

    // Allocate discrete state (rounded Potts state) history if requested
    py::array_t<int> discrete_state_history;
    int* ptr_discrete_states = nullptr;
    if (return_discrete_states && !return_best_only) {
        discrete_state_history = py::array_t<int>(
            { size_t(num_steps), size_t(num_spins) }
        );
        auto bi = discrete_state_history.request();
        ptr_discrete_states = static_cast<int*>(bi.ptr);
    }

    // Allocate energy (Hamiltonian) history if requested
    py::array_t<double> energy_history;
    double* ptr_energy = nullptr;
    if (return_energy && !return_best_only) {
        energy_history = py::array_t<double>(num_steps);
        auto bi = energy_history.request();
        ptr_energy = static_cast<double*>(bi.ptr);
    }

    // Allocate cut value history if requested
    py::array_t<double> cut_value_history;
    double* ptr_cut_value = nullptr;
    if (return_cut_value && !return_best_only) {
        cut_value_history = py::array_t<double>(num_steps);
        auto bi = cut_value_history.request();
        ptr_cut_value = static_cast<double*>(bi.ptr);
    }
    
    //----------------------------------------------------------------
    // Time-stepping loop
    //----------------------------------------------------------------

    for (int step = 0; step < num_steps; ++step) {
        // Compute coupling between spins
        compute_coupling(x, coupling_arr, sources, targets);

        // Model-specific update
        switch (model_type) {
            case ModelType::POLYNOMIAL:
                // RK4 integration
                // Stage 1
                compute_polynomial_derivative(x, coupling_arr, k1, num_states, polynomial_order, 
                                           beta_schedule[step], gamma_schedule[step]);
                
                // Stage 2
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k1[j] * dt / 2.0;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_polynomial_derivative(x_temp, coupling_temp, k2, num_states, polynomial_order, 
                                           beta_schedule[step], gamma_schedule[step]);
                
                // Stage 3
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k2[j] * dt / 2.0;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_polynomial_derivative(x_temp, coupling_temp, k3, num_states, polynomial_order, 
                                           beta_schedule[step], gamma_schedule[step]);
                
                // Stage 4
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k3[j] * dt;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_polynomial_derivative(x_temp, coupling_temp, k4, num_states, polynomial_order, 
                                           beta_schedule[step], gamma_schedule[step]);
                
                // Update state
                for (int i = 0; i < num_spins; ++i) {
                    // Generate noise
                    noise = std::complex<double>(dist(gen), dist(gen)) * noise_scale;
                    x[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (dt / 6.0) + noise;
                }
                break;

            case ModelType::NEC:
                // Update alpha array using forward Euler (unchanged)
                for (int i = 0; i < num_spins; ++i) {
                    // Precompute state-dependent value for alpha update
                    abs_x = std::abs(x[i]);
                    abs_x_sqrt = std::sqrt(abs_x);
                    
                    // Update alpha using forward Euler
                    alpha_arr[i] += dt * alpha_rate * (sqrt_r_target - abs_x_sqrt);
                }
                
                // RK4 integration
                // Stage 1
                compute_nec_derivative(x, coupling_arr, k1, num_states, alpha_arr, gamma_schedule[step]);
                
                // Stage 2
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k1[j] * dt / 2.0;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_nec_derivative(x_temp, coupling_temp, k2, num_states, alpha_arr, gamma_schedule[step]);
                
                // Stage 3
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k2[j] * dt / 2.0;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_nec_derivative(x_temp, coupling_temp, k3, num_states, alpha_arr, gamma_schedule[step]);
                
                // Stage 4
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k3[j] * dt;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_nec_derivative(x_temp, coupling_temp, k4, num_states, alpha_arr, gamma_schedule[step]);
                
                // Update state
                for (int i = 0; i < num_spins; ++i) {
                    // Generate noise
                    noise = std::complex<double>(dist(gen), dist(gen)) * noise_scale;
                    x[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (dt / 6.0) + noise;
                }
                break;

            case ModelType::SIGMOID:
                // RK4 integration
                // Stage 1
                compute_sigmoid_derivative(x, coupling_arr, k1, num_states, alpha, 
                                         beta_schedule[step], gamma_schedule[step]);
                
                // Stage 2
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k1[j] * dt / 2.0;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_sigmoid_derivative(x_temp, coupling_temp, k2, num_states, alpha, 
                                         beta_schedule[step], gamma_schedule[step]);
                
                // Stage 3
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k2[j] * dt / 2.0;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_sigmoid_derivative(x_temp, coupling_temp, k3, num_states, alpha, 
                                         beta_schedule[step], gamma_schedule[step]);
                
                // Stage 4
                for (int j = 0; j < num_spins; ++j) {
                    x_temp[j] = x[j] + k3[j] * dt;
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_sigmoid_derivative(x_temp, coupling_temp, k4, num_states, alpha, 
                                         beta_schedule[step], gamma_schedule[step]);
                
                // Update state
                for (int i = 0; i < num_spins; ++i) {
                    // Generate noise
                    noise = std::complex<double>(dist(gen), dist(gen)) * noise_scale;
                    x[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (dt / 6.0) + noise;
                }
                break;

            case ModelType::FIXED_AMPLITUDE:
                // RK4 integration for theta (phase)
                // Stage 1
                compute_fixed_amplitude_derivative(x, coupling_arr, dtheta_dt_vec, num_states, gamma_schedule[step]);
                
                // Stage 2
                for (int j = 0; j < num_spins; ++j) {
                    double theta_j = std::arg(x[j]);
                    x_temp[j] = std::polar(1.0, theta_j + dtheta_dt_vec[j] * dt / 2.0);
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_fixed_amplitude_derivative(x_temp, coupling_temp, dtheta_dt_vec, num_states, gamma_schedule[step]);
                
                // Stage 3
                for (int j = 0; j < num_spins; ++j) {
                    double theta_j = std::arg(x[j]);
                    x_temp[j] = std::polar(1.0, theta_j + dtheta_dt_vec[j] * dt / 2.0);
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_fixed_amplitude_derivative(x_temp, coupling_temp, dtheta_dt_vec, num_states, gamma_schedule[step]);
                
                // Stage 4
                for (int j = 0; j < num_spins; ++j) {
                    double theta_j = std::arg(x[j]);
                    x_temp[j] = std::polar(1.0, theta_j + dtheta_dt_vec[j] * dt);
                }
                compute_coupling(x_temp, coupling_temp, sources, targets);
                compute_fixed_amplitude_derivative(x_temp, coupling_temp, dtheta_dt_vec, num_states, gamma_schedule[step]);
                
                // Update theta using RK4
                for (int i = 0; i < num_spins; ++i) {
                    // Generate real noise
                    phase_noise = dist(gen) * noise_scale;
                    
                    // Current state
                    double theta = std::arg(x[i]);
                    
                    // Create temporary storage for RK4 steps
                    double k1_i = dtheta_dt_vec[i];
                    double k2_i = dtheta_dt_vec[i]; // From stage 2
                    double k3_i = dtheta_dt_vec[i]; // From stage 3
                    double k4_i = dtheta_dt_vec[i]; // From stage 4
                    
                    // Update theta using RK4
                    double dtheta = (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) * (dt / 6.0);
                    
                    // Update state (keeping amplitude = 1.0)
                    theta += dtheta + phase_noise;
                    x[i] = std::polar(1.0, theta);
                }
                break;
        }   // end switch

        // Round to nearest Potts state (if return_discrete_states, return_energy, or return_cut_value is true)
        if (return_discrete_states || return_energy || return_cut_value) {
            for (int i = 0; i < num_spins; ++i) {
                double max_dot = -1e300;
                int closest_state = 0;
                for (int j = 0; j < num_states; ++j) {
                    double dot = std::real(ref_state_conj[j] * x[i]); // dot product with reference Potts state
                    if (dot > max_dot) {
                        max_dot = dot;
                        closest_state = j;
                    }
                }
                rounded_state[i] = closest_state; // Store the rounded state
            }
        }

        if (return_energy || return_cut_value) {
            energy = 0.0;
            cut_value = 0.0;
            for (size_t conn_idx = 0; conn_idx < sources.size(); ++conn_idx) {
                int s_idx = sources[conn_idx];
                int t_idx = targets[conn_idx];
                if (rounded_state[s_idx] == rounded_state[t_idx]) {
                    energy += 1.0;
                }
            }
            if (return_cut_value) {
                cut_value = num_edges - energy;
            }
            
            // Track the best solution
            bool is_better_solution = false;
            if (return_energy && energy < best_energy) {
                best_energy = energy;
                is_better_solution = true;
            }
            if (return_cut_value && cut_value > best_cut_value) {
                best_cut_value = cut_value;
                is_better_solution = true;
            }
            
            // If we found a better solution, store it
            if (is_better_solution) {
                best_step = step;
                
                if (return_continuous_states) {
                    std::copy(x.begin(), x.end(), best_continuous_state.begin());
                }
                
                if (return_discrete_states) {
                    std::copy(rounded_state.begin(), rounded_state.end(), best_discrete_state.begin());
                }
            }
        }

        // Store states and energy if requested (only for full history mode)
        if (!return_best_only) {
            if (return_continuous_states) {
                std::memcpy(ptr_continuous_states + static_cast<size_t>(step) * num_spins,
                    x.data(),
                    static_cast<size_t>(num_spins) * sizeof(std::complex<double>));
            }
            if (return_discrete_states) {
                std::memcpy(ptr_discrete_states + step * num_spins, 
                    rounded_state.data(), 
                    num_spins * sizeof(int));
            }
            if (return_energy)       ptr_energy[step]    = energy;
            if (return_cut_value)    ptr_cut_value[step] = cut_value;
        }
    }  // end time-stepping loop

    // Create return values
    py::dict out;
    
    // If returning only the best solution, create single-step arrays
    if (return_best_only) {
        if (return_continuous_states) {
            continuous_state_history = py::array_t<std::complex<double>>(
                { size_t(1), size_t(num_spins) }
            );
            auto bi = continuous_state_history.request();
            ptr_continuous_states = static_cast<std::complex<double>*>(bi.ptr);
            std::memcpy(ptr_continuous_states, best_continuous_state.data(), 
                        static_cast<size_t>(num_spins) * sizeof(std::complex<double>));
        }
        
        if (return_discrete_states) {
            discrete_state_history = py::array_t<int>(
                { size_t(1), size_t(num_spins) }
            );
            auto bi = discrete_state_history.request();
            ptr_discrete_states = static_cast<int*>(bi.ptr);
            std::memcpy(ptr_discrete_states, best_discrete_state.data(), 
                        static_cast<size_t>(num_spins) * sizeof(int));
        }
        
        if (return_energy) {
            energy_history = py::array_t<double>(1);
            auto bi = energy_history.request();
            ptr_energy = static_cast<double*>(bi.ptr);
            ptr_energy[0] = best_energy;
        }
        
        if (return_cut_value) {
            cut_value_history = py::array_t<double>(1);
            auto bi = cut_value_history.request();
            ptr_cut_value = static_cast<double*>(bi.ptr);
            ptr_cut_value[0] = best_cut_value;
        }
    }
    
    // Add the results to the output dictionary
    out["continuous_states"] = return_continuous_states ? py::object(continuous_state_history) : py::none();
    out["discrete_states"] = return_discrete_states ? py::object(discrete_state_history) : py::none();
    out["energy"] = return_energy ? py::object(energy_history) : py::none();
    out["cut_value"] = return_cut_value ? py::object(cut_value_history) : py::none();
    out["step"] = return_best_only ? py::cast(best_step) : py::none();
    
    return out;
    
}

// wrapper for polynomial model
py::object run_polynomial(
    double T, double dt, int num_spins, int num_states,
    const py::array_t<int>& edges,
    double noise_factor, int seed,
    int polynomial_order,
    const std::vector<double>& beta_schedule,
    const std::vector<double>& gamma_schedule,
    bool return_continuous_states,
    bool return_discrete_states,
    bool return_energy,
    bool return_cut_value,
    bool return_best_only = false)
{
    return run(
        T, dt, num_spins, num_states,
        edges,
        noise_factor, seed,
        ModelType::POLYNOMIAL,
        polynomial_order,
        /*alpha*/0.0, /*alpha_rate*/0.0, /*r_target*/0.0,
        /*initial_alpha_arr*/{}, beta_schedule, gamma_schedule,
        return_continuous_states,
        return_discrete_states,
        return_energy,
        return_cut_value,
        return_best_only);
}

// wrapper for NEC model
py::object run_nec(
    double T, double dt, int num_spins, int num_states,
    const py::array_t<int>& edges,
    double noise_factor, int seed,
    double alpha_rate,
    double r_target,
    const std::vector<double>& initial_alpha_arr,
    const std::vector<double>& gamma_schedule,
    bool return_continuous_states,
    bool return_discrete_states,
    bool return_energy,
    bool return_cut_value,
    bool return_best_only = false)
{
    return run(
        T, dt, num_spins, num_states,
        edges,
        noise_factor, seed,
        ModelType::NEC,
        /*polynomial_order*/0,
        /*alpha*/0.0, alpha_rate, r_target,
        initial_alpha_arr, /*beta_schedule*/{}, gamma_schedule,
        return_continuous_states,
        return_discrete_states,
        return_energy,
        return_cut_value,
        return_best_only);
}

// wrapper for sigmoid model
py::object run_sigmoid(
    double T, double dt, int num_spins, int num_states,
    const py::array_t<int>& edges,
    double noise_factor, int seed,
    double alpha,
    const std::vector<double>& beta_schedule,
    const std::vector<double>& gamma_schedule,
    bool return_continuous_states,
    bool return_discrete_states,
    bool return_energy,
    bool return_cut_value,
    bool return_best_only = false)
{
    return run(
        T, dt, num_spins, num_states,
        edges,
        noise_factor, seed,
        ModelType::SIGMOID,
        /*polynomial_order*/0,
        alpha, /*alpha_rate*/0.0, /*r_target*/0.0,
        /*initial_alpha_arr*/{}, beta_schedule, gamma_schedule,
        return_continuous_states,
        return_discrete_states,
        return_energy,
        return_cut_value,
        return_best_only);
}

// wrapper for fixed_amplitude model
py::object run_fixed_amplitude(
    double T, double dt, int num_spins, int num_states,
    const py::array_t<int>& edges,
    double noise_factor, int seed,
    const std::vector<double>& gamma_schedule,
    bool return_continuous_states,
    bool return_discrete_states,
    bool return_energy,
    bool return_cut_value,
    bool return_best_only = false)
{
    return run(
        T, dt, num_spins, num_states,
        edges,
        noise_factor, seed,
        ModelType::FIXED_AMPLITUDE,
        /*polynomial_order*/0,
        /*alpha*/0.0, /*alpha_rate*/0.0, /*r_target*/0.0,
        /*initial_alpha_arr*/{}, /*beta_schedule*/{}, gamma_schedule,
        return_continuous_states,
        return_discrete_states,
        return_energy,
        return_cut_value,
        return_best_only);
}

//-----------------------------------------------------------
// Pybind11 module definition
//-----------------------------------------------------------
PYBIND11_MODULE(potts_sim, m) {
    m.doc() = R"pbdoc(
        Discrete time evolution of a q-state Potts machine model.
    )pbdoc";

    py::enum_<ModelType>(m, "ModelType")
        .value("NEC",           ModelType::NEC)
        .value("POLYNOMIAL",    ModelType::POLYNOMIAL)
        .value("SIGMOID",       ModelType::SIGMOID)
        .value("FIXED_AMPLITUDE",ModelType::FIXED_AMPLITUDE)
        .export_values();

    // register model‐specific run functions
    m.def("run_polynomial", &run_polynomial,
          py::arg("T"), py::arg("dt"), py::arg("num_spins"), py::arg("num_states"),
          py::arg("edges"), py::arg("noise_factor"), py::arg("seed") = 1,
          py::arg("polynomial_order"),
          py::arg("beta_schedule"), py::arg("gamma_schedule"),
          py::arg("return_continuous_states") = true,
          py::arg("return_discrete_states") = false,
          py::arg("return_energy") = false,
          py::arg("return_cut_value") = false,
          py::arg("return_best_only") = false);

    m.def("run_nec", &run_nec,
          py::arg("T"), py::arg("dt"), py::arg("num_spins"), py::arg("num_states"),
          py::arg("edges"), py::arg("noise_factor"), py::arg("seed") = 1,
          py::arg("alpha_rate"), py::arg("r_target"),
          py::arg("initial_alpha_arr"), py::arg("gamma_schedule"),
          py::arg("return_continuous_states") = true,
          py::arg("return_discrete_states") = false,
          py::arg("return_energy") = false,
          py::arg("return_cut_value") = false,
          py::arg("return_best_only") = false);

    m.def("run_sigmoid", &run_sigmoid,
          py::arg("T"), py::arg("dt"), py::arg("num_spins"), py::arg("num_states"),
          py::arg("edges"), py::arg("noise_factor"), py::arg("seed") = 1,
          py::arg("alpha"), py::arg("beta_schedule"), py::arg("gamma_schedule"),
          py::arg("return_continuous_states") = true,
          py::arg("return_discrete_states") = false,
          py::arg("return_energy") = false,
          py::arg("return_cut_value") = false,
          py::arg("return_best_only") = false);

    m.def("run_fixed_amplitude", &run_fixed_amplitude,
          py::arg("T"), py::arg("dt"), py::arg("num_spins"), py::arg("num_states"),
          py::arg("edges"), py::arg("noise_factor"), py::arg("seed") = 1,
          py::arg("gamma_schedule"),
          py::arg("return_continuous_states") = true,
          py::arg("return_discrete_states") = false,
          py::arg("return_energy") = false,
          py::arg("return_cut_value") = false,
          py::arg("return_best_only") = false);
}
