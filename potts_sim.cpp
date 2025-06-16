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
    bool return_best_only,
    bool return_last_only
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
    //      where theta_j = 2*pi/q * (j - floor(q/2))
    std::vector<std::complex<double>> ref_state_conj(num_states);
    for (int j = 0; j < num_states; ++j) {
        double theta_j = 2.0 * M_PI / num_states * (j - std::floor(num_states / 2.0));
        ref_state_conj[j] = std::polar(1.0, -theta_j);
    }

    // Allocate partial results for update rules
    std::complex<double> noise;
    double phase_noise;
    double abs_x;
    double abs_x_sqrt;
    double abs_x_pow;
    double abs_x_pow1;
    double abs_x_pow2;
    std::complex<double> conj_x;
    std::complex<double> conj_pow;
    double theta;
    double cos_theta;
    double sin_theta;
    double sin_q_theta;
    double phase_coupling;
    std::complex<double> rotated_coupling;
    double dr_dt;
    double dtheta_dt;
    std::complex<double> dx_dt;
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

    // Allocate continuous state (complex amplitudes) history if requested
    py::array_t<std::complex<double>> continuous_state_history;
    std::complex<double>* ptr_continuous_states = nullptr;
    if (return_continuous_states && !return_best_only && !return_last_only) {
        continuous_state_history = py::array_t<std::complex<double>>(
            { size_t(num_steps), size_t(num_spins) }
        );
        auto bi = continuous_state_history.request();
        ptr_continuous_states = static_cast<std::complex<double>*>(bi.ptr);
    }

    // Allocate discrete state (rounded Potts state) history if requested
    py::array_t<int> discrete_state_history;
    int* ptr_discrete_states = nullptr;
    if (return_discrete_states && !return_best_only && !return_last_only) {
        discrete_state_history = py::array_t<int>(
            { size_t(num_steps), size_t(num_spins) }
        );
        auto bi = discrete_state_history.request();
        ptr_discrete_states = static_cast<int*>(bi.ptr);
    }

    // Allocate energy (Hamiltonian) history if requested
    py::array_t<double> energy_history;
    double* ptr_energy = nullptr;
    if (return_energy && !return_best_only && !return_last_only) {
        energy_history = py::array_t<double>(num_steps);
        auto bi = energy_history.request();
        ptr_energy = static_cast<double*>(bi.ptr);
    }

    // Allocate cut value history if requested
    py::array_t<double> cut_value_history;
    double* ptr_cut_value = nullptr;
    if (return_cut_value && !return_best_only && !return_last_only) {
        cut_value_history = py::array_t<double>(num_steps);
        auto bi = cut_value_history.request();
        ptr_cut_value = static_cast<double*>(bi.ptr);
    }
    
    //----------------------------------------------------------------
    // Time-stepping loop
    //----------------------------------------------------------------

    for (int step = 0; step < num_steps; ++step) {
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

        // Model-specific update
        switch (model_type) {
            case ModelType::POLYNOMIAL:
                for (int i = 0; i < num_spins; ++i) {
                    // Generate noise
                    noise = std::complex<double>(dist(gen), dist(gen)) * noise_scale;

                    // Precompute state-dependent values
                    abs_x = std::abs(x[i]); // |x_i|
                    abs_x_pow = std::pow(abs_x, polynomial_order - 1); // |x_i|^(n-1)
                    conj_x = std::conj(x[i]); // conj(x_i)
                    conj_pow = std::pow(conj_x, num_states - 1); // conj(x_i)^(q-1)

                    // Calculate dx/dt
                    //      dx_i/dt = -x - |x|^(n-1) * x + gamma[step] * conj(x_i)^(q-1) + beta[step] * coupling
                    dx_dt = - x[i]
                            - abs_x_pow * x[i]
                            + gamma_schedule[step] * conj_pow
                            + beta_schedule[step] * coupling_arr[i];

                    // Update state
                    x[i] += dx_dt * dt + noise;
                }
                break;

            case ModelType::NEC:
                for (int i = 0; i < num_spins; ++i) {
                    // Generate noise
                    noise = std::complex<double>(dist(gen), dist(gen)) * noise_scale;

                    // Precompute state-dependent values
                    abs_x = std::abs(x[i]); // |x_i|
                    abs_x_sqrt = std::sqrt(abs_x); // sqrt(|x_i|)
                    abs_x_pow = std::pow(abs_x, polynomial_order - 1); // |x_i|^(n-1), n=3 in original NEC model
                    conj_x = std::conj(x[i]); // conj(x_i)
                    conj_pow = std::pow(conj_x, num_states - 1); // conj(x_i)^(q-1)

                    // Update alpha array
                    //      alpha_i(t+dt) += dt * alpha_rate * (sqrt(r_target) - sqrt(|x_i|))
                    alpha_arr[i] += dt * alpha_rate * (sqrt_r_target - abs_x_sqrt);

                    // Calculate dx/dt
                    //   dx_i/dt = -x_i - |x_i|^2 * x + gamma[step] * conj(x_i)^(q-1) + coupling
                    dx_dt = alpha_arr[i] * x[i]
                            - abs_x_pow * x[i]
                            + gamma_schedule[step] * conj_pow
                            + coupling_arr[i];
                    
                    // Update state
                    x[i] += dx_dt * dt + noise; 
                }
                break;

            case ModelType::SIGMOID:
                for (int i = 0; i < num_spins; ++i) {
                    // Generate noise
                    noise = std::complex<double>(dist(gen), dist(gen)) * noise_scale;

                    // Precompute state-dependent values
                    abs_x = std::abs(x[i]); // |x_i|
                    abs_x_pow1 = std::pow(abs_x, num_states - 1); // |x_i|^(q-1)
                    abs_x_pow2 = std::pow(abs_x, num_states - 2); // |x_i|^(q-2)
                    theta = std::arg(x[i]); // argument of x_i
                    cos_theta = std::cos(theta); // cos(theta_i)
                    sin_theta = std::sin(theta); // sin(theta_i)
                    // rotated_coupling = sum_j( J_ij * |x_j| * exp(i (theta_j - theta_i))) 
                    //                  = sum_j( J_ij * |x_j| * exp(i theta_j)) * exp(-i theta_i)
                    //                  = coupling_arr[i] * exp(-i theta_i)
                    rotated_coupling = coupling_arr[i] * std::polar(1.0, -theta);

                    // Calculate dr/dt
                    //      dr_i/dt = -|x_i| + tanh(alpha * |x_i| + gamma[step] * |x_i|^(q-1) * cos(q * theta_i) + beta[step] * Re(rotated_coupling))
                    dr_dt = -abs_x + std::tanh(
                            alpha * abs_x
                            + gamma_schedule[step] * abs_x_pow1 * std::cos(num_states * theta)
                            + beta_schedule[step] * std::real(rotated_coupling));

                    // Calculate dtheta/dt
                    //      dtheta_i/dt = - gamma[step] * |x_i|^(q-2) * sin(q * theta_i) + beta[step] / (|x_i| + 1e-10) * Im(rotated_coupling)
                    dtheta_dt = - gamma_schedule[step] * abs_x_pow2 * std::sin(num_states * theta)
                                + beta_schedule[step] / (abs_x + 1e-10) * std::imag(rotated_coupling);
                    // Note: 1e-10 is a small constant to avoid division by zero

                    // Calculate dx/dt
                    //      dx_i/dt = dr_i/dt * exp(i theta_i) + |x_i| * dtheta_i/dt * exp(i theta_i) * i
                    //      = (dr_i/dt - |x_i| * dtheta_i/dt * i) * exp(i theta_i),
                    //      Re(dx_i/dt) = dr_i/dt * cos(theta_i) - |x_i| * dtheta_i/dt * sin(theta_i),
                    //      Im(dx_i/dt) = dr_i/dt * sin(theta_i) + |x_i| * dtheta_i/dt * cos(theta_i)
                    dx_dt = std::complex<double>(
                        dr_dt * cos_theta - abs_x * dtheta_dt * sin_theta,
                        dr_dt * sin_theta + abs_x * dtheta_dt * cos_theta);

                    // Update state
                    x[i] += dx_dt * dt + noise;
                }
                break;

            case ModelType::FIXED_AMPLITUDE:
                for (int i = 0; i < num_spins; ++i) {
                    // Generate real noise
                    phase_noise = dist(gen) * noise_scale;

                    // Precompute state-dependent values
                    theta = std::arg(x[i]); // argument of x_i
                    sin_q_theta = std::sin(num_states * theta); // sin(q * theta_i)
                    // phase_coupling   = sum_j ( J_ij * sin(i (theta_j - theta_i))) 
                    //                  = Im(J_ij * exp(i (theta_j - theta_i))) 
                    //                  = Im(coupling_arr[i] * exp(i theta_j) * exp(-i theta_i))
                    phase_coupling = std::imag(coupling_arr[i] * std::polar(1.0, -theta));

                    // Calculate dtheta/dt
                    //      dtheta_i/dt = - gamma[step] * sin(q * theta_i) + phase_coupling
                    dtheta_dt = - gamma_schedule[step] * sin_q_theta + phase_coupling;

                    // Update argument
                    theta += dt * dtheta_dt + phase_noise;

                    // Update state
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
        if (!return_best_only && !return_last_only) {
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
    
    if (return_best_only) {
        if (return_continuous_states) {
            continuous_state_history = py::array_t<std::complex<double>>(
                { py::ssize_t(1), py::ssize_t(num_spins) }
            );
            auto bi = continuous_state_history.request();
            ptr_continuous_states = static_cast<std::complex<double>*>(bi.ptr);
            std::memcpy(ptr_continuous_states, best_continuous_state.data(), 
                        static_cast<size_t>(num_spins) * sizeof(std::complex<double>));
        }
        
        if (return_discrete_states) {
            discrete_state_history = py::array_t<int>(
                { py::ssize_t(1), py::ssize_t(num_spins) }
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
        out["step"] = py::cast(best_step);
    }
    else if (return_last_only) {
        // single‐step arrays for the last time‐step
        if (return_continuous_states) {
            continuous_state_history = py::array_t<std::complex<double>>(
                { py::ssize_t(1), py::ssize_t(num_spins) }
            );
            auto bi = continuous_state_history.request();
            ptr_continuous_states = static_cast<std::complex<double>*>(bi.ptr);
            std::memcpy(ptr_continuous_states, x.data(),
                        num_spins * sizeof(std::complex<double>));
        }
        if (return_discrete_states) {
            discrete_state_history = py::array_t<int>(
                { py::ssize_t(1), py::ssize_t(num_spins) }
            );
            auto bi = discrete_state_history.request();
            ptr_discrete_states = static_cast<int*>(bi.ptr);
            std::memcpy(ptr_discrete_states, rounded_state.data(),
                        num_spins * sizeof(int));
        }
        if (return_energy) {
            energy_history = py::array_t<double>(1);
            auto bi = energy_history.request();
            ptr_energy = static_cast<double*>(bi.ptr);
            ptr_energy[0] = energy;
        }
        if (return_cut_value) {
            cut_value_history = py::array_t<double>(1);
            auto bi = cut_value_history.request();
            ptr_cut_value = static_cast<double*>(bi.ptr);
            ptr_cut_value[0] = cut_value;
        }
        out["step"] = py::cast(num_steps - 1);
    }
    // Add the results to the output dictionary
    out["continuous_states"] = return_continuous_states ? py::object(continuous_state_history) : py::none();
    out["discrete_states"]   = return_discrete_states   ? py::object(discrete_state_history)   : py::none();
    out["energy"]            = return_energy            ? py::object(energy_history)           : py::none();
    out["cut_value"]         = return_cut_value         ? py::object(cut_value_history)        : py::none();
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
    bool return_best_only = false,
    bool return_last_only = false)
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
        return_best_only,
        return_last_only);
}

// wrapper for NEC model
py::object run_nec(
    double T, double dt, int num_spins, int num_states,
    const py::array_t<int>& edges,
    double noise_factor, int seed,
    int polynomial_order,
    double alpha_rate,
    double r_target,
    const std::vector<double>& initial_alpha_arr,
    const std::vector<double>& gamma_schedule,
    bool return_continuous_states,
    bool return_discrete_states,
    bool return_energy,
    bool return_cut_value,
    bool return_best_only = false,
    bool return_last_only = false)
{
    return run(
        T, dt, num_spins, num_states,
        edges,
        noise_factor, seed,
        ModelType::NEC,
        polynomial_order,
        /*alpha*/0.0, alpha_rate, r_target,
        initial_alpha_arr, /*beta_schedule*/{}, gamma_schedule,
        return_continuous_states,
        return_discrete_states,
        return_energy,
        return_cut_value,
        return_best_only,
        return_last_only);
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
    bool return_best_only = false,
    bool return_last_only = false)
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
        return_best_only,
        return_last_only);
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
    bool return_best_only = false,
    bool return_last_only = false)
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
        return_best_only,
        return_last_only);
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
          py::arg("return_best_only") = false,
          py::arg("return_last_only") = false);

    m.def("run_nec", &run_nec,
          py::arg("T"), py::arg("dt"), py::arg("num_spins"), py::arg("num_states"),
          py::arg("edges"), py::arg("noise_factor"), py::arg("seed") = 1,
          py::arg("polynomial_order"), py::arg("alpha_rate"), py::arg("r_target"),
          py::arg("initial_alpha_arr"), py::arg("gamma_schedule"),
          py::arg("return_continuous_states") = true,
          py::arg("return_discrete_states") = false,
          py::arg("return_energy") = false,
          py::arg("return_cut_value") = false,
          py::arg("return_best_only") = false,
          py::arg("return_last_only") = false);

    m.def("run_sigmoid", &run_sigmoid,
          py::arg("T"), py::arg("dt"), py::arg("num_spins"), py::arg("num_states"),
          py::arg("edges"), py::arg("noise_factor"), py::arg("seed") = 1,
          py::arg("alpha"), py::arg("beta_schedule"), py::arg("gamma_schedule"),
          py::arg("return_continuous_states") = true,
          py::arg("return_discrete_states") = false,
          py::arg("return_energy") = false,
          py::arg("return_cut_value") = false,
          py::arg("return_best_only") = false,
          py::arg("return_last_only") = false);

    m.def("run_fixed_amplitude", &run_fixed_amplitude,
          py::arg("T"), py::arg("dt"), py::arg("num_spins"), py::arg("num_states"),
          py::arg("edges"), py::arg("noise_factor"), py::arg("seed") = 1,
          py::arg("gamma_schedule"),
          py::arg("return_continuous_states") = true,
          py::arg("return_discrete_states") = false,
          py::arg("return_energy") = false,
          py::arg("return_cut_value") = false,
          py::arg("return_best_only") = false,
          py::arg("return_last_only") = false);
}
