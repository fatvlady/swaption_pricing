#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
#include <tuple>
#include <Eigen/Dense>


#include "polynomial_regression.hpp"

using std::sqrt;
using std::exp;


class ornstein_uhlenbeck_process
{
public:
    ornstein_uhlenbeck_process(double speed, double vol, double level = 0.0)
        : speed_(speed), level_(level), volatility_(vol) {}

    template <typename V>
    V evolve(double t0, V x0, double dt, V dw) const {
        return expectation(t0, x0, dt) + stdDeviation(t0, x0, dt) * dw;
    }

    template <typename V>
    V expectation(double, V x0, double dt) const {
        return level_ + (x0 - level_) * std::exp(-speed_ * dt);
    }

    template <typename V>
    V stdDeviation(double t, V x0, double dt) const {
        return sqrt(variance(t, x0, dt));
    }

    template <typename V>
    V variance(double, V x0, double dt) const {
        V ones = x0 * 0.0 + 1.0;
        if (std::abs(speed_) < std::sqrt(std::numeric_limits<double>::epsilon())) {
            // algebraic limit for small speed
            return volatility_ * volatility_ * dt * ones;
        }
        else {
            return 0.5 * volatility_ * volatility_ / speed_ * (1.0 - std::exp(-2.0 * speed_ * dt)) * ones;
        }
    }
private:
    double speed_, level_;
    double volatility_;
};

inline std::vector<value_t> integrate_sr(const std::vector<value_t>& short_rate, const std::vector<double>& grid)
{
    value_t integrated_sr = 0.0 * short_rate.front();
    std::vector<value_t> sdf(grid.size(), integrated_sr + 1.0);
    for (int i = 1; i < grid.size(); ++i)
    {
        const value_t& r = short_rate[i];
        const double delta = grid[i] - grid[i - 1];
        integrated_sr += r * delta;
        sdf[i] = exp(-integrated_sr);
    }
    return sdf;
}

inline constexpr double rate_std = 0.001;
inline constexpr double swap_init_mean = 10.0;
inline constexpr double swap_init_std = 0.5;
inline constexpr double swap_speed = 1.0;
inline constexpr double swap_std = 0.2;
inline constexpr double swap_level = swap_init_mean;

struct inputs_holder
{
	std::vector<value_t> swap_exposure;
	std::vector<value_t> sdf;
	const std::vector<int> ex_indices;
	int maturity_index;
};

struct outputs_holder
{
	double npv;
	std::vector<value_t> swaption_exposure;
};

auto setup(size_t paths, size_t grid_size, size_t ex_times_size, unsigned long seed)
{
    std::vector<double> grid(grid_size);
    std::iota(begin(grid), end(grid), 0);
    std::mt19937 rng(seed);
    std::vector<double> ex_times(ex_times_size);
    std::generate(begin(ex_times), end(ex_times), [&rng, grid_size] { return std::uniform_real_distribution<>(grid_size * 0.2, grid_size * 0.6)(rng); });
    std::sort(begin(ex_times), end(ex_times));
    std::vector<value_t> wiener(grid_size, value_t(paths));
    for (auto& w : wiener)
    {
        std::generate_n(std::data(w), std::size(w), [&rng] { return std::normal_distribution<>(0.0, rate_std)(rng); });
    }
    std::partial_sum(begin(wiener), end(wiener), begin(wiener));
    std::vector<value_t> sdf = integrate_sr(wiener, grid);

    ornstein_uhlenbeck_process swap_process(swap_speed, swap_std, swap_level);
    value_t swap_init(paths);
    std::generate_n(std::data(swap_init), std::size(swap_init), [&rng] { return std::normal_distribution<>(swap_init_mean, swap_init_std)(rng); });

    std::vector<value_t> swap_exposure{ swap_init };
    for (size_t i = 1; i < grid_size; ++i)
    {
        swap_exposure.emplace_back(swap_process.evolve<value_t>(grid[i - 1], swap_exposure.back(), grid[i] - grid[i - 1], wiener[i] - wiener[i - 1]));
    }

    std::vector<int> ex_indices(ex_times.size());
    std::transform(begin(ex_times), end(ex_times), begin(ex_indices), [](double t) { return int(std::ceil(t)); });
    int maturity_index = std::min(ex_indices.back() + size_t(5), grid.size() - 1);

	return inputs_holder{ std::move(swap_exposure), std::move(sdf), std::move(ex_indices), maturity_index };
}

value_t cmp_lt(const value_t& left, const value_t& right, const value_t t, const value_t& f)
{
    return (left < right).select(t, f);
}

value_t cmp_lt(const value_t& left, double right, const value_t t, const value_t& f)
{
	return (left < right).select(t, f);
}

auto mean(const value_t& v)
{
    return v.sum() / v.size();
}

auto swaption_exposure(const std::vector<value_t>& swap_exposure, const std::vector<value_t>& sdf, const std::vector<int>& ex_indices, int maturity_index)
{
    const int total_steps = swap_exposure.size(); // number of discretization points in paths.
    assert(sdf.size() == total_steps && "Size of underlyings and sdfs must agree.");
    const int ex_times_count = ex_indices.size(); // number of possible exersice dates.
    assert(ex_times_count >= 1 && "It should be at least one exercise date.");

    int current_ex = ex_times_count - 1; // current index for exercise times.
    const value_t zeros = swap_exposure.front() * 0.0;
    const value_t ones = zeros + 1.0;

    std::vector<value_t> is_exercised(ex_times_count);
    std::vector<value_t> swaption_exposure(total_steps); // swaption price                                            

    for (int step = maturity_index; step < total_steps; ++step)
        swaption_exposure[step] = zeros;

    value_t current_option_slice = zeros; // at last step swaption price is zero.
    // Backward loop.
    for (int i = 0; i < maturity_index; ++i)
    {
        const auto step = maturity_index - 1 - i;
        // Perform approximation, recalculate current_option_slice.
        if (!!i)
        {
            value_t discounted_opt_price = current_option_slice * sdf[step + 1] / sdf[step];
            current_option_slice = approximate(sdf[step], swap_exposure[step], discounted_opt_price, 5);
        }

        // If there are some exercise dates left and 
        // there is some exercise date on interval (t_{step - 1}; t_{step}].
        if (current_ex >= 0 && step == ex_indices[current_ex])
        {
            is_exercised[current_ex] = cmp_lt(current_option_slice, swap_exposure[step], ones, zeros);
            current_option_slice = cmp_lt(current_option_slice, swap_exposure[step], swap_exposure[step], current_option_slice);
            current_ex--;
            while (current_ex >= 0 && step == ex_indices[current_ex])
            {
                is_exercised[current_ex] = zeros;
                current_ex--;
            }
        }
        swaption_exposure[step] = current_option_slice;
    }

    // At start point nothing is exercised.
    value_t is_exercised_slice = zeros;

    // Forward loop to fill option state slices.
    for (int ex_index = 1; ex_index < ex_times_count; ++ex_index)
    {
        // disjunction
        is_exercised[ex_index] = is_exercised[ex_index - 1] + is_exercised[ex_index] - is_exercised[ex_index - 1] * is_exercised[ex_index];
    }

    for (int ex_index = 0; ex_index < ex_times_count; ++ex_index)
    {
        int interval_end = ex_index < (ex_times_count - 1) ? ex_indices[ex_index + 1] + 1 : total_steps;
		for (int step = ex_indices[ex_index] + 1; step < interval_end; step++)
			swaption_exposure[step] = cmp_lt(is_exercised[ex_index], 1.0, swaption_exposure[step], swap_exposure[step]);
    }
	return outputs_holder{ mean(swaption_exposure[0]), std::move(swaption_exposure) };
}

auto swaption_exposure(const inputs_holder& holder)
{
	return swaption_exposure(holder.swap_exposure, holder.sdf, holder.ex_indices, holder.maturity_index);
}
