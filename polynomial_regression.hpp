#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
#include <tuple>
#include <Eigen/Dense>


typedef Eigen::ArrayXd value_t;
typedef Eigen::VectorXd vector_t;
typedef Eigen::MatrixXd matrix_t;

struct polynomial_regression
{
	explicit polynomial_regression(const value_t& x, size_t size, const value_t& weights = {})
		: size_(size), data_x_(x), weights_(weights)
	{
		if (weights_.size() == 0)
			weights_.setOnes(data_x_.size());
	}


	value_t solve(const value_t& y)
	{
		vector_t xty = x_degrees().transpose() * (y * weights_).matrix();
		vector_t coef = calculate_gramian().llt().solve(xty);
		return x_degrees() * coef;
	}

	// Gramian matrix of basis.
	matrix_t calculate_gramian()
	{
		matrix_t result;

		// Weighted powers of data_x_.
		value_t x_power_sum(2 * size_ - 1);
		x_power_sum(0) = weights_.sum();
		for (size_t i = 1; i < size_; i++)
			x_power_sum(i) = x_degrees().col(i).dot(weights_.matrix());

		value_t x_power = x_degrees().col(size_ - 1).array() * weights_;
		for (size_t i = size_; i < 2 * size_ - 1; i++)
		{
			x_power *= data_x_;
			x_power_sum(i) = x_power.sum();
		}

		// Calculate matrix X^T * X.
		result.resize(size_, size_);
		for (size_t i = 0; i < size_; i++)
			for (size_t j = 0; j <= i; j++)
				result(i, j) = result(j, i) = x_power_sum[i + j];

		//matrix_t result_dummy(size_, size_);
		//for (size_t i = 0; i < size_; i++)
		//	for (size_t j = i; j < size_; j++)
		//		result_dummy(i, j) = result_dummy(j, i) = (x_degrees().col(i).array() * x_degrees().col(j).array() * weights_).sum();
		//double diff = (result - result_dummy).cwiseAbs().maxCoeff();
		return result;
	}

	const matrix_t& x_degrees()
	{
		if (x_degrees_.size() == 0)
		{
			x_degrees_.resize(data_x_.size(), size_);
			x_degrees_.col(0) = value_t::Ones(data_x_.size());
			for (size_t i = 1; i < size_; i++)
				x_degrees_.col(i) = x_degrees_.col(i-1).array() * data_x_;
		}
		return x_degrees_;
	}

protected:
	size_t size_;					///< number of basis elements
	const value_t data_x_;			///< input data x, dim = x_dim
	value_t weights_;			    ///< weights of inputs, dim = dim_x
	matrix_t x_degrees_;			///< degrees of x, dim = (size, x_dim)
	matrix_t xtx_;					///< Gramian matrix X^T * X, dim = (size, size)
};

value_t approximate(const value_t& w, const value_t& x, const value_t& y, size_t dim)
{
	polynomial_regression regression(x, dim, w);
	return regression.solve(y);
}

