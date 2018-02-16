#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>
#include <boost/multi_array.hpp>
#include <Eigen/Dense>
#include "measurements.h"
#include "fast_update.h"
#include "configuration.h"

typedef fast_update<arg_t>::dmatrix_t matrix_t;
typedef fast_update<arg_t>::numeric_t numeric_t;

struct wick_sp_matrix
{
	configuration& config;
	Random& rng;
	std::vector<double> values;

	wick_sp_matrix(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{
		values.resize(4);
	}
	
	std::vector<double>& get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		const numeric_t *ca_td_gf = td_gf.data();
		auto& K = config.l.symmetry_point("K");
		auto& Kp = config.l.symmetry_point("Kp");
		const int N = config.l.n_sites();
		std::fill(values.begin(), values.end(), 0.);
		for (int i = 0; i < N; i+=2)
		for (int u = 0; u < 2; ++u)
		{
			auto& r_i = config.l.real_space_coord(i);
			for (int j = 0; j < N; j+=2)
			for (int v = 0; v < 2; ++v)
			{
				auto& r_j = config.l.real_space_coord(j);
				double kdot = K.dot(r_i) + Kp.dot(r_j);
				values[u*2+v] += std::real(std::cos(kdot) * ca_td_gf[(j+v)*N+(i+u)] * config.l.parity((i+u))*config.l.parity((j+v)) / static_cast<double>(N/2));
				//values[u*2+v] += std::cos(kdot) * ca_td_gf[(j+v)*N+(i+u)] / (N/2);
			}
		}
		return values;
	}
};

struct wick_tp_matrix
{
	configuration& config;
	Random& rng;
	std::vector<double> values;

	wick_tp_matrix(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{
		values.resize(4);
	}
	
	std::vector<double>& get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		const numeric_t *ca_td_gf = td_gf.data();
		auto& K = config.l.symmetry_point("K");
		const int N = config.l.n_sites();
		std::fill(values.begin(), values.end(), 0.);
		for (int i = 0; i < N; i+=2)
		//int i = 2 * static_cast<int>(rng() * N / 2);
		for (int m = 0; m < N; m+=2)
		for (int u = 0; u < 2; ++u)
		{
			auto& r_i = config.l.real_space_coord(i);
			auto& r_m = config.l.real_space_coord(m);
			for (int j = 0; j < N; j+=2)
			for (int n = 0; n < N; n+=2)
			for (int v = 0; v < 2; ++v)
			{
				auto& r_j = config.l.real_space_coord(j);
				auto& r_n = config.l.real_space_coord(n);
				double kdot = K.dot(r_i - r_j - r_m + r_n);
				values[u*2+v] += std::real(std::cos(kdot) * (ca_td_gf[(i+u)*N+(m+u)] * ca_td_gf[(j+v)*N+(n+v)] - ca_td_gf[(i+u)*N+(n+v)] * ca_td_gf[(j+v)*N+(m+u)]) / static_cast<double>(N*N/4));
			}
		}
		return values;
	}
};
