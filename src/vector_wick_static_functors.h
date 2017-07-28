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

struct wick_static_S_chernAA
{
	configuration& config;
	Random& rng;
	const std::vector<std::pair<int, int>>& bonds;
	std::vector<double> fourier_coeff;
	std::vector<Eigen::Vector2d> q_vec;
	Eigen::Vector2d delta;
	std::vector<double> values;

	wick_static_S_chernAA(configuration& config_, Random& rng_,
		const std::vector<std::pair<int, int>>& bonds_, const Eigen::Vector2d& delta_)
		: config(config_), rng(rng_), bonds(bonds_), delta(delta_)
	{
		std::vector<Eigen::Vector2d> hexagon_pos;
		for (int i = 0; i < bonds.size(); i+=3)
		{
			Eigen::Vector2d r = config.l.real_space_coord(bonds[i].first) + delta;
			hexagon_pos.push_back(r);
		}
		
		auto& G = config.l.symmetry_point("Gamma");
		auto M = config.l.symmetry_point("M");
		auto& K = config.l.symmetry_point("K");
		for (int i = 0; i <= config.param.Lx / 2; ++i)
			q_vec.push_back(G + (config.l.b1 + config.l.b2) * static_cast<double>(i) / static_cast<double>(config.param.Lx));
		if (config.param.Lx % 2 != 0)
		{
			M = q_vec[config.param.Lx / 2] + config.l.b1 / static_cast<double>(config.param.Lx);
			q_vec.push_back(M);
		}
		for (int i = 1; i <= config.param.Lx / 6; ++i)
			q_vec.push_back(M + (config.l.b1 - config.l.b2) * static_cast<double>(i) / static_cast<double>(config.param.Lx));
		for (int i = 1; i < config.param.Lx / 3; ++i)
			q_vec.push_back(K + (-2.*config.l.b1 - config.l.b2) * static_cast<double>(i) / static_cast<double>(config.param.Lx));
		values.resize(q_vec.size());
		
		for (int i = 0; i < hexagon_pos.size(); ++i)
			for (int j = 0; j < hexagon_pos.size(); ++j)
			{
				auto delta_r = hexagon_pos[i] - hexagon_pos[j];
				for (int k = 0; k < q_vec.size(); ++k)
					fourier_coeff.push_back(std::cos(q_vec[k].dot(delta_r)));
			}
	}
	
	std::vector<double>& get_obs(const matrix_t& et_gf)
	{
		const numeric_t *ca_et_gf_0 = et_gf.data();
		const int N = bonds.size(), ns = config.l.n_sites(), nq = q_vec.size();
		std::fill(values.begin(), values.end(), 0.);
		for (int i = 0; i < N; ++i)
		{
			auto& a = bonds[i];
			for (int j = 0; j < N; ++j)
			{
				auto& b = bonds[j];
				numeric_t ch = -2.*(ca_et_gf_0[a.first*ns+a.second] * ca_et_gf_0[b.first*ns+b.second]
					+ ca_et_gf_0[b.second*ns+a.first] * ca_et_gf_0[b.first*ns+a.second]
					- ca_et_gf_0[a.second*ns+a.first] * ca_et_gf_0[b.first*ns+b.second]
					- ca_et_gf_0[b.second*ns+a.second] * ca_et_gf_0[b.first*ns+a.first])
					/ std::pow(config.l.n_bonds(), 2);
				for (int k = 0; k < nq; ++k)
					values[k] += ch * fourier_coeff[(i/3)*(N/3)*nq + (j/3)*nq + k];
			}
		}
		return values;
	}
};
