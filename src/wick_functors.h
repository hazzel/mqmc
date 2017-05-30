#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>
#include <boost/multi_array.hpp>
#include "measurements.h"
#include "configuration.h"

typedef fast_update<arg_t>::dmatrix_t matrix_t;
typedef fast_update<arg_t>::numeric_t numeric_t;

// M2(tau) = sum_ij <(n_i(tau) - 1/2)(n_j - 1/2)>
struct wick_M2
{
	configuration& config;
	Random& rng;

	wick_M2(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t M2 = 0.;
		const int N = config.l.n_sites();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
						/*
				M2 += config.l.parity(i) * config.l.parity(j)
					* std::real((1. - et_gf_t(i, i)) * (1. - et_gf_0(j, j))
					+ config.l.parity(i) * config.l.parity(j) * td_gf(i, j) * td_gf(i, j)
					- (et_gf_t(i, i) + et_gf_0(j, j))/2. + 1./4.);
						*/
				M2 += td_gf(i, j) * td_gf(i, j);
			}
		return std::real(M2) / std::pow(N, 2.);
	}
};

// kekule(tau) = sum_{kekule} <c_i^dag(tau) c_j(tau) c_n^dag c_m>
struct wick_kekule
{
	configuration& config;
	Random& rng;

	wick_kekule(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t kek = 0.;
		std::array<const std::vector<std::pair<int, int>>*, 3> kek_bonds =
			{&config.l.bonds("kekule"), &config.l.bonds("kekule_2"),
			&config.l.bonds("kekule_3")};
		std::array<double, 3> factors = {-1., -1., 2.};
		
		const int N = kek_bonds.size(), M = kek_bonds[0]->size();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < M; ++j)
			{
				auto& a = (*kek_bonds[i])[j];
				for (int m = 0; m < N; ++m)
					for (int n = 0; n < M; ++n)
					{
						auto& b = (*kek_bonds[m])[n];
						
						kek += factors[i] * factors[m]
							* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second));
					}
			}
		return std::real(kek) / std::pow(config.l.n_bonds(), 2.);
	}
};

// ep(tau) = sum_{<ij>,<mn>} <c_i^dag(tau) c_j(tau) c_n^dag c_m>
struct wick_epsilon
{
	configuration& config;
	Random& rng;

	wick_epsilon(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t ep = 0.;
		auto& single_bonds = config.l.bonds("single_d1_bonds");
		auto& bonds = config.l.bonds("single_d1_bonds");
		const int N = single_bonds.size(), M = bonds.size();
		for (int i = 0; i < N; ++i)
		{
			auto& a = single_bonds[i];
			for (int j = 0; j < M; ++j)
			{
				auto& b = bonds[j];
				ep += et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
						+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second);
			}
		}
		return std::real(2.*ep) / std::pow(M, 2.);
	}
};

struct wick_epsilon_V
{
	configuration& config;
	Random& rng;

	wick_epsilon_V(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t ep = 0.;
		auto& single_bonds = config.l.bonds("single_d1_bonds");
		const int N = single_bonds.size();
		for (int s = 0; s < N; ++s)
		{
			int i = single_bonds[s].first, j = single_bonds[s].second;
			double delta_ii = 1., delta_jj = 1.;
			double delta_ij = (i == j ? 1. : 0.);
			for (int t = 0; t < N; ++t)
			{
				int m = single_bonds[t].first, n = single_bonds[t].second;
				double delta_mm = 1., delta_nn = 1.;
				double delta_mn = (m == n ? 1. : 0.);
				ep += (delta_ii - et_gf_t(i, i)) * ((delta_jj - et_gf_t(j, j)) * ((delta_mm - et_gf_0(m, m)) * ((delta_nn - et_gf_0(n, n))) + (delta_mn - et_gf_0(n, m)) * (et_gf_0(m, n))) + (config.l.parity(j)*config.l.parity(m)*td_gf(m, j)) * (td_gf(j, m) * ((delta_nn - et_gf_0(n, n))) + (-td_gf(j, n)) * ((delta_mn - et_gf_0(n, m)))) + (config.l.parity(j)*config.l.parity(n)*td_gf(n, j)) * (td_gf(j, m) * (et_gf_0(m, n)) + td_gf(j, n) * ((delta_mm - et_gf_0(m, m)))))
				+ (delta_ij - et_gf_t(j, i)) * (et_gf_t(i, j) * ((delta_mm - et_gf_0(m, m)) * ((delta_nn - et_gf_0(n, n))) + (delta_mn - et_gf_0(n, m)) * (et_gf_0(m, n))) + (-td_gf(i, m)) * ((config.l.parity(j)*config.l.parity(m)*td_gf(m, j)) * ((delta_nn - et_gf_0(n, n))) + (config.l.parity(j)*config.l.parity(n)*td_gf(n, j)) * (et_gf_0(m, n))) + (-td_gf(i, n)) * ((-config.l.parity(j)*config.l.parity(m)*td_gf(m, j)) * ((delta_mn - et_gf_0(n, m))) + (config.l.parity(j)*config.l.parity(n)*td_gf(n, j)) * ((delta_mm - et_gf_0(m, m)))))
				+ (config.l.parity(i)*config.l.parity(m)*td_gf(m, i)) * (et_gf_t(i, j) * (td_gf(j, m) * ((delta_nn - et_gf_0(n, n))) + (-td_gf(j, n)) * ((delta_mn - et_gf_0(n, m)))) + td_gf(i, m) * ((delta_jj - et_gf_t(j, j)) * ((delta_nn - et_gf_0(n, n))) + (config.l.parity(j)*config.l.parity(n)*td_gf(n, j)) * (td_gf(j, n))) + (-td_gf(i, n)) * ((delta_jj - et_gf_t(j, j)) * ((delta_mn - et_gf_0(n, m))) + (config.l.parity(j)*config.l.parity(n)*td_gf(n, j)) * (td_gf(j, m))))
				+ (config.l.parity(i)*config.l.parity(n)*td_gf(n, i)) * (et_gf_t(i, j) * (td_gf(j, m) * (et_gf_0(m, n)) + td_gf(j, n) * ((delta_mm - et_gf_0(m, m)))) + td_gf(i, m) * ((delta_jj - et_gf_t(j, j)) * (et_gf_0(m, n)) + (-config.l.parity(j)*config.l.parity(m)*td_gf(m, j)) * (td_gf(j, n))) + td_gf(i, n) * ((delta_jj - et_gf_t(j, j)) * ((delta_mm - et_gf_0(m, m))) + (config.l.parity(j)*config.l.parity(m)*td_gf(m, j)) * (td_gf(j, m))));
			}
		}
		return std::real(ep) / std::pow(config.l.n_bonds(), 2.);
	}
};

struct wick_epsilon_as
{
	configuration& config;
	Random& rng;

	wick_epsilon_as(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t ep = 0.;
		std::vector<const std::vector<std::pair<int, int>>*> bonds =
			{&config.l.bonds("nn_bond_1"), &config.l.bonds("nn_bond_2"),
			&config.l.bonds("nn_bond_3")};
		
		const int N = bonds.size(), M = bonds[0]->size();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < M; ++j)
			{
				auto& a = (*bonds[i])[j];
				for (int m = 0; m < N; ++m)
					for (int n = 0; n < M; ++n)
					{
						auto& b = (*bonds[m])[n];
						
						ep += 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second));
							
						ep -= 2.*(et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(a.second, b.first) * td_gf(a.first, b.second));
						
						/*
						ep -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
							+ config.l.parity(a.first) * config.l.parity(b.second) * td_gf(a.first, b.second) * td_gf(a.second, b.first);
							
						ep += et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
							+ config.l.parity(a.second) * config.l.parity(b.second) * td_gf(a.second, b.second) * td_gf(a.first, b.first);
						*/
					}
			}
		return std::real(ep) / std::pow(config.l.n_bonds(), 2.);
	}
};

struct wick_cdw_s
{
	configuration& config;
	Random& rng;

	wick_cdw_s(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t ch = 0.;
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first)
					+ et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(a.second, b.first) * td_gf(a.first, b.second);
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern"))
			{
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first)
					+ et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(a.second, b.first) * td_gf(a.first, b.second);
			}
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first)
					+ et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(a.second, b.first) * td_gf(a.first, b.second);
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first)
					+ et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(a.second, b.first) * td_gf(a.first, b.second);
			}
		return std::real(ch) / std::pow(config.l.n_bonds(), 2.);
	}
};

// chern(tau) = sum_{chern} <c_i^dag(tau) c_j(tau) c_n^dag c_m>
struct wick_chern
{
	configuration& config;
	Random& rng;
	bond_map& xx;

	wick_chern(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t ch = 0.;
		auto& bonds_c1 = config.l.bonds("chern");
		auto& bonds_c2 = config.l.bonds("chern_2");
		const int N = bonds_c1.size();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
				auto& a = bonds_c1[i];
				auto& b = bonds_c1[j];
				/*
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(a.second, b.first) * td_gf(a.first, b.second);
				*/
				ch -= 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first));
			}
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
				auto& a = bonds_c2[i];
				auto& b = bonds_c1[j];
				/*
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(a.second, b.first) * td_gf(a.first, b.second);
				*/
				ch += 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first));
			}
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
				auto& a = bonds_c1[i];
				auto& b = bonds_c2[j];
				/*
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(a.second, b.first) * td_gf(a.first, b.second);
				*/
				ch += 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first));
			}
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
				auto& a = bonds_c2[i];
				auto& b = bonds_c2[j];
				/*
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(a.second, b.first) * td_gf(a.first, b.second);
				*/
				
				ch -= 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first));
			}
		return std::real(ch) / std::pow(config.l.n_bonds(), 2.);
	}
};

/*
	std::vector<numeric_t> values;
	numeric_t x = ;
	bool found = false;
	for (auto v : values)
		if (std::abs(v - x) < std::pow(10., -12.))
			found = true;
	if (!found)
		values.push_back(x);
	std::cout << values.size() << " unique values of " << std::pow(3.*bonds[0]->size(), 2) << std::endl;
	*/

struct wick_gamma_mod
{
	configuration& config;
	Random& rng;

	wick_gamma_mod(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t gm = 0.;
		double pi = 4. * std::atan(1.);
		
		/*
		std::vector<const std::vector<std::pair<int, int>>*> bonds =
			{&config.l.bonds("nn_bond_1"), &config.l.bonds("nn_bond_2"),
			&config.l.bonds("nn_bond_3")};
		std::vector<double> phases = {2.*std::sin(0. * pi), 2.*std::sin(2./3. * pi), 2.*std::sin(4./3. * pi)};
		*/
		std::vector<const std::vector<std::pair<int, int>>*> bonds =
			{&config.l.bonds("nn_bond_2"), &config.l.bonds("nn_bond_3")};
		std::vector<double> phases = {2.*std::sin(2./3. * pi), 2.*std::sin(4./3. * pi)};
		
		const int N = bonds.size(), M = bonds[0]->size();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < M; ++j)
			{
				auto& a = (*bonds[i])[j];
				for (int m = 0; m < N; ++m)
					for (int n = 0; n < M; ++n)
					{
						auto& b = (*bonds[m])[n];
						
						gm += 2.*phases[i] * phases[m]
							* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second));
							
						gm -= 2.*phases[i] * phases[m]
							* (et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(a.second, b.first) * td_gf(a.first, b.second));
						
						/*
						gm -= phases[i] * phases[m]
							* (et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
							+ config.l.parity(a.first) * config.l.parity(b.second) * td_gf(a.first, b.second) * td_gf(a.second, b.first));
							
						gm += phases[i] * phases[m]
							* (et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
							+ config.l.parity(a.second) * config.l.parity(b.second) * td_gf(a.second, b.second) * td_gf(a.first, b.first));
						*/
					}
			}
		return std::real(gm) / std::pow(config.l.n_bonds(), 2.);
	}
};

// sp(tau) = sum_ij e^{-i K (r_i - r_j)} <c_i(tau) c_j^dag>
struct wick_sp
{
	configuration& config;
	Random& rng;

	wick_sp(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t sp = 0.;
		auto& K = config.l.symmetry_point("K");
		const int N = config.l.n_sites();
		for (int i = 0; i < N; ++i)
		{
			auto& r_i = config.l.real_space_coord(i);
			for (int j = 0; j < N; ++j)
			{
				auto& r_j = config.l.real_space_coord(j);
				double kdot = K.dot(r_i - r_j);
			
				sp += std::cos(kdot) * td_gf(i, j);
			}
		}
		return std::real(sp);
	}
};

// tp(tau) = sum_ijmn e^{-i K (r_i - r_j + r_m - r_n)}
			//		<c_i(tau) c_j(tau) c_n^dag c_m^dag>
struct wick_tp
{
	configuration& config;
	Random& rng;

	wick_tp(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		numeric_t tp = 0.;
		auto& K = config.l.symmetry_point("K");
		/*
		std::vector<std::complex<double>> unique_values;
		std::vector<std::array<int, 4>> unique_sites;
		
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
				for (int k = 0; k < config.l.n_sites(); ++k)
					for (int l = 0; l < config.l.n_sites(); ++l)
					{
						auto& r_i = config.l.real_space_coord(i);
						auto& r_j = config.l.real_space_coord(j);
						auto& r_k = config.l.real_space_coord(k);
						auto& r_l = config.l.real_space_coord(l);
						int x = i % (2*config.l.L);
						int y = i / (2*config.l.L);
						double kdot = K.dot(r_i - r_j + r_k - r_l);
						
						std::complex<double> x = std::cos(kdot) * (td_gf(i, l) * td_gf(j, k) - td_gf(i, k) * td_gf(j, l));
						tp += x;
						bool exists = false;
						for (int a = 0; a < unique_values.size(); ++a)
							if (std::abs(x - unique_values[a]) < std::pow(10., -13.))
							{
								exists = true;
								break;
							}
						if (!exists)
						{
							unique_values.push_back(x);
							unique_sites.push_back({i, j, k, l});
						}
					}
		std::cout << unique_values.size() << " of " << std::pow(config.l.n_sites(), 4) << std::endl;
		for (auto& i : unique_sites)
			std::cout << i[0] << ", " << i[1] << ", " << i[2] << ", " << i[3] << std::endl;
		*/
		return std::real(tp);
	}
};
