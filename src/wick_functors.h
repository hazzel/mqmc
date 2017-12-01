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
		const numeric_t *ca_et_gf_0 = et_gf_0.data(), *ca_et_gf_t = et_gf_t.data(), *ca_td_gf = td_gf.data();
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
				//M2 += td_gf(i, j) * td_gf(i, j);
				M2 += ca_td_gf[j * N + i] * ca_td_gf[j * N + i];
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
		const numeric_t *ca_et_gf_0 = et_gf_0.data(), *ca_et_gf_t = et_gf_t.data(), *ca_td_gf = td_gf.data();
		numeric_t kek = 0.;
		std::array<const std::vector<std::pair<int, int>>*, 3> single_kek_bonds =
			{&config.l.bonds("single_kekule"), &config.l.bonds("single_kekule_2"),
			&config.l.bonds("single_kekule_3")};
		std::array<const std::vector<std::pair<int, int>>*, 3> kek_bonds =
			{&config.l.bonds("kekule"), &config.l.bonds("kekule_2"),
			&config.l.bonds("kekule_3")};
		std::array<double, 3> factors = {-1., -1., 2.};
		
		const int N = kek_bonds.size(), M = single_kek_bonds[0]->size(), O = kek_bonds[0]->size(), ns = config.l.n_sites();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < M; ++j)
			{
				auto& a = (*single_kek_bonds[i])[j];
				for (int m = 0; m < N; ++m)
					for (int n = 0; n < O; ++n)
					{
						auto& b = (*kek_bonds[m])[n];
						
						/*
						kek += factors[i] * factors[m]
							* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second));
						*/
						
						kek += factors[i] * factors[m]
							* (ca_et_gf_t[a.first*ns + a.second] * ca_et_gf_0[b.second*ns + b.first]
							+ config.l.parity(a.first) * config.l.parity(b.first) * ca_td_gf[b.first*ns + a.first] * ca_td_gf[b.second*ns + a.second]);
					}
			}
		return std::real(2.*kek) / std::pow(config.l.n_bonds(), 2.);
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
		const numeric_t *ca_et_gf_0 = et_gf_0.data(), *ca_et_gf_t = et_gf_t.data(), *ca_td_gf = td_gf.data();
		numeric_t ep = 0.;
		auto& single_bonds = config.l.bonds("single_d1_bonds");
		auto& bonds = config.l.bonds("nearest neighbors");
		const int N = single_bonds.size(), M = bonds.size(), ns = config.l.n_sites();
		for (int i = 0; i < N; ++i)
		{
			auto& a = single_bonds[i];
			for (int j = 0; j < M; ++j)
			{
				auto& b = bonds[j];
				
				/*
				ep += et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second);
				*/
				
				ep += ca_et_gf_t[a.first*ns + a.second] * ca_et_gf_0[b.second*ns + b.first]
					+ config.l.parity(a.first) * config.l.parity(b.first) * ca_td_gf[b.first*ns + a.first] * ca_td_gf[b.second*ns + a.second];
				
			}
		}
		return std::real(2.*ep) / std::pow(N, 2.);
	}
};

struct wick_epsilon_V
{
	configuration& config;
	Random& rng;
	const numeric_t* ca_et_gf_0;
	const numeric_t* ca_et_gf_t;
	const numeric_t* ca_td_gf;

	wick_epsilon_V(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	numeric_t evaluate(Eigen::Matrix<numeric_t, 4, 4>& mat44, int ns, int i, int j, int k, int l)
	{
		const double pi = config.l.parity(i), pj = config.l.parity(j), pk = config.l.parity(k), pl = config.l.parity(l);
		double delta_ij = (i==j) ? 1.0 : 0.0;
		double delta_ki = (k==i) ? 1.0 : 0.0;
		double delta_kj = (k==j) ? 1.0 : 0.0;
		double delta_li = (l==i) ? 1.0 : 0.0;
		double delta_lj = (l==j) ? 1.0 : 0.0;
		double delta_lk = (l==k) ? 1.0 : 0.0;
		
		mat44(0, 1) = -pi * pj * ca_et_gf_t[j*ns+i];
		mat44(1, 0) = ca_et_gf_t[j*ns+i];
		
		mat44(0, 2) = -pi * pk * ca_td_gf[k*ns+i];
		mat44(2, 0) = ca_td_gf[k*ns+i];
		
		mat44(0, 3) = -pi * pl * ca_td_gf[l*ns+i];
		mat44(3, 0) = ca_td_gf[l*ns+i];
		
		mat44(1, 2) = -pj * pk * ca_td_gf[k*ns+j];
		mat44(2, 1) = ca_td_gf[k*ns+j];
		
		mat44(1, 3) = -pj * pl * ca_td_gf[l*ns+j];
		mat44(3, 1) = ca_td_gf[l*ns+j];
		
		mat44(2, 3) = -pk * pl * ca_et_gf_0[l*ns+k];
		mat44(3, 2) = ca_et_gf_0[l*ns+k];
		
		mat44(0, 0) = ca_et_gf_t[i*ns+i];
		mat44(1, 1) = ca_et_gf_t[j*ns+j];
		mat44(2, 2) = ca_et_gf_0[k*ns+k];
		mat44(3, 3) = ca_et_gf_0[l*ns+l];
		
		return mat44.determinant();
	}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		ca_et_gf_0 = et_gf_0.data();
		ca_et_gf_t = et_gf_t.data();
		ca_td_gf = td_gf.data();
		numeric_t ep = 0.;
		auto& single_bonds = config.l.bonds("single_d1_bonds");
		const int N = single_bonds.size(), ns = config.l.n_sites();
		
		for (int s = 0; s < N; ++s)
		{
			int i = single_bonds[s].first, j = single_bonds[s].second;
			const double pi = config.l.parity(i), pj = config.l.parity(j);
			#pragma distribute_point
			for (int t = 0; t < N; ++t)
			{
				int m = single_bonds[t].first, n = single_bonds[t].second;
				const double pm = config.l.parity(m), pn = config.l.parity(n);
				
				ep += pi*pi*ca_et_gf_t[i*ns+i] * (pj*pj*ca_et_gf_t[j*ns+j] * (pm*pm*ca_et_gf_0[m*ns+m] * (pn*pn*ca_et_gf_0[n*ns+n]) + pm*pn*ca_et_gf_0[n*ns+m] * (ca_et_gf_0[n*ns+m])) + pj*pm*ca_td_gf[m*ns+j] * (ca_td_gf[m*ns+j] * (pn*pn*ca_et_gf_0[n*ns+n]) + (-ca_td_gf[n*ns+j]) * (pm*pn*ca_et_gf_0[n*ns+m])) + pj*pn*ca_td_gf[n*ns+j] * (ca_td_gf[m*ns+j] * (ca_et_gf_0[n*ns+m]) + ca_td_gf[n*ns+j] * (pm*pm*ca_et_gf_0[m*ns+m])));
				ep += pi*pj*ca_et_gf_t[j*ns+i] * (ca_et_gf_t[j*ns+i] * (pm*pm*ca_et_gf_0[m*ns+m] * (pn*pn*ca_et_gf_0[n*ns+n]) + pm*pn*ca_et_gf_0[n*ns+m] * (ca_et_gf_0[n*ns+m])) + (-ca_td_gf[m*ns+i]) * (pj*pm*ca_td_gf[m*ns+j] * (pn*pn*ca_et_gf_0[n*ns+n]) + pj*pn*ca_td_gf[n*ns+j] * (ca_et_gf_0[n*ns+m])) + (-ca_td_gf[n*ns+i]) * ((-pj*pm*ca_td_gf[m*ns+j]) * (pm*pn*ca_et_gf_0[n*ns+m]) + pj*pn*ca_td_gf[n*ns+j] * (pm*pm*ca_et_gf_0[m*ns+m])));
				ep += pi*pm*ca_td_gf[m*ns+i] * (ca_et_gf_t[j*ns+i] * (ca_td_gf[m*ns+j] * (pn*pn*ca_et_gf_0[n*ns+n]) + (-ca_td_gf[n*ns+j]) * (pm*pn*ca_et_gf_0[n*ns+m])) + ca_td_gf[m*ns+i] * (pj*pj*ca_et_gf_t[j*ns+j] * (pn*pn*ca_et_gf_0[n*ns+n]) + pj*pn*ca_td_gf[n*ns+j] * (ca_td_gf[n*ns+j])) + (-ca_td_gf[n*ns+i]) * (pj*pj*ca_et_gf_t[j*ns+j] * (pm*pn*ca_et_gf_0[n*ns+m]) + pj*pn*ca_td_gf[n*ns+j] * (ca_td_gf[m*ns+j])));
				ep += pi*pn*ca_td_gf[n*ns+i] * (ca_et_gf_t[j*ns+i] * (ca_td_gf[m*ns+j] * (ca_et_gf_0[n*ns+m]) + ca_td_gf[n*ns+j] * (pm*pm*ca_et_gf_0[m*ns+m])) + ca_td_gf[m*ns+i] * (pj*pj*ca_et_gf_t[j*ns+j] * (ca_et_gf_0[n*ns+m]) + (-pj*pm*ca_td_gf[m*ns+j]) * (ca_td_gf[n*ns+j])) + ca_td_gf[n*ns+i] * (pj*pj*ca_et_gf_t[j*ns+j] * (pm*pm*ca_et_gf_0[m*ns+m]) + pj*pm*ca_td_gf[m*ns+j] * (ca_td_gf[m*ns+j])));
				
				/*
				ep += pi*pj*ca_et_gf_t[j*ns+i] * (ca_et_gf_t[j*ns+i] * pm*pn*ca_et_gf_0[n*ns+m] * (ca_et_gf_0[n*ns+m]) + (-ca_td_gf[m*ns+i]) * pj*pn*ca_td_gf[n*ns+j] * (ca_et_gf_0[n*ns+m]) + (-ca_td_gf[n*ns+i]) * (-pj*pm*ca_td_gf[m*ns+j]) * (pm*pn*ca_et_gf_0[n*ns+m]));
				ep += pi*pm*ca_td_gf[m*ns+i] * (ca_et_gf_t[j*ns+i] * (-ca_td_gf[n*ns+j]) * (pm*pn*ca_et_gf_0[n*ns+m]) + ca_td_gf[m*ns+i] * (pj*pn*ca_td_gf[n*ns+j] * (ca_td_gf[n*ns+j])) + (-ca_td_gf[n*ns+i]) * pj*pn*ca_td_gf[n*ns+j] * (ca_td_gf[m*ns+j]));
				ep += pi*pn*ca_td_gf[n*ns+i] * (ca_et_gf_t[j*ns+i] * ca_td_gf[m*ns+j] * (ca_et_gf_0[n*ns+m]) + ca_td_gf[m*ns+i] * (pj*pj*ca_et_gf_t[j*ns+j] * (ca_et_gf_0[n*ns+m]) + (-pj*pm*ca_td_gf[m*ns+j]) * (ca_td_gf[n*ns+j])) + ca_td_gf[n*ns+i] * pj*pm*ca_td_gf[m*ns+j] * (ca_td_gf[m*ns+j]));
				*/
			}
		}
		
		/*
		Eigen::Matrix<numeric_t, 4, 4> mat44 = Eigen::Matrix<numeric_t, 4, 4>::Zero();
		for (int s = 0; s < N; ++s)
		{
			int i = single_bonds[s].first, j = single_bonds[s].second;
			//#pragma distribute_point
			for (int t = 0; t < N; ++t)
			{
				int m = single_bonds[t].first, n = single_bonds[t].second;
				ep += evaluate(mat44, ns, i, j, m, n);
			}
		}
		*/
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
		const numeric_t *ca_et_gf_0 = et_gf_0.data(), *ca_et_gf_t = et_gf_t.data(), *ca_td_gf = td_gf.data();
		numeric_t ep = 0.;
		std::vector<const std::vector<std::pair<int, int>>*> bonds =
			{&config.l.bonds("nn_bond_1"), &config.l.bonds("nn_bond_2"),
			&config.l.bonds("nn_bond_3")};
		
		const int N = bonds.size(), M = bonds[0]->size(), ns = config.l.n_sites();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < M; ++j)
			{
				auto& a = (*bonds[i])[j];
				for (int m = 0; m < N; ++m)
					for (int n = 0; n < M; ++n)
					{
						auto& b = (*bonds[m])[n];
						
						/*
						ep += 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second));
							
						ep -= 2.*(et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(a.second, b.first) * td_gf(a.first, b.second));
						*/
						ep += 2.*(ca_et_gf_t[a.first*ns+a.second] * ca_et_gf_0[b.second*ns+b.first]
							+ config.l.parity(a.first) * config.l.parity(b.first) * ca_td_gf[b.first*ns+a.first] * ca_td_gf[b.second*ns+a.second]);
							
						ep -= 2.*(ca_et_gf_t[a.second*ns+a.first] * ca_et_gf_0[b.second*ns+b.first]
							+ config.l.parity(a.second) * config.l.parity(b.first) * ca_td_gf[b.first*ns+a.second] * ca_td_gf[b.second*ns+a.first]);
						
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

	wick_chern(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		const numeric_t *ca_et_gf_0 = et_gf_0.data(), *ca_et_gf_t = et_gf_t.data(), *ca_td_gf = td_gf.data();
		numeric_t ch = 0.;
		auto& bonds_c1 = config.l.bonds("chern");
		auto& bonds_c2 = config.l.bonds("chern_2");
		const int N = bonds_c1.size(), ns = config.l.n_sites();
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
				/*
				ch -= 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first));
				*/
				ch -= 2.*(ca_et_gf_t[a.first*ns+a.second] * ca_et_gf_0[b.first*ns+b.second]
					+ ca_td_gf[b.second*ns+a.first] * ca_td_gf[b.first*ns+a.second]
					- ca_et_gf_t[a.second*ns+a.first] * ca_et_gf_0[b.first*ns+b.second]
					- ca_td_gf[b.second*ns+a.second] * ca_td_gf[b.first*ns+a.first]);
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
				/*
				ch += 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first));
				*/
				ch += 2.*(ca_et_gf_t[a.first*ns+a.second] * ca_et_gf_0[b.first*ns+b.second]
					- ca_td_gf[b.second*ns+a.first] * ca_td_gf[b.first*ns+a.second]
					- ca_et_gf_t[a.second*ns+a.first] * ca_et_gf_0[b.first*ns+b.second]
					+ ca_td_gf[b.second*ns+a.second] * ca_td_gf[b.first*ns+a.first]);
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
				/*
				ch += 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first));
				*/
				ch += 2.*(ca_et_gf_t[a.first*ns+a.second] * ca_et_gf_0[b.first*ns+b.second]
					- ca_td_gf[b.second*ns+a.first] * ca_td_gf[b.first*ns+a.second]
					- ca_et_gf_t[a.second*ns+a.first] * ca_et_gf_0[b.first*ns+b.second]
					+ ca_td_gf[b.second*ns+a.second] * ca_td_gf[b.first*ns+a.first]);
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
				
				/*
				ch -= 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first));
				*/
				ch -= 2.*(ca_et_gf_t[a.first*ns+a.second] * ca_et_gf_0[b.first*ns+b.second]
					+ ca_td_gf[b.second*ns+a.first] * ca_td_gf[b.first*ns+a.second]
					- ca_et_gf_t[a.second*ns+a.first] * ca_et_gf_0[b.first*ns+b.second]
					- ca_td_gf[b.second*ns+a.second] * ca_td_gf[b.first*ns+a.first]);
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
		const numeric_t *ca_et_gf_0 = et_gf_0.data(), *ca_et_gf_t = et_gf_t.data(), *ca_td_gf = td_gf.data();
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
		
		const int N = bonds.size(), M = bonds[0]->size(), ns = config.l.n_sites();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < M; ++j)
			{
				auto& a = (*bonds[i])[j];
				for (int m = 0; m < N; ++m)
					for (int n = 0; n < M; ++n)
					{
						auto& b = (*bonds[m])[n];
						
						/*
						gm += 2.*phases[i] * phases[m]
							* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second));
							
						gm -= 2.*phases[i] * phases[m]
							* (et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(a.second, b.first) * td_gf(a.first, b.second));
						*/
						gm += 2.*phases[i] * phases[m]
							* (ca_et_gf_t[a.first*ns+a.second] * ca_et_gf_0[b.second*ns+b.first]
							+ config.l.parity(a.first) * config.l.parity(b.first) * ca_td_gf[b.first*ns+a.first] * ca_td_gf[b.second*ns+a.second]);
							
						gm -= 2.*phases[i] * phases[m]
							* (ca_et_gf_t[a.second*ns+a.first] * ca_et_gf_0[b.second*ns+b.first]
							+ config.l.parity(a.second) * config.l.parity(b.first) * ca_td_gf[b.first*ns+a.second] * ca_td_gf[b.second*ns+a.first]);
						
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
		const numeric_t *ca_td_gf = td_gf.data();
		numeric_t sp = 0.;
		auto& K = config.l.symmetry_point("K");
		const int N = config.l.n_sites();
		for (int i = 0; i < N; ++i)
		{
			auto& r_i = config.l.real_space_coord((i/2)*2);
			for (int j = 0; j < N; ++j)
			{
				auto& r_j = config.l.real_space_coord((j/2)*2);
				double kdot = K.dot(r_i - r_j);
				sp += std::cos(kdot) * ca_td_gf[j*N+i] * config.l.parity(i)*config.l.parity(j);
				//sp += std::cos(kdot) * ca_td_gf[j*N+i] * (1. + config.l.parity(i)*config.l.parity(j));
			}
		}
		return std::real(sp) / std::pow(N, 2.);
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
		const numeric_t *ca_td_gf = td_gf.data();
		numeric_t tp = 0.;
		auto& K = config.l.symmetry_point("K");
		const int N = config.l.n_sites();
		
		//std::vector<numeric_t> unique_values;
		//std::vector<std::array<int, 4>> unique_sites;
		
		//for (int i = 0; i < N; ++i)
		//for (int n = 0; n < 10; ++n)
		{
			int i = rng() * N;
			{
				auto& r_i = config.l.real_space_coord(i);
				//for (int j = 0; j < N; ++j)
				int j = rng() * N;
				{
					auto& r_j = config.l.real_space_coord(j);
					for (int k = 0; k < N; ++k)
					{
						auto& r_k = config.l.real_space_coord(k);
						for (int l = 0; l < N; ++l)
						{
							auto& r_l = config.l.real_space_coord(l);
							double kdot = K.dot(r_i - r_j + r_k - r_l);
							tp += std::cos(kdot) * (ca_td_gf[i*N+l] * ca_td_gf[j*N+k] - ca_td_gf[i*N+k] * ca_td_gf[j*N+l])
								* config.l.parity(i) * config.l.parity(j) * config.l.parity(k) * config.l.parity(l);
							
							//tp += std::cos(kdot) * (ca_td_gf[i*N+l] * ca_td_gf[j*N+k] - ca_td_gf[i*N+k] * ca_td_gf[j*N+l])
							//	* (1. + config.l.parity(i) * config.l.parity(j) * config.l.parity(k) * config.l.parity(l));
							/*
							numeric_t x = std::cos(kdot) * (td_gf(l, i) * td_gf(k, j) - td_gf(k, i) * td_gf(l, j));
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
							*/
						}
					}
				}
			}
		}
		//std::cout << unique_values.size() << " of " << std::pow(config.l.n_sites(), 4) << std::endl;
		//for (auto& i : unique_sites)
		//	std::cout << i[0] << ", " << i[1] << ", " << i[2] << ", " << i[3] << std::endl;
		
		return std::real(tp) / std::pow(N, 2.);
	}
};
