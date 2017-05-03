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
		double M2 = 0.;
		if (config.param.decoupling == "majorana")
		{
			for (int i = 0; i < config.l.n_sites(); ++i)
				for (int j = 0; j < config.l.n_sites(); ++j)
					M2 += config.l.parity(i) * config.l.parity(j) * std::real(td_gf(j, i)
						* td_gf(j, i));
		}
		else
		{
			for (int i = 0; i < config.l.n_sites(); ++i)
				for (int j = 0; j < config.l.n_sites(); ++j)
				{
                    /*
					M2 += config.l.parity(i) * config.l.parity(j)
						* std::real((1. - et_gf_t(i, i)) * (1. - et_gf_0(j, j))
						+ config.l.parity(i) * config.l.parity(j) * td_gf(i, j) * td_gf(i, j)
						- (et_gf_t(i, i) + et_gf_0(j, j))/2. + 1./4.);
                    */
                    M2 += std::real(td_gf(i, j) * td_gf(i, j));
				}
		}
		return std::real(M2) / std::pow(config.l.n_sites(), 2.);
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
		std::complex<double> kek = 0.;
		std::array<const std::vector<std::pair<int, int>>*, 3> kek_bonds =
			{&config.l.bonds("kekule"), &config.l.bonds("kekule_2"),
			&config.l.bonds("kekule_3")};
		std::array<double, 3> factors = {-1., -1., 2.};
		if (config.param.decoupling == "majorana")
		{
			for (int i = 0; i < kek_bonds.size(); ++i)
				for (int m = 0; m < kek_bonds.size(); ++m)
					for (int j = 0; j < kek_bonds[i]->size(); ++j)
						for (int n = 0; n < kek_bonds[m]->size(); ++n)
						{
							auto& a = (*kek_bonds[i])[j];
							auto& b = (*kek_bonds[m])[n];
							
							kek += factors[i] * factors[m]
								* config.l.parity(a.first) * config.l.parity(b.first)
								* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
								+ td_gf(b.first, a.first) * td_gf(b.second, a.second));
						}
		}
		else
		{
			for (int i = 0; i < kek_bonds.size(); ++i)
				for (int m = 0; m < kek_bonds.size(); ++m)
					for (int j = 0; j < kek_bonds[i]->size(); ++j)
						for (int n = 0; n < kek_bonds[m]->size(); ++n)
						{
							auto& a = (*kek_bonds[i])[j];
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
		std::complex<double> ep = 0.;
		std::complex<double> im = {0., 1.};
		if (config.param.decoupling == "majorana")
		{
			for (auto& a : config.l.bonds("nearest neighbors"))
				for (auto& b : config.l.bonds("nearest neighbors"))
				{
					ep += config.l.parity(a.first) * config.l.parity(b.first)
						* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
						+ td_gf(b.first, a.first) * td_gf(b.second, a.second));
				}
		}
		else
		{
			for (auto& a : config.l.bonds("nearest neighbors"))
				for (auto& b : config.l.bonds("nearest neighbors"))
				{
					ep += et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second);
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
		std::complex<double> ep = 0.;
		std::vector<const std::vector<std::pair<int, int>>*> bonds =
			{&config.l.bonds("nn_bond_1"), &config.l.bonds("nn_bond_2"),
			&config.l.bonds("nn_bond_3")};
		
		if (config.param.decoupling == "majorana")
		{
			for (int i = 0; i < bonds.size(); ++i)
				for (int j = 0; j < bonds[i]->size(); ++j)
					for (int m = 0; m < bonds.size(); ++m)
						for (int n = 0; n < bonds[m]->size(); ++n)
						{
							auto& a = (*bonds[i])[j];
							auto& b = (*bonds[m])[n];
							
							ep += config.l.parity(a.first) * config.l.parity(b.first)
								* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
								+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(b.first, a.first) * td_gf(b.second, a.second));
							
							ep -= config.l.parity(a.second) * config.l.parity(b.first)
								* (et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
								+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(b.first, a.second) * td_gf(b.second, a.first));
							
							ep -= config.l.parity(a.first) * config.l.parity(b.second)
								* (et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
								+ config.l.parity(a.first) * config.l.parity(b.second) * td_gf(b.second, a.first) * td_gf(b.first, a.second));
							
							ep += config.l.parity(a.second) * config.l.parity(b.second)
								* (et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
								+ config.l.parity(a.second) * config.l.parity(b.second) * td_gf(b.second, a.second) * td_gf(b.first, a.first));
						}
		}
		else
		{
			for (int i = 0; i < bonds.size(); ++i)
				for (int j = 0; j < bonds[i]->size(); ++j)
					for (int m = 0; m < bonds.size(); ++m)
						for (int n = 0; n < bonds[m]->size(); ++n)
						{
							auto& a = (*bonds[i])[j];
							auto& b = (*bonds[m])[n];
							
							ep += et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
								+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second);
								
							ep -= et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
								+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(a.second, b.first) * td_gf(a.first, b.second);
								
							ep -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
								+ config.l.parity(a.first) * config.l.parity(b.second) * td_gf(a.first, b.second) * td_gf(a.second, b.first);
								
							ep += et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
								+ config.l.parity(a.second) * config.l.parity(b.second) * td_gf(a.second, b.second) * td_gf(a.first, b.first);
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
		std::complex<double> ch = 0.;
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
		std::complex<double> ch = 0.;
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern"))
			{
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(a.second, b.first) * td_gf(a.first, b.second);
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(a.second, b.first) * td_gf(a.first, b.second);
			}
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(a.second, b.second) * td_gf(a.first, b.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(a.second, b.first) * td_gf(a.first, b.second);
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(a.second, b.second) * td_gf(a.first, b.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(a.first, b.first) * td_gf(a.second, b.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(a.second, b.first) * td_gf(a.first, b.second);
			}
		return std::real(ch) / std::pow(config.l.n_bonds(), 2.);
	}
};

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
		std::complex<double> gm = 0.;
		double pi = 4. * std::atan(1.);
			
		std::vector<const std::vector<std::pair<int, int>>*> bonds =
			{&config.l.bonds("nn_bond_1"), &config.l.bonds("nn_bond_2"),
			&config.l.bonds("nn_bond_3")};
		std::vector<double> phases = {2.*std::sin(0. * pi), 2.*std::sin(2./3. * pi), 2.*std::sin(4./3. * pi)};
		
		if (config.param.decoupling == "majorana")
		{
			for (int i = 0; i < bonds.size(); ++i)
				for (int j = 0; j < bonds[i]->size(); ++j)
					for (int m = 0; m < bonds.size(); ++m)
						for (int n = 0; n < bonds[m]->size(); ++n)
						{
							auto& a = (*bonds[i])[j];
							auto& b = (*bonds[m])[n];
							
							gm += phases[i] * phases[m]
								* config.l.parity(a.first) * config.l.parity(b.first)
								* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
								+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(b.first, a.first) * td_gf(b.second, a.second));
							
							gm -= phases[i] * phases[m]
								* config.l.parity(a.second) * config.l.parity(b.first)
								* (et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
								+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(b.first, a.second) * td_gf(b.second, a.first));
							
							gm -= phases[i] * phases[m]
								* config.l.parity(a.first) * config.l.parity(b.second)
								* (et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
								+ config.l.parity(a.first) * config.l.parity(b.second) * td_gf(b.second, a.first) * td_gf(b.first, a.second));
							
							gm += phases[i] * phases[m]
								* config.l.parity(a.second) * config.l.parity(b.second)
								* (et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
								+ config.l.parity(a.second) * config.l.parity(b.second) * td_gf(b.second, a.second) * td_gf(b.first, a.first));
						}
		}
		else
		{
			for (int i = 0; i < bonds.size(); ++i)
				for (int j = 0; j < bonds[i]->size(); ++j)
					for (int m = 0; m < bonds.size(); ++m)
						for (int n = 0; n < bonds[m]->size(); ++n)
						{
							auto& a = (*bonds[i])[j];
							auto& b = (*bonds[m])[n];
							
							gm += phases[i] * phases[m]
								* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
								+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(a.first, b.first) * td_gf(a.second, b.second));
								
							gm -= phases[i] * phases[m]
								* (et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
								+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(a.second, b.first) * td_gf(a.first, b.second));
								
							gm -= phases[i] * phases[m]
								* (et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
								+ config.l.parity(a.first) * config.l.parity(b.second) * td_gf(a.first, b.second) * td_gf(a.second, b.first));
								
							gm += phases[i] * phases[m]
								* (et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
								+ config.l.parity(a.second) * config.l.parity(b.second) * td_gf(a.second, b.second) * td_gf(a.first, b.first));
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
		std::complex<double> sp = 0.;
		auto& K = config.l.symmetry_point("K");
		std::complex<double> im = {0., 1.};
		if (config.param.decoupling == "majorana")
		{
			for (int i = 0; i < config.l.n_sites(); ++i)
				for (int j = 0; j < config.l.n_sites(); ++j)
				{
					auto& r_i = config.l.real_space_coord(i);
					auto& r_j = config.l.real_space_coord(j);
					double kdot = K.dot(r_i - r_j);
				
					if (config.l.sublattice(i) == config.l.sublattice(j))
						sp += std::cos(kdot) * td_gf(j, i) + im * std::sin(kdot) * td_gf(j, i);
					else
						sp += config.l.parity(i) * (-im * std::cos(kdot) * td_gf(j, i)
							+ std::sin(kdot) * td_gf(j, i));
				}
		}
		else
		{
			for (int i = 0; i < config.l.n_sites(); ++i)
				for (int j = 0; j < config.l.n_sites(); ++j)
				{
					auto& r_i = config.l.real_space_coord(i);
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
		std::complex<double> tp = 0.;
		auto& K = config.l.symmetry_point("K");
		std::complex<double> im = {0., 1.};
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
