#pragma once
#include <map>
#include <vector>
#include "measurements.h"
#include "configuration.h"
#include "fast_update.h"
#include "wick_base.h"
#include "wick_functors.h"
#include "wick_static_base.h"
#include "wick_static_functors.h"
#include "vector_wick_static_base.h"
#include "vector_wick_static_functors.h"

struct event_build
{
	configuration& config;
	Random& rng;

	void trigger()
	{
		boost::multi_array<arg_t, 3> initial_vertices(boost::extents[config.param.n_flavor]
			[config.param.n_flavor][config.param.n_tau_slices]);
		for (int i = 0; i < initial_vertices.shape()[0]; ++i)
			for (int j = 0; j < initial_vertices.shape()[1]; ++j)
				for (int k = 0; k < initial_vertices.shape()[2]; ++k)
				{
					initial_vertices[i][j][k] = arg_t(config.l.n_bonds() / sizeof(int) / 8 + 1);
					for (int m = 0; m < config.l.n_sites(); ++m)
						for (int n = m; n < config.l.n_sites(); ++n)
							if (config.l.distance(m, n) == 1)
								if (rng() < 0.5)
									initial_vertices[i][j][k].set(config.M.bond_index(m, n));
				}
		config.M.build(initial_vertices);
	}
};

struct event_flip_all
{
	configuration& config;
	Random& rng;

	/*
	void flip_cb(int bond_type, int alpha = 0, int beta = 0)
	{
		int cnt = 0;
		for (auto& b : config.M.get_cb_bonds(bond_type))
		{
			if (b.first > b.second) continue;
			int s = 0;
			std::complex<double> p_0 = config.M.try_ising_flip(b.first, b.second, alpha, beta);
			//if (config.param.mu != 0 || config.param.stag_mu != 0)
			config.param.sign_phase *= std::exp(std::complex<double>(0, std::arg(p_0)));
			if (rng() < std::abs(p_0))
			{
				config.M.update_equal_time_gf_after_flip();
				config.M.flip_spin(b, alpha, beta);
			}
		}
	}

	void trigger()
	{
		if (config.param.V > 0.)
		{
			if (config.param.multiply_T)
				config.M.prepare_flip();

			for (int alpha = 0; alpha < config.param.n_flavor; ++alpha)
				for (int beta = 0; beta < config.param.n_flavor; ++beta)
				{
					config.M.set_partial_vertex(0);
					for (int bt = 0; bt < config.M.n_cb_bonds(); ++bt)
					{
						config.M.partial_advance(bt, alpha, beta);
						flip_cb(bt, alpha, beta);
					}
				}

			for (int alpha = config.param.n_flavor - 1; alpha >= 0; --alpha)
				for (int beta = config.param.n_flavor - 1; beta >= 0; --beta)
				{
					config.M.set_partial_vertex(config.M.n_cb_bonds() - 1);
					config.M.partial_advance(0, alpha, beta);
				}
			if (config.param.multiply_T)
				config.M.prepare_measurement();
		}
	}
	*/
	
	void flip_cb(int bond_type, int alpha, int beta)
	{
		if (config.param.direction == 1)
			config.M.multiply_Gamma_matrix(bond_type, alpha, beta);
		for (auto& bond : config.M.get_nn_bonds(bond_type))
		{
			std::complex<double> p_0 = config.M.try_ising_flip(bond.first, bond.second, alpha, beta);
			config.param.sign_phase *= std::exp(std::complex<double>(0, std::arg(p_0)));
			if (rng() < std::abs(p_0))
			{
				config.M.update_equal_time_gf_after_flip();
				config.M.flip_spin({bond.first, bond.second});
			}
		}
		if (config.param.direction == -1)
			config.M.multiply_Gamma_matrix(bond_type, alpha, beta);
	}

	void trigger()
	{
		if (config.param.V > 0.)
		{
			if (config.param.direction == 1)
			{
				config.M.update_tau();
				config.M.multiply_T_matrix();
				
				for (int alpha = 0; alpha < config.param.n_flavor; ++alpha)
					for (int beta = 0; beta < config.param.n_flavor; ++beta)
						for (int bt = config.M.n_cb_bonds() - 1; bt >= 0; --bt)
							flip_cb(bt, alpha, beta);
						
				
				
				//config.M.multiply_T_matrix();
			}
			else if (config.param.direction == -1)
			{
				//config.M.multiply_T_matrix();
				
				
				
				
				for (int alpha = config.param.n_flavor - 1; alpha >= 0; --alpha)
					for (int beta = config.param.n_flavor - 1; beta >= 0; --beta)
						for (int bt = 0; bt < config.M.n_cb_bonds(); ++bt)
							flip_cb(bt, alpha, beta);
				
				config.M.update_tau();
				config.M.multiply_T_matrix();
			}
		}
	}
};

struct event_static_measurement
{
	typedef fast_update<arg_t>::dmatrix_t matrix_t;

	configuration& config;
	Random& rng;
	std::vector<wick_static_base<matrix_t>> obs;
	std::vector<vector_wick_static_base<matrix_t>> vec_obs;
	std::vector<std::string> names;
	std::vector<std::string> vec_names;

	event_static_measurement(configuration& config_, Random& rng_,
		int n_prebin, const std::vector<std::string>& observables)
		: config(config_), rng(rng_)
	{
		obs.reserve(100);
		vec_obs.reserve(100);
		for (int i = 0; i < observables.size(); ++i)
		{
			if (observables[i] == "energy")
				add_wick(wick_static_energy{config, rng}, observables[i]);
			else if (observables[i] == "h_t")
				add_wick(wick_static_h_t{config, rng}, observables[i]);
			else if (observables[i] == "h_v")
				add_wick(wick_static_h_v{config, rng}, observables[i]);
			else if (observables[i] == "h_mu")
				add_wick(wick_static_h_mu{config, rng}, observables[i]);
			else if (observables[i] == "M2")
				add_wick(wick_static_M2{config, rng}, observables[i]);
			else if (observables[i] == "S_cdw_q")
				add_wick(wick_static_S_cdw_q{config, rng}, observables[i]);
			else if (observables[i] == "M4")
				add_wick(wick_static_M4{config, rng}, observables[i]);
			else if (observables[i] == "epsilon")
				add_wick(wick_static_epsilon{config, rng}, observables[i]);
			else if (observables[i] == "epsilon_V")
				add_wick(wick_static_epsilon_V{config, rng}, observables[i]);
			else if (observables[i] == "kekule")
				add_wick(wick_static_kek{config, rng}, observables[i]);
			else if (observables[i] == "chern2")
				add_wick(wick_static_chern2{config, rng}, observables[i]);
			else if (observables[i] == "S_chern_q")
				add_wick(wick_static_S_chern_q{config, rng}, observables[i]);
			else if (observables[i] == "chernAA")
				add_wick(wick_static_chernAA{config, rng, config.l.bonds("chern")}, observables[i]);
			else if (observables[i] == "S_chernAA_q")
			{
				Eigen::Vector2d delta = {1., 0};
				add_wick(wick_static_S_chernAA_q{config, rng, config.l.bonds("chern"), delta}, observables[i]);
			}
			else if (observables[i] == "chernBB")
				add_wick(wick_static_chernAA{config, rng, config.l.bonds("chern_2")}, observables[i]);
			else if (observables[i] == "S_chernBB_q")
			{
				Eigen::Vector2d delta = {0.5, -std::sqrt(3.)/2.};
				add_wick(wick_static_S_chernAA_q{config, rng, config.l.bonds("chern_2"), delta}, observables[i]);
			}
			else if (observables[i] == "S_chernAA")
			{
				Eigen::Vector2d delta = {1., 0};
				add_vector_wick(wick_static_S_chernAA{config, rng, config.l.bonds("chern"), delta}, observables[i]);
			}
			else if (observables[i] == "S_chernBB")
			{
				Eigen::Vector2d delta = {0.5, -std::sqrt(3.)/2.};
				add_vector_wick(wick_static_S_chernAA{config, rng, config.l.bonds("chern_2"), delta}, observables[i]);
			}
			else if (observables[i] == "S_chern_real_space")
			{
				add_vector_wick(wick_static_S_chern_real_space{config, rng, config.l.bonds("chern")}, observables[i]);
			}
			else if (observables[i] == "chern4")
				add_wick(wick_static_chern4{config, rng}, observables[i]);
		}
	}

	template<typename T>
	void add_wick(T&& functor, const std::string& name)
	{
		obs.push_back(wick_static_base<matrix_t>(std::forward<T>(functor)));
		names.push_back(name);
	}
	
	template<typename T>
	void add_vector_wick(T&& functor, const std::string& name)
	{
		vec_obs.push_back(vector_wick_static_base<matrix_t>(std::forward<T>(functor)));
		vec_names.push_back(name);
	}

	void trigger()
	{
		config.M.measure_static_observable(names, obs, vec_names, vec_obs);
		config.measure.add("sign_phase_re", std::real(config.param.sign_phase));
		config.measure.add("sign_phase_im", std::imag(config.param.sign_phase));
	}
};

struct event_dynamic_measurement
{
	typedef fast_update<arg_t>::dmatrix_t matrix_t;

	configuration& config;
	Random& rng;
	std::vector<std::vector<double>> dyn_tau;
	std::vector<wick_base<matrix_t>> obs;
	std::vector<std::string> names;

	event_dynamic_measurement(configuration& config_, Random& rng_,
		int n_prebin, const std::vector<std::string>& observables)
		: config(config_), rng(rng_)
	{
		obs.reserve(100);
		for (int i = 0; i < observables.size(); ++i)
		{
			dyn_tau.push_back(std::vector<double>(2*config.param.n_discrete_tau+1,
				0.));

			if (observables[i] == "M2")
				add_wick(wick_M2{config, rng});
			else if (observables[i] == "kekule")
				add_wick(wick_kekule{config, rng});
			else if (observables[i] == "epsilon")
				add_wick(wick_epsilon{config, rng});
			else if (observables[i] == "epsilon_V")
				add_wick(wick_epsilon_V{config, rng});
			else if (observables[i] == "epsilon_as")
				add_wick(wick_epsilon_as{config, rng});
			else if (observables[i] == "cdw_s")
				add_wick(wick_cdw_s{config, rng});
			else if (observables[i] == "chern")
				add_wick(wick_chern{config, rng});
			else if (observables[i] == "gamma_mod")
				add_wick(wick_gamma_mod{config, rng});
			else if (observables[i] == "sp")
				add_wick(wick_sp{config, rng});
			else if (observables[i] == "tp")
				add_wick(wick_tp{config, rng});

			names.push_back("dyn_"+observables[i]);
		}
	}

	template<typename T>
	void add_wick(T&& functor)
	{
		obs.push_back(wick_base<matrix_t>(std::forward<T>(functor)));
	}

	void trigger()
	{
		if (config.param.n_discrete_tau == 0)
			return;
		for (int i = 0; i < dyn_tau.size(); ++i)
			std::fill(dyn_tau[i].begin(), dyn_tau[i].end(), 0.);
		config.M.measure_dynamical_observable(dyn_tau, obs);

		for (int i = 0; i < dyn_tau.size(); ++i)
			if (config.param.n_discrete_tau > 0)
				config.measure.add(names[i]+"_tau", dyn_tau[i]);
	}
};
