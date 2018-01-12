#pragma once
#include <ostream>
#include <vector>
#include <cmath>
#include "measurements.h"
#include "parser.h"
#include "configuration.h"

void eval_B_cdw(double& out,
	std::vector<std::valarray<double>*>& o)
{
	out = (*o[1])[0] / ((*o[0])[0] * (*o[0])[0]);
}

void eval_R_cdw(double& out,
	std::vector<std::valarray<double>*>& o)
{
	out = 1. - (*o[1])[0] / (*o[0])[0];
}

void eval_B_chern(double& out,
	std::vector<std::valarray<double>*>& o)
{
	out = (*o[1])[0] / ((*o[0])[0] * (*o[0])[0]);
}

void eval_R_chern(double& out,
	std::vector<std::valarray<double>*>& o)
{
	out = 1. - (*o[1])[0] / (*o[0])[0];
}

void eval_epsilon(std::valarray<double>& out,
	std::vector<std::valarray<double>*>& o)
{
	std::valarray<double>* ep_tau = o[0];
	double epsilon = (*o[1])[0];
	out.resize(ep_tau->size());
	for (int i = 0; i < ep_tau->size(); ++i)
		out[i] = (*ep_tau)[i] - epsilon * epsilon;
}

void eval_log_ratio(std::valarray<double>& out,
	std::vector<std::valarray<double>*>& o)
{
	int N = 1;
	std::valarray<double>* c_tau = o[0];
	out.resize(c_tau->size() - N);
	for (int i = 0; i < c_tau->size() - N; ++i)
		out[i] = std::log((*c_tau)[i] / (*c_tau)[i+N]);
}

void eval_n(double& out,
	std::vector<std::valarray<double>*>& o)
{
	double sign_re = (*o[0])[0];
	double sign_im = (*o[1])[0];
	double n_re = (*o[2])[0];
	double n_im = (*o[3])[0];
	out = (n_re * sign_re + n_im * sign_im)
		/ (sign_re * sign_re + sign_im * sign_im);
}

void eval_sign(double& out,
	std::vector<std::valarray<double>*>& o)
{
	double sign_re = (*o[0])[0];
	double sign_im = (*o[1])[0];
	out = std::sqrt(sign_re * sign_re + sign_im * sign_im);
}

struct measure_M
{
	configuration& config;
	parser& pars;
	typedef fast_update<arg_t>::numeric_t numeric_t;

	void perform()
	{
		/*
		std::vector<double> c(config.l.max_distance() + 1, 0.);
		std::complex<double> energy = 0., m2 = 0., ep = 0., chern = 0.;
		std::complex<double> n = 0.;
		config.M.static_measure(c, n, energy, m2, ep, chern);
		for (int i = 0; i < c.size(); ++i)
			c[i] /= config.shellsize[i];
		if (config.param.mu != 0 || config.param.stag_mu != 0)
		{
			config.measure.add("n_re", std::real(n*config.param.sign_phase));
			config.measure.add("n_im", std::imag(n*config.param.sign_phase));
			config.measure.add("n", std::real(n));
		}
		config.measure.add("sign_phase_re", std::real(config.param.sign_phase));
		config.measure.add("sign_phase_im", std::imag(config.param.sign_phase));
		config.measure.add("energy", std::real(energy));
		config.measure.add("M2", std::real(m2));
		config.measure.add("epsilon", std::real(ep));
		config.measure.add("chern", std::real(chern));
		config.measure.add("corr", c);
		*/
	}

	void collect(std::ostream& os)
	{
		//if (config.param.mu != 0 || config.param.stag_mu != 0)
			//config.measure.add_evalable("n_jack", "sign_phase_re", "sign_phase_im", "n_re", "n_im", eval_n);
		config.measure.add_evalable("sign_jack", "sign_phase_re", "sign_phase_im", eval_sign);
		if (contains("M2") && contains("M4"))
			config.measure.add_evalable("B_cdw", "M2", "M4", eval_B_cdw);
		if (contains("M2") && contains("S_cdw_q"))
			config.measure.add_evalable("R_cdw", "M2", "S_cdw_q", eval_R_cdw);
		if (contains("chern2") && contains("chern4"))
			config.measure.add_evalable("B_chern", "chern2", "chern4", eval_B_chern);
		if (contains("chernAA") && contains("S_chernAA_q"))
			config.measure.add_evalable("R_chernAA", "chernAA", "S_chernAA_q", eval_R_chern);
		if (contains("chernBB") && contains("S_chernBB_q"))
			config.measure.add_evalable("R_chernBB", "chernBB", "S_chernBB_q", eval_R_chern);
		
		if (config.param.n_discrete_tau > 0)
			for (int i = 0; i < config.param.obs.size(); ++i)
			{
				if (config.param.obs[i] == "epsilon")
					config.measure.add_vectorevalable("dyn_epjack_tau", "dyn_epsilon_tau", "epsilon", eval_epsilon);
			}
		
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		config.measure.get_statistics(os);
	}
	
	bool contains(const std::string& name)
	{
		return std::find(config.param.static_obs.begin(), config.param.static_obs.end(), name) != config.param.static_obs.end();
	}
};
