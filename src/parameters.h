#pragma once
#include <complex>
#include <string>

struct parameters
{
	double beta, dtau, t, tprime, V, mu, lambda;
	int L, n_delta, n_tau_slices, n_discrete_tau, n_dyn_tau;
	std::string method, geometry;
	bool use_projector;
	std::vector<std::string> obs, static_obs;
};