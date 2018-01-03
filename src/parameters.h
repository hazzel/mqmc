#pragma once
#include <complex>
#include <string>

struct parameters
{
	double beta, dtau, t, tprime, V, W, mu, stag_mu, gamma, lambda, kappa, epsilon;
	double inv_symmetry, ph_symmetry;
	int n_flavor, Lx, Ly, n_delta, n_tau_slices, n_discrete_tau, n_dyn_tau, n_rebuild, direction, slater;
	std::string method, geometry, decoupling;
	bool use_projector, multiply_T;
	std::vector<std::string> obs, static_obs;
	std::complex<double> sign_phase=1.;
};
