#pragma once
#include <vector>
#include <array>
#include <complex>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "dump.h"
#include "lattice.h"
#include "measurements.h"
#include "parameters.h"
#include "qr_stabilizer.h"
#include "wick_base.h"
#include "wick_static_base.h"

template <class data_t, class index_t>
class SortIndicesInc
{
	protected:
		const data_t& data;
	public:
		SortIndicesInc(const data_t& data_) : data(data_) {}
		bool operator()(const index_t& i, const index_t& j) const
		{
			return data[i] < data[j];
		}
};

template<typename arg_t>
class fast_update
{
	public:
		//using numeric_t = std::complex<double>;
		using numeric_t = double;
		template<int n, int m>
		using matrix_t = Eigen::Matrix<numeric_t, n, m, Eigen::ColMajor>;
		using dmatrix_t = matrix_t<Eigen::Dynamic, Eigen::Dynamic>;
		using stabilizer_t = qr_stabilizer<numeric_t, dmatrix_t>;
		using row_matrix_t = Eigen::Matrix<numeric_t, Eigen::Dynamic,
			Eigen::Dynamic, Eigen::RowMajor>;
		using col_matrix_t = Eigen::Matrix<numeric_t, Eigen::Dynamic,
			Eigen::Dynamic, Eigen::ColMajor>;

		fast_update(Random& rng_, const lattice& l_, parameters& param_,
			measurements& measure_)
			: rng(rng_), l(l_), param(param_), measure(measure_), tau(1),
				update_time_displaced_gf(false),
				stabilizer{measure, equal_time_gf, time_displaced_gf,
				proj_W_l, proj_W_r, proj_W}
		{}

		void serialize(odump& out)
		{
			/*
			int size = aux_spins.size();
			int cnt = 0;
			out.write(size);
			for (arg_t& v : aux_spins)
			{
				v.serialize(out);
				++cnt;
			}
			*/
			int size = aux_spins.num_elements();
			out.write(size);
			for (int i = 0; i < aux_spins.shape()[0]; ++i)
				for (int j = 0; j < aux_spins.shape()[1]; ++j)
					for (int k = 0; k < aux_spins.shape()[2]; ++k)
						aux_spins[i][j][k].serialize(out);
		}

		void serialize(idump& in)
		{
			/*
			int size; in.read(size);
			aux_spins.resize(size);
			for (int i = 0; i < size; ++i)
			{
				arg_t v;
				v.serialize(in);
				aux_spins[i] = v;
			}
			max_tau = size;
			tau = max_tau;
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, n_matrix_size);
			rebuild();
			*/
			int size; in.read(size);
			aux_spins.resize(boost::extents[param.n_flavor][param.n_flavor][param.n_tau_slices]);
			for (int i = 0; i < aux_spins.shape()[0]; ++i)
				for (int j = 0; j < aux_spins.shape()[1]; ++j)
					for (int k = 0; k < aux_spins.shape()[2]; ++k)
					{
						arg_t v;
						v.serialize(in);
						aux_spins[i][j][k] = v;
					}
			max_tau = param.n_tau_slices;
			tau = max_tau;
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, n_matrix_size);
			rebuild();
		}

		void initialize()
		{
			n_vertex_size = 2;
			n_matrix_size = param.n_flavor*l.n_sites();

			if (param.geometry == "hex")
				cb_bonds.resize(2);
			else
				cb_bonds.resize(3);

			delta = dmatrix_t(n_vertex_size, n_vertex_size);
			delta_W_r = dmatrix_t(n_vertex_size, n_matrix_size / 2);
			W_W_l = dmatrix_t(n_matrix_size / 2, n_vertex_size);
			M = dmatrix_t(n_vertex_size, n_vertex_size);
			create_checkerboard();
			id = dmatrix_t::Identity(n_matrix_size, n_matrix_size);
			id_2 = dmatrix_t::Identity(n_vertex_size, n_vertex_size);
			equal_time_gf = 0.5 * id;
			time_displaced_gf = 0.5 * id;
			build_vertex_matrices();

			dmatrix_t H0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			build_dirac_T(H0);
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(H0);
			expH0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			invExpH0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (int i = 0; i < expH0.rows(); ++i)
			{
				expH0(i, i) = std::exp(- solver.eigenvalues()[i] * param.dtau);
				invExpH0(i, i) = std::exp(solver.eigenvalues()[i] * param.dtau);
			}
			expH0 = solver.eigenvectors() * expH0 * solver.eigenvectors()
				.inverse();
			invExpH0 = solver.eigenvectors() * invExpH0 * solver.eigenvectors()
				.inverse();

			if (param.use_projector)
			{
				H0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
				build_dirac_H0(H0);
				get_trial_wavefunction(H0);
				stabilizer.set_P(P, Pt);
			}
			stabilizer.set_method(param.use_projector);
		}

		dmatrix_t symmetrize_EV(const dmatrix_t& S, const Eigen::VectorXd& en, const dmatrix_t& pm)
		{
			double epsilon = std::pow(10., -4.);

			dmatrix_t S_s = S + pm * S;
			dmatrix_t S_a = S - pm * S;
			dmatrix_t S_so(n_matrix_size, S_s.cols());
			dmatrix_t S_ao(n_matrix_size, S_s.cols());
			dmatrix_t S_f = dmatrix_t::Zero(n_matrix_size, 2*S_s.cols());

			for (int i = 0; i < S_s.cols(); ++i)
			{
				if (S_s.col(i).norm() > epsilon)
					S_s.col(i) /= S_s.col(i).norm();
				else
					S_s.col(i) *= 0.;
				if (S_a.col(i).norm() > epsilon)
					S_a.col(i) /= S_a.col(i).norm();
				else
					S_a.col(i) *= 0.;
			}

			int cnt = 0;
			for (int i = 0; i < S_s.cols(); ++i)
			{
				int j;
				for (j = i; j < S_s.cols() && std::abs(en(j)-en(i)) < epsilon ; ++j)
				{
					S_so.col(j) = S_s.col(j);
					S_ao.col(j) = S_a.col(j);
					for (int k = i; k < j; ++k)
					{
						S_so.col(j) -= S_so.col(k) * (S_so.col(k).dot(S_s.col(j)));
						S_ao.col(j) -= S_ao.col(k) * (S_ao.col(k).dot(S_a.col(j)));
					}
					//std::cout << "E=" << en(i) << ", orth: i=" << i << ", j=" << j << ": " << S_so.col(j).norm() << " " << S_ao.col(j).norm() << std::endl;
					if (S_so.col(j).norm() > epsilon)
					{
						S_so.col(j) /= S_so.col(j).norm();
						S_f.col(cnt) = S_so.col(j);
						++cnt;
					}
					if (S_ao.col(j).norm() > epsilon)
					{
						S_ao.col(j) /= S_ao.col(j).norm();
						S_f.col(cnt) = S_ao.col(j);
						++cnt;
					}
				}
				i = j - 1;
			}
			if (cnt != S.cols())
			{
				std::cout << "Error! Found " << cnt << " out of " << 2*S.cols() << std::endl;
				throw(std::runtime_error("Error in symmetrization. Wrong number of states."));
			}
			return S_f.leftCols(S.cols());
		}
		
		dmatrix_t ph_symmetrize_EV(const dmatrix_t& S, const dmatrix_t& pm)
		{
			double epsilon = std::pow(10., -4.);

			dmatrix_t S_s(n_matrix_size, 2), S_a(n_matrix_size, 2);
			S_s.col(0) = S.col(0) + S.col(1);
			S_s.col(1) = S.col(2) + S.col(3);
			S_a.col(0) = S.col(0) - S.col(1);
			S_a.col(1) = S.col(2) - S.col(3);
			dmatrix_t S_so(n_matrix_size, S_s.cols());
			dmatrix_t S_ao(n_matrix_size, S_s.cols());
			dmatrix_t S_sf = dmatrix_t::Zero(n_matrix_size, S_s.cols());
			dmatrix_t S_af = dmatrix_t::Zero(n_matrix_size, S_s.cols());

			for (int i = 0; i < S_s.cols(); ++i)
			{
				if (S_s.col(i).norm() > epsilon)
					S_s.col(i) /= S_s.col(i).norm();
				else
					S_s.col(i) *= 0.;
				if (S_a.col(i).norm() > epsilon)
					S_a.col(i) /= S_a.col(i).norm();
				else
					S_a.col(i) *= 0.;
			}

			int cnt_s = 0, cnt_a = 0;
			for (int i = 0; i < S_s.cols(); ++i)
			{
				int j;
				for (j = i; j < S_s.cols(); ++j)
				{
					S_so.col(j) = S_s.col(j);
					S_ao.col(j) = S_a.col(j);
					for (int k = i; k < j; ++k)
					{
						S_so.col(j) -= S_so.col(k) * (S_so.col(k).dot(S_s.col(j)));
						S_ao.col(j) -= S_ao.col(k) * (S_ao.col(k).dot(S_a.col(j)));
					}
					//std::cout << "orth: i=" << i << ", j=" << j << ": " << S_so.col(j).norm() << " " << S_ao.col(j).norm() << std::endl;
					if (S_so.col(j).norm() > epsilon)
					{
						S_so.col(j) /= S_so.col(j).norm();
						S_sf.col(cnt_s) = S_so.col(j);
						++cnt_s;
					}
					if (S_ao.col(j).norm() > epsilon)
					{
						S_ao.col(j) /= S_ao.col(j).norm();
						S_af.col(cnt_a) = S_ao.col(j);
						++cnt_a;
					}
				}
				i = j - 1;
			}
			dmatrix_t S_f(n_matrix_size, cnt_s + cnt_a);
			S_f.leftCols(cnt_s) = S_sf.leftCols(cnt_s);
			S_f.rightCols(cnt_a) = S_af.leftCols(cnt_a);
			return S_f.leftCols(S.cols());
		}
		
		void get_trial_wavefunction(const dmatrix_t& H)
		{
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(H);
			if (param.geometry != "rhom")
			{
				std::cout << param.geometry << std::endl;
				std::cout << n_matrix_size << " sites." << std::endl;
				std::cout << solver.eigenvalues() << std::endl;
				
				//std::cout << "H - P * H * P" << std::endl;
				//std::cout << H - inv_pm * H * inv_pm << std::endl;
			}
			if (l.n_sites() % 3 != 0)
			{
				P = solver.eigenvectors().leftCols(n_matrix_size/2);
				Pt = P.adjoint();
				return;
			}
			dmatrix_t inv_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				ph_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (int i = 0; i < n_matrix_size; ++i)
			{
				inv_pm(i, l.inverted_site(i)) = 1.;
				ph_pm(i, i) = l.parity(i);
				//std::cout << i << " <-> " << l.inverted_site(i) << std::endl;
			}
			
			double epsilon = std::pow(10., -4.), total_inv_parity = 1, total_ph_parity = 1;
			auto S_f = symmetrize_EV(solver.eigenvectors(), solver.eigenvalues(), inv_pm);
			std::vector<double> inv_parity(n_matrix_size), ph_2p_parity(6);
			//std::cout << "after inversion symmetry" << std::endl;
			for (int i = 0; i < n_matrix_size; ++i)
			{
				inv_parity[i] = std::real(S_f.col(i).dot(inv_pm * S_f.col(i)));
				//std::cout << i << ": e = " << solver.eigenvalues()[i] << ", P = " << inv_parity[i] << std::endl;
			}
			
			dmatrix_t ph_1p_block = S_f.block(0, n_matrix_size/2-2, n_matrix_size, 4);
			Eigen::VectorXd ph_ev = Eigen::VectorXd::Zero(4);
			//std::cout << "PH first" << std::endl;
			ph_1p_block = symmetrize_EV(ph_1p_block, ph_ev, ph_pm);
			for (int i = 0; i < ph_1p_block.cols(); ++i)
			{
				double inv_p = std::real(ph_1p_block.col(i).dot(inv_pm * ph_1p_block.col(i)));
				ph_2p_parity[i] = std::real(ph_1p_block.col(i).dot(ph_pm * ph_1p_block.col(i)));
				//std::cout << i << ": e = " << ph_ev[i] << ", P = " << inv_p << ", PH = " << ph_2p_parity[i] << std::endl;
			}
			//std::cout << "PH second" << std::endl;
			ph_1p_block = ph_symmetrize_EV(ph_1p_block, ph_pm);
			for (int i = 0; i < ph_1p_block.cols(); ++i)
			{
				double inv_p = std::real(ph_1p_block.col(i).dot(inv_pm * ph_1p_block.col(i)));
				ph_2p_parity[i] = std::real(ph_1p_block.col(i).dot(ph_pm * ph_1p_block.col(i)));
				//std::cout << i << ": e = " << ph_ev[i] << ", P = " << inv_p << ", PH = " << ph_2p_parity[i] << std::endl;
			}
			std::vector<dmatrix_t> ph_2p_block(6, dmatrix_t(n_matrix_size, 2));
			//PH = 1
			ph_2p_block[0].col(0) = ph_1p_block.col(0);
			ph_2p_block[0].col(1) = ph_1p_block.col(3);
			ph_2p_block[1].col(0) = ph_1p_block.col(1);
			ph_2p_block[1].col(1) = ph_1p_block.col(2);
			//ph_2p_block[2].col(0) = (ph_1p_block.col(0) - ph_1p_block.col(1))/std::sqrt(2.);
			//ph_2p_block[2].col(1) = (ph_1p_block.col(2) - ph_1p_block.col(3))/std::sqrt(2.);
			//PH = -1
			ph_2p_block[3].col(0) = ph_1p_block.col(0);
			ph_2p_block[3].col(1) = ph_1p_block.col(1);
			ph_2p_block[4].col(0) = ph_1p_block.col(2);
			ph_2p_block[4].col(1) = ph_1p_block.col(3);
			//ph_2p_block[5].col(0) = (ph_1p_block.col(0) + ph_1p_block.col(1))/std::sqrt(2.);
			//ph_2p_block[5].col(1) = (ph_1p_block.col(2) + ph_1p_block.col(3))/std::sqrt(2.);
			ph_2p_parity = {-1., -1., -1., 1., 1., 1.};
			std::vector<double> inv_2p_parity;// = {-1., -1., -1., 1., 1., -1.};
			inv_2p_parity.push_back(ph_2p_block[0].col(0).dot(inv_pm * ph_2p_block[0].col(0))
				* ph_2p_block[0].col(1).dot(inv_pm * ph_2p_block[0].col(1)));
			inv_2p_parity.push_back(ph_2p_block[1].col(0).dot(inv_pm * ph_2p_block[1].col(0))
				* ph_2p_block[1].col(1).dot(inv_pm * ph_2p_block[1].col(1)));
			inv_2p_parity.push_back(ph_1p_block.col(0).dot(inv_pm * ph_1p_block.col(0))
				* ph_1p_block.col(2).dot(inv_pm * ph_1p_block.col(2)));
			
			inv_2p_parity.push_back(ph_2p_block[3].col(0).dot(inv_pm * ph_2p_block[3].col(0))
				* ph_2p_block[3].col(1).dot(inv_pm * ph_2p_block[3].col(1)));
			inv_2p_parity.push_back(ph_2p_block[4].col(0).dot(inv_pm * ph_2p_block[4].col(0))
				* ph_2p_block[4].col(1).dot(inv_pm * ph_2p_block[4].col(1)));
			inv_2p_parity.push_back(ph_1p_block.col(0).dot(inv_pm * ph_1p_block.col(0))
				* ph_1p_block.col(2).dot(inv_pm * ph_1p_block.col(2)));
			
			
			P.resize(n_matrix_size, n_matrix_size / 2);
			for (int i = 0; i < n_matrix_size/2-2; ++i)
			{
				total_inv_parity *= inv_parity[i];
				P.col(i) = S_f.col(i);
			}
			for (int i = 0; i < ph_2p_block.size(); ++i)
				std::cout << "i = " << i << ": E = 0, total invP = " << total_inv_parity*inv_2p_parity[i] << ", invP = " << inv_2p_parity[i] << ", phP = " << ph_2p_parity[i] << std::endl;
			int indices[] = {0, 1, 3, 4};
			for (int i = 0; i < 4; ++i)
				if (std::abs(total_inv_parity * inv_2p_parity[indices[i]] - param.inv_symmetry) < epsilon)
				{
					std::cout << "Taken: i=" << indices[i] << std::endl;
					P.block(0, n_matrix_size/2-2, n_matrix_size, 2) = ph_2p_block[indices[i]];
					total_inv_parity *= inv_2p_parity[indices[i]];
					break;
				}
			
			Pt = P.adjoint();
			//std::cout << "Total inversion parity: " << total_inv_parity << std::endl;
			if (std::abs(param.inv_symmetry - total_inv_parity) > epsilon)
			{
				std::cout << "Error! Wrong parity of trial wave function." << std::endl;
				throw(std::runtime_error("Wrong parity in trial wave function."));
			}
		}

		void build_dirac_H0(dmatrix_t& H0)
		{
			for (int alpha = 0; alpha < param.n_flavor; ++alpha)
			{
				int as = alpha * l.n_sites();
				for (auto& a : l.bonds("nearest neighbors"))
					H0(a.first+as, a.second+as) = -param.t;
				for (auto& a : l.bonds("d3_bonds"))
					H0(a.first+as, a.second+as) = -param.tprime;
				for (int i = 0; i < l.n_sites(); ++i)
					H0(i+as, i+as) = l.parity(i) * param.stag_mu + param.mu;
			}
		}
		
		void build_dirac_T(dmatrix_t& H0)
		{
			for (int alpha = 0; alpha < param.n_flavor; ++alpha)
			{
				int as = alpha * l.n_sites();
				for (auto& a : l.bonds("d3_bonds"))
					H0(a.first+as, a.second+as) = -param.tprime;
				for (int i = 0; i < l.n_sites(); ++i)
					H0(i+as, i+as) = l.parity(i) * param.stag_mu + param.mu;
			}
		}

		void build_dirac_vertex(int cnt, int flavor, double parity, double spin)
		{
			if (flavor == 0)
			{
				numeric_t c = std::cosh(param.t * param.dtau + param.lambda * spin);
				numeric_t s = std::sinh(param.t * param.dtau + param.lambda * spin);
				numeric_t cp = std::cosh(param.t * param.dtau - param.lambda * spin);
				numeric_t sp = std::sinh(param.t * param.dtau - param.lambda * spin);

				vertex_matrices[cnt] << c, s, s, c;
				inv_vertex_matrices[cnt] << c, -s, -s, c;
				delta_matrices[cnt] << cp*c - sp*s - 1., -cp*s+sp*c, sp*c-cp*s, -sp*s + cp*c - 1.;
			}
			else if (flavor == 1)
			{
				numeric_t x = param.kappa * spin;

				vertex_matrices[cnt] << 1., x, x, 1.;
				inv_vertex_matrices[cnt] << 1., -x, -x, 1.;
				delta_matrices[cnt] << 0., -2.*x, -2.*x, 0.;
			}
		}

		void build_vertex_matrices()
		{
			vertex_matrices.resize(4*param.n_flavor, dmatrix_t(n_vertex_size, n_vertex_size));
			inv_vertex_matrices.resize(4*param.n_flavor, dmatrix_t(n_vertex_size, n_vertex_size));
			delta_matrices.resize(4*param.n_flavor, dmatrix_t(n_vertex_size, n_vertex_size));
			int cnt = 0;
			for (int flavor = 0; flavor < param.n_flavor; ++flavor)
				for (double parity : {1., -1.})
					for (double spin : {1., -1.})
					{
						build_dirac_vertex(cnt, flavor, parity, spin);
						++cnt;
					}
		}

		dmatrix_t& get_vertex_matrix(int i, int j, int alpha, int beta, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			return vertex_matrices[(alpha+beta)%2*4 + i%2*2 + static_cast<int>(s<0)];
		}

		dmatrix_t& get_inv_vertex_matrix(int i, int j, int alpha, int beta, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			return inv_vertex_matrices[4*(alpha+beta)%2 + i%2*2 + static_cast<int>(s<0)];
		}

		dmatrix_t& get_delta_matrix(int i, int j, int alpha, int beta, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			return delta_matrices[4*(alpha+beta)%2 + i%2*2 + static_cast<int>(s<0)];
		}

		int get_bond_type(const std::pair<int, int>& bond) const
		{
			for (int i = 0; i < cb_bonds.size(); ++i)
				if (cb_bonds[i].at(bond.first) == bond.second)
					return i;
		}

		const arg_t& get_spins(int t, int alpha=0, int beta=0) const
		{
			return aux_spins[alpha][beta][t-1];
		}
		
		arg_t& get_spins(int t, int alpha=0, int beta=0)
		{
			return aux_spins[alpha][beta][t-1];
		}

		int get_tau()
		{
			return tau;
		}

		int get_max_tau()
		{
			return max_tau;
		}

		void update_tau()
		{
			tau += param.direction;
		}

		int bond_index(int i, int j) const
		{
			return bond_indices.at({std::min(i, j), std::max(i, j)});
		}

		int n_cb_bonds() const
		{
			return cb_bonds.size();
		}

		const std::map<int, int>& get_cb_bonds(int i) const
		{
			return cb_bonds[i];
		}
		
		const std::vector<std::pair<int, int>>& get_nn_bonds(int b) const
		{
			return nn_bonds[b];
		}
		
		const std::vector<std::pair<int, int>>& get_inv_nn_bonds(int b) const
		{
			return inv_nn_bonds[b];
		}

		void flip_spin(const std::pair<int, int>& b, int alpha = 0, int beta = 0)
		{
			get_spins(tau, alpha, beta).flip(bond_index(b.first, b.second));
		}

		void buffer_equal_time_gf()
		{
			if (param.use_projector)
			{
				W_l_buffer = proj_W_l;
				W_r_buffer = proj_W_r;
				W_buffer = proj_W;

				//gf_buffer = equal_time_gf;
			}
			else
				gf_buffer = equal_time_gf;
			gf_buffer_tau = tau;
			dir_buffer = param.direction;
		}

		void reset_equal_time_gf_to_buffer()
		{
			if (param.use_projector)
			{
				proj_W_l = W_l_buffer;
				proj_W_r = W_r_buffer;
				proj_W = W_buffer;

				//equal_time_gf = gf_buffer;
			}
			else
				equal_time_gf = gf_buffer;
			tau = gf_buffer_tau;
			param.direction = dir_buffer;
		}

		void enable_time_displaced_gf(int direction)
		{
			update_time_displaced_gf = true;
			stabilizer.enable_time_displaced_gf(direction);
		}

		void disable_time_displaced_gf()
		{
			update_time_displaced_gf = false;
			stabilizer.disable_time_displaced_gf();
		}

		void build(boost::multi_array<arg_t, 3>& args)
		{
			//aux_spins.swap(args);
			aux_spins.resize(boost::extents[args.shape()[0]][args.shape()[1]][args.shape()[2]]);
			aux_spins = args;
			max_tau = param.n_tau_slices;
			tau = max_tau;
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, n_matrix_size);
			rebuild();
		}

		void rebuild()
		{
			if (aux_spins.size() == 0) return;
			if (param.use_projector)
			{
				stabilizer.set_proj_l(n_intervals, id);
				for (int n = n_intervals - 1; n >= 0; --n)
				{
					dmatrix_t b = propagator((n + 1) * param.n_delta,
						n * param.n_delta);
					stabilizer.set_proj_l(n, b);
				}
				stabilizer.set_proj_r(0, id);
				for (int n = 1; n <= n_intervals; ++n)
				{
					dmatrix_t b = propagator(n * param.n_delta,
						(n - 1) * param.n_delta);
					stabilizer.set_proj_r(n, b);
				}
			}
			else
			{
				for (int n = 1; n <= n_intervals; ++n)
				{
					dmatrix_t b = propagator(n * param.n_delta,
						(n - 1) * param.n_delta);
					stabilizer.set(n, b);
				}
			}
		}

		void multiply_vertex_from_left(dmatrix_t& m,
			int bond_type, const arg_t& vertex, int alpha, int beta, int inv)
		{
			dmatrix_t old_m = m;
			int as = alpha * l.n_sites(), bs = beta * l.n_sites();
			const int N = nn_bonds[bond_type].size();
			if (inv == 1)
			{
				for (int b = 0; b < N; ++b)
				{
					int i = nn_bonds[bond_type][b].first, j = nn_bonds[bond_type][b].second;
					double sigma = vertex.get(bond_index(i, j));
					dmatrix_t& vm = get_vertex_matrix(i, j, alpha, beta, sigma);
					m.row(i+as).noalias() = old_m.row(i+as) * vm(0, 0) + old_m.row(j+bs) * vm(0, 1);
					m.row(j+as).noalias() = old_m.row(i+as) * vm(1, 0) + old_m.row(j+bs) * vm(1, 1);
				}
			}
			else
			{
				for (int b = 0; b < N; ++b)
				{
					int i = nn_bonds[bond_type][b].first, j = nn_bonds[bond_type][b].second;
					double sigma = vertex.get(bond_index(i, j));
					dmatrix_t& vm = get_inv_vertex_matrix(i, j, alpha, beta, sigma);
					m.row(i+as).noalias() = old_m.row(i+as) * vm(0, 0) + old_m.row(j+bs) * vm(0, 1);
					m.row(j+as).noalias() = old_m.row(i+as) * vm(1, 0) + old_m.row(j+bs) * vm(1, 1);
				}
			}
		}

		void multiply_vertex_from_right(dmatrix_t& m,
			int bond_type, const arg_t& vertex, int alpha, int beta, int inv)
		{
			dmatrix_t old_m = m;
			int as = alpha * l.n_sites(), bs = beta * l.n_sites();
			const int N = nn_bonds[bond_type].size();
			if (inv == 1)
			{
				for (int b = 0; b < N; ++b)
				{
					int i = nn_bonds[bond_type][b].first, j = nn_bonds[bond_type][b].second;
					double sigma = vertex.get(bond_index(i, j));
					dmatrix_t& vm = get_vertex_matrix(i, j, alpha, beta, sigma);
					m.col(i+bs).noalias() = old_m.col(i+as) * vm(0, 0)
						+ old_m.col(j+bs) * vm(1, 0);
					m.col(j+bs).noalias() = old_m.col(i+as) * vm(0, 1)
						+ old_m.col(j+bs) * vm(1, 1);
				}
			}
			else
			{
				for (int b = 0; b < N; ++b)
				{
					int i = nn_bonds[bond_type][b].first, j = nn_bonds[bond_type][b].second;
					double sigma = vertex.get(bond_index(i, j));
					dmatrix_t& vm = get_inv_vertex_matrix(i, j, alpha, beta, sigma);
					m.col(i+bs).noalias() = old_m.col(i+as) * vm(0, 0)
						+ old_m.col(j+bs) * vm(1, 0);
					m.col(j+bs).noalias() = old_m.col(i+as) * vm(0, 1)
						+ old_m.col(j+bs) * vm(1, 1);
				}
			}
		}
		
		void multiply_T_matrix()
		{
			if (!param.multiply_T)
				return;
			if (param.use_projector)
			{
				if (param.direction == 1)
				{
					proj_W_l = proj_W_l * invExpH0;
					proj_W_r = expH0 * proj_W_r;
				}
				else if (param.direction == -1)
				{
					proj_W_l = proj_W_l * expH0;
					proj_W_r = invExpH0 * proj_W_r;
				}
			}
			else
			{
				if (param.direction == 1)
					equal_time_gf = expH0 * equal_time_gf * invExpH0;
				else if (param.direction == -1)
					equal_time_gf = invExpH0 * equal_time_gf * expH0;
			}
		}
		
		void multiply_Gamma_matrix(int bond_type, int alpha, int beta)
		{
			auto& vertex = get_spins(tau, alpha, beta);
			if (param.use_projector)
			{
				if (param.direction == 1)
				{
					multiply_vertex_from_left(proj_W_r, bond_type, vertex, alpha, beta, 1);
					multiply_vertex_from_right(proj_W_l, bond_type, vertex, alpha, beta, -1);
				}
				else if (param.direction == -1)
				{
					multiply_vertex_from_left(proj_W_r, bond_type, vertex, alpha, beta, -1);
					multiply_vertex_from_right(proj_W_l, bond_type, vertex, alpha, beta, 1);
				}
			}
			else
			{
				if (param.direction == 1)
				{
					multiply_vertex_from_left(equal_time_gf, bond_type, vertex, alpha, beta, 1);
					multiply_vertex_from_right(equal_time_gf, bond_type, vertex, alpha, beta, -1);
				}	
				else if (param.direction == -1)
				{
					multiply_vertex_from_left(equal_time_gf, bond_type, vertex, alpha, beta, -1);
					multiply_vertex_from_right(equal_time_gf, bond_type, vertex, alpha, beta, 1);
				}
			}
		}

		dmatrix_t propagator(int tau_n, int tau_m)
		{
			dmatrix_t b = id;
			for (int n = tau_n; n > tau_m; --n)
			{
				for (int alpha = 0; alpha < param.n_flavor; ++alpha)
					for (int beta = 0; beta < param.n_flavor; ++beta)
					{
						auto& vertex = get_spins(n, alpha, beta);
						for (int bt = 0; bt < cb_bonds.size(); ++bt)
							multiply_vertex_from_right(b, bt, vertex, alpha, beta, 1);
					}
				if (param.multiply_T)
					b *= expH0;
			}
			return b;
		}
		
		void advance_time_slice()
		{
			if (param.direction == 1)
			{
				update_tau();
				multiply_T_matrix();
				
				for (int alpha = param.n_flavor - 1; alpha >= 0; --alpha)
					for (int beta = param.n_flavor - 1; beta >= 0; --beta)
						for (int bt = nn_bonds.size() - 1; bt >= 0; --bt)
							multiply_Gamma_matrix(bt, alpha, beta);
			}
			else if (param.direction == -1)
			{
				for (int alpha = 0; alpha < param.n_flavor; ++alpha)
					for (int beta = 0; beta < param.n_flavor; ++beta)
						for (int bt = 0; bt < nn_bonds.size(); ++bt)
							multiply_Gamma_matrix(bt, alpha, beta);
				
				update_tau();
				multiply_T_matrix();
			}
		}

		void stabilize_forward()
		{
			if (tau % param.n_delta != 0)
					return;
			// n = 0, ..., n_intervals - 1
			int n = tau / param.n_delta - 1;
			dmatrix_t b = propagator((n+1)*param.n_delta, n*param.n_delta);
			stabilizer.stabilize_forward(n, b);
		}

		void stabilize_backward()
		{
			if (tau % param.n_delta != 0)
					return;
			//n = n_intervals, ..., 1
			int n = tau / param.n_delta + 1;
			dmatrix_t b = propagator(n*param.n_delta, (n-1)*param.n_delta);
			stabilizer.stabilize_backward(n, b);
		}

		numeric_t try_ising_flip(int i, int j, int alpha = 0, int beta = 0)
		{
			auto& vertex = get_spins(tau, alpha, beta);
			double sigma = vertex.get(bond_index(i, j));
			int m = std::min(i, j), n = std::max(i, j);
			delta = get_delta_matrix(m, n, alpha, beta, sigma);
			m += alpha * l.n_sites();
			n += beta * l.n_sites();
			last_flip = {m, n};

			if (param.use_projector)
			{
				dmatrix_t b_l(P.cols(), n_vertex_size);
				b_l.col(0) = proj_W_l.col(m);
				b_l.col(1) = proj_W_l.col(n);
				W_W_l.noalias() = proj_W * b_l;
				dmatrix_t b_r(n_vertex_size, P.cols());
				b_r.row(0) = proj_W_r.row(m);
				b_r.row(1) = proj_W_r.row(n);
				delta_W_r.noalias() = delta * b_r;

				M = id_2;
				M.noalias() += delta_W_r * W_W_l;
				return M.determinant();
			}
			else
			{
				dmatrix_t& gf = equal_time_gf;
				dmatrix_t g(n_vertex_size, n_vertex_size);
				g << 1.-gf(m, m), -gf(m, n), -gf(n, m), 1.-gf(n, n);
				M = id_2; M.noalias() += g * delta;
				return M.determinant();
			}
		}

		void update_equal_time_gf_after_flip()
		{
			int indices[2] = {last_flip.first, last_flip.second};
			
			if (param.use_projector)
			{
				proj_W_r.row(indices[0]).noalias() += delta_W_r.row(0);
				proj_W_r.row(indices[1]).noalias() += delta_W_r.row(1);

				/*
				M = M.inverse().eval();
				dmatrix_t delta_W_r_W = delta_W_r * proj_W;
				dmatrix_t W_W_l_M = W_W_l * M;
				proj_W.noalias() -= W_W_l_M * delta_W_r_W;
				*/
				
				M = M.inverse().eval();
				proj_W.noalias() -= (W_W_l * M) * (delta_W_r * proj_W);
			}
			else
			{
				M = M.inverse().eval();
				dmatrix_t& gf = equal_time_gf;
				dmatrix_t g_cols(n_matrix_size, n_vertex_size);
				g_cols.col(0) = gf.col(indices[0]);
				g_cols.col(1) = gf.col(indices[1]);
				dmatrix_t g_rows(n_vertex_size, n_matrix_size);
				g_rows.row(0) = gf.row(indices[0]);
				g_rows.row(1) = gf.row(indices[1]);
				g_rows(0, indices[0]) -= 1.;
				g_rows(1, indices[1]) -= 1.;
				dmatrix_t gd = g_cols * delta;
				dmatrix_t mg = M * g_rows;
				gf.noalias() += gd * mg;
			}
		}

		/*
		void static_measure(std::vector<double>& c, numeric_t& n, numeric_t& energy, numeric_t& m2, numeric_t& epsilon, numeric_t& chern)
		{
			if (param.use_projector)
				equal_time_gf = id - proj_W_r * proj_W * proj_W_l;
			numeric_t im = {0., 1.};
			for (int i = 0; i < l.n_sites(); ++i)
			{
				n += equal_time_gf(i, i) / numeric_t(l.n_sites());
				for (int j = 0; j < l.n_sites(); ++j)
					{
						double re = std::real(equal_time_gf(i, j)
							* equal_time_gf(i, j));
						//Correlation function
						c[l.distance(i, j)] += re / l.n_sites();
						//M2 structure factor
						m2 += l.parity(i) * l.parity(j) * re
							/ std::pow(l.n_sites(), 2);
					}
			}
			for (auto& i : l.bonds("nearest neighbors"))
			{
				energy += -l.parity(i.first) * param.t * std::imag(equal_time_gf(i.second, i.first))
					+ param.V * std::real(equal_time_gf(i.second, i.first) * equal_time_gf(i.second, i.first)) / 2.;

				epsilon += im * l.parity(i.first) * equal_time_gf(i.second, i.first) / numeric_t(l.n_bonds());
			}
			for (auto& i : l.bonds("chern"))
				chern += im * (equal_time_gf(i.second, i.first) - equal_time_gf(i.first, i.second)) / numeric_t(l.n_bonds());
		}
		*/

		void measure_static_observable(std::vector<double>& values,
			const std::vector<wick_static_base<dmatrix_t>>& obs)
		{
			if (param.use_projector)
			{
				dmatrix_t wl = proj_W * proj_W_l;
				equal_time_gf = id;
				equal_time_gf.noalias() -= proj_W_r * wl;
			}
			for (int i = 0; i < values.size(); ++i)
					values[i] = obs[i].get_obs(equal_time_gf);

			if (param.mu != 0 || param.stag_mu != 0)
			{
				numeric_t n = 0.;
				for (int i = 0; i < l.n_sites(); ++i)
					n += equal_time_gf(i, i) / numeric_t(l.n_sites());
				measure.add("n_re", std::real(n*param.sign_phase));
				measure.add("n_im", std::imag(n*param.sign_phase));
				measure.add("n", std::real(n));
			}
		}

		void measure_dynamical_observable(std::vector<std::vector<double>>&
			dyn_tau, const std::vector<wick_base<dmatrix_t>>& obs)
		{
			//check_td_gf_stability();
			if (param.use_projector)
			{
				buffer_equal_time_gf();
				stabilizer.set_buffer();
				std::vector<dmatrix_t> et_gf_L(param.n_discrete_tau);
				std::vector<dmatrix_t> et_gf_R(2*param.n_discrete_tau);
				std::vector<dmatrix_t> n1_td_gf(2*param.n_discrete_tau);
				std::vector<dmatrix_t> td_gf(2*param.n_discrete_tau);
				time_displaced_gf = id;
				
				if (tau == max_tau/2 + param.n_discrete_tau * param.n_dyn_tau)
					param.direction = -1;
				else if (tau == max_tau/2 - param.n_discrete_tau * param.n_dyn_tau)
					param.direction = 1;

				for (int n = 0; n < param.n_discrete_tau; ++n)
				{
					for (int m = 0; m < param.n_dyn_tau; ++m)
					{
						advance_time_slice();
						if (param.direction == -1)
							stabilize_backward();
						else
							stabilize_forward();
					}
					dmatrix_t wl = proj_W * proj_W_l;
					et_gf_L[n] = id;
					et_gf_L[n].noalias() -= proj_W_r * wl;
					td_gf[n] = et_gf_L[n];
				}
				dmatrix_t& et_gf_0 = et_gf_L[param.n_discrete_tau - 1];
				for (int n = 0; n < 2*param.n_discrete_tau; ++n)
				{
					for (int m = 0; m < param.n_dyn_tau; ++m)
					{
						advance_time_slice();
						if (param.direction == -1)
							stabilize_backward();
						else
							stabilize_forward();
					}
					dmatrix_t wl = proj_W * proj_W_l;
					et_gf_R[n] = id;
					et_gf_R[n].noalias() -= proj_W_r * wl;
					if (n < param.n_discrete_tau)
						td_gf[n+param.n_discrete_tau] = et_gf_R[n];
				}

				/*
				//n = 0
				for (int m = 0; m < param.n_discrete_tau; ++m)
				{
					for (int i = 0; i < dyn_tau.size(); ++i)
					{
						dyn_tau[i][0] += obs[i].get_obs(et_gf_L[m], et_gf_L[m], et_gf_L[m]) / (2.*param.n_discrete_tau);
						dyn_tau[i][0] += obs[i].get_obs(et_gf_R[m], et_gf_R[m], et_gf_R[m]) / (2.*param.n_discrete_tau);
					}
				}
				//n = 1
				for (int m = 0; m < 2*param.n_discrete_tau; ++m)
				{
					int tl = max_tau/2 + param.n_discrete_tau * param.n_dyn_tau;
					dmatrix_t p = propagator(tl - m*param.n_dyn_tau, tl - (m+1)*param.n_dyn_tau);
					td_gf[m] = p * td_gf[m];
					n1_td_gf[m] = td_gf[m];
					
					for (int i = 0; i < dyn_tau.size(); ++i)
						dyn_tau[i][1] += obs[i].get_obs(et_gf_0, et_gf_R[0], td_gf[m]) / (2.*param.n_discrete_tau);
				}
				//n > 1
				for (int n = 2; n < 2*param.n_discrete_tau; ++n)
					for (int m = n - 1; m < 2*param.n_discrete_tau; ++m)
					{
						td_gf[m] = n1_td_gf[m-n+1] * td_gf[m];
						
						for (int i = 0; i < dyn_tau.size(); ++i)
							dyn_tau[i][n] += obs[i].get_obs(et_gf_0, et_gf_R[n-1], td_gf[m]) / (2.*param.n_discrete_tau - n + 1);
					}
				*/
				
				
				
				for (int i = 0; i < dyn_tau.size(); ++i)
					dyn_tau[i][0] = obs[i].get_obs(et_gf_0, et_gf_0, et_gf_0);
				for (int n = 1; n <= param.n_discrete_tau; ++n)
				{
					dmatrix_t g_l, g_r;
					if (param.direction == -1)
					{
						g_l = propagator(max_tau/2 + n*param.n_dyn_tau,
							max_tau/2 + (n-1)*param.n_dyn_tau) * et_gf_L[et_gf_L.size() - n];
						g_r = propagator(max_tau/2 - (n-1)*param.n_dyn_tau,
							max_tau/2 - n*param.n_dyn_tau) * et_gf_R[n-1];
					}
					else
					{
						g_l = et_gf_R[n-1] * propagator(max_tau/2 + n*param.n_dyn_tau,
							max_tau/2 + (n-1)*param.n_dyn_tau);
						g_r = et_gf_L[et_gf_L.size() - n] * propagator(max_tau/2 - (n-1)*param.n_dyn_tau,
							max_tau/2 - n*param.n_dyn_tau);
					}
					
					time_displaced_gf = g_l * time_displaced_gf;
					for (int i = 0; i < dyn_tau.size(); ++i)
						dyn_tau[i][2*n-1] = obs[i].get_obs(et_gf_0, et_gf_R[2*n-2],
							time_displaced_gf);
					time_displaced_gf = time_displaced_gf * g_r;
					for (int i = 0; i < dyn_tau.size(); ++i)
						dyn_tau[i][2*n] = obs[i].get_obs(et_gf_0, et_gf_R[2*n-1],
							time_displaced_gf);
				}
				

				reset_equal_time_gf_to_buffer();
				stabilizer.restore_buffer();
			}
			else
			{
				// 1 = forward, -1 = backward
				int direction = tau == 0 ? 1 : -1;
				dir_buffer = param.direction;
				param.direction = direction;
				dmatrix_t et_gf_0 = equal_time_gf;
				enable_time_displaced_gf(direction);
				time_displaced_gf = equal_time_gf;
				for (int n = 0; n <= max_tau; ++n)
				{
					if (n % (max_tau / param.n_discrete_tau) == 0)
					{
						int t = n / (max_tau / param.n_discrete_tau);
						for (int i = 0; i < dyn_tau.size(); ++i)
							dyn_tau[i][t] = obs[i].get_obs(et_gf_0, equal_time_gf,
								time_displaced_gf);
					}
					if (direction == 1 && tau < max_tau)
					{
						advance_time_slice();
						stabilize_forward();
					}
					else if (direction == -1 && tau > 0)
					{
						advance_time_slice();
						stabilize_backward();
					}
				}
				disable_time_displaced_gf();
				if (direction == 1)
					tau = 0;
				else if (direction == -1)
					tau = max_tau;
				param.direction = dir_buffer;
			}
		}
		
		void check_td_gf_stability()
		{
			buffer_equal_time_gf();
			stabilizer.set_buffer();
			std::vector<dmatrix_t> et_gf_L(param.n_discrete_tau);
			std::vector<dmatrix_t> et_gf_R(2*param.n_discrete_tau);
			std::vector<dmatrix_t> td_gf(2);
			time_displaced_gf = id;
			
			if (tau == max_tau/2 + param.n_discrete_tau * param.n_dyn_tau)
				param.direction = -1;
			else if (tau == max_tau/2 - param.n_discrete_tau * param.n_dyn_tau)
				param.direction = 1;

			for (int n = 0; n < param.n_discrete_tau; ++n)
			{
				for (int m = 0; m < param.n_dyn_tau; ++m)
				{
					advance_time_slice();
					if (param.direction == -1)
						stabilize_backward();
					else
						stabilize_forward();
				}
				et_gf_L[n] = id;
				et_gf_L[n].noalias() -= proj_W_r * proj_W * proj_W_l;
			}
			dmatrix_t& et_gf_0 = et_gf_L[param.n_discrete_tau - 1];
			for (int n = 0; n < 2*param.n_discrete_tau; ++n)
			{
				for (int m = 0; m < param.n_dyn_tau; ++m)
				{
					advance_time_slice();
					if (param.direction == -1)
						stabilize_backward();
					else
						stabilize_forward();
				}
				et_gf_R[n] = id;
				et_gf_R[n].noalias() -= proj_W_r * proj_W * proj_W_l;
			}
			
			int dist[] = {1, 4};
			for (int j = 0; j < 2; ++j)
			{
				int i = dist[j];
				time_displaced_gf = id;
				for (int n = 1; n <= param.n_discrete_tau / i; ++n)
				{
					dmatrix_t g_l, g_r;
					if (param.direction == -1)
					{
						g_l = propagator(max_tau/2 + n*param.n_dyn_tau*i,
							max_tau/2 + (n-1)*param.n_dyn_tau*i) * et_gf_L[et_gf_L.size() - (n-1)*i - 1];
						g_r = propagator(max_tau/2 - (n-1)*param.n_dyn_tau*i,
							max_tau/2 - n*param.n_dyn_tau*i) * et_gf_R[n*i-1];
					}
					else
					{
						g_l = et_gf_R[n*i-1] * propagator(max_tau/2 + n*param.n_dyn_tau*i,
							max_tau/2 + (n-1)*param.n_dyn_tau*i);
						g_r = et_gf_L[et_gf_L.size() - (n-1)*i - 1] * propagator(max_tau/2 - (n-1)*param.n_dyn_tau*i,
							max_tau/2 - n*param.n_dyn_tau*i);
					}
					
					time_displaced_gf = g_l * time_displaced_gf;
					time_displaced_gf = time_displaced_gf * g_r;
				}
				td_gf[j] = time_displaced_gf;
			}
			double tde = (td_gf[0] - td_gf[1]).norm();
			measure.add("td_norm_error", tde);
			if (tde > std::pow(10., -10.))
				std::cout << "TD error: " << tde << std::endl;

			reset_equal_time_gf_to_buffer();
			stabilizer.restore_buffer();
		}
	private:
		void create_checkerboard()
		{
			int cnt = 0;
			for (int i = 0; i < l.n_sites(); ++i)
				for (int j = i+1; j < l.n_sites(); ++j)
					if (l.distance(i, j) == 1)
					{
						bond_indices[{i, j}] = cnt;
						++cnt;
					}

			for (int i = 0; i < l.n_sites(); ++i)
			{
				auto& nn = l.neighbors(i, "nearest neighbors");
				for (int j : nn)
				{
					for (auto& b : cb_bonds)
					{
						if (!b.count(i) && !b.count(j))
						{
							b[i] = j;
							b[j] = i;
							break;
						}
					}
				}
			}
			
			nn_bonds.resize(cb_bonds.size());
			for (int b = 0; b < nn_bonds.size(); ++b)
				for (int i = 0; i < l.n_sites(); ++i)
				{
					int j = cb_bonds[b][i];
					if (i > j) continue;
					nn_bonds[b].push_back({i, j});
				}
			inv_nn_bonds.resize(nn_bonds.size());
			for (int b = 0; b < nn_bonds.size(); ++b)
				for (int i = nn_bonds[b].size() - 1; i >= 0; --i)
					inv_nn_bonds[b].push_back(nn_bonds[b][i]);
		}

		void print_matrix(const dmatrix_t& m)
		{
			Eigen::IOFormat clean(6, 0, ", ", "\n", "[", "]");
			std::cout << m.format(clean) << std::endl << std::endl;
		}
	private:
		Random& rng;
		const lattice& l;
		parameters& param;
		measurements& measure;
		int n_intervals;
		int tau;
		int max_tau;
		boost::multi_array<arg_t, 3> aux_spins;
		std::map<std::pair<int, int>, int> bond_indices;
		std::vector<int> pos_buffer;
		bool update_time_displaced_gf;
		int n_vertex_size;
		int n_matrix_size;
		std::vector<dmatrix_t> vertex_matrices;
		std::vector<dmatrix_t> inv_vertex_matrices;
		std::vector<dmatrix_t> delta_matrices;
		dmatrix_t equal_time_gf;
		dmatrix_t time_displaced_gf;
		dmatrix_t proj_W_l;
		dmatrix_t proj_W_r;
		dmatrix_t proj_W;
		dmatrix_t gf_buffer;
		dmatrix_t W_l_buffer;
		dmatrix_t W_r_buffer;
		dmatrix_t W_buffer;
		int dir_buffer;
		int gf_buffer_tau;
		dmatrix_t id;
		dmatrix_t id_2;
		dmatrix_t expH0;
		dmatrix_t invExpH0;
		dmatrix_t P;
		dmatrix_t Pt;
		dmatrix_t delta;
		dmatrix_t delta_W_r;
		dmatrix_t W_W_l;
		dmatrix_t M;
		std::pair<int, int> last_flip;
		std::vector<std::map<int, int>> cb_bonds;
		std::vector<std::vector<std::pair<int, int>>> nn_bonds;
		std::vector<std::vector<std::pair<int, int>>> inv_nn_bonds;
		stabilizer_t stabilizer;
};
