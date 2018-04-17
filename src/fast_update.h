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
#include "vector_wick_static_base.h"
#include "vector_wick_base.h"

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
		using numeric_t = std::complex<double>;
		//using numeric_t = double;
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
			//Eigen::ComplexEigenSolver<dmatrix_t> solver(H0);
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
				//get_momentum_trial_wavefunction(H0);
				stabilizer.set_P(P, Pt);
			}
			stabilizer.set_method(param.use_projector);
		}

		dmatrix_t symmetrize_EV(const dmatrix_t& S, const Eigen::VectorXd& en, const dmatrix_t& pm)
		{
			dmatrix_t S_s = S + pm * S;
			dmatrix_t S_a = S - pm * S;
			dmatrix_t S_so(n_matrix_size, S_s.cols());
			dmatrix_t S_ao(n_matrix_size, S_s.cols());
			dmatrix_t S_f = dmatrix_t::Zero(n_matrix_size, 2*S_s.cols());

			for (int i = 0; i < S_s.cols(); ++i)
			{
				if (S_s.col(i).norm() > param.epsilon)
					S_s.col(i) /= S_s.col(i).norm();
				else
					S_s.col(i) *= 0.;
				if (S_a.col(i).norm() > param.epsilon)
					S_a.col(i) /= S_a.col(i).norm();
				else
					S_a.col(i) *= 0.;
			}

			int cnt = 0;
			for (int i = 0; i < S_s.cols(); ++i)
			{
				int j;
				for (j = i; j < S_s.cols() && std::abs(en(j)-en(i)) < param.epsilon ; ++j)
				{
					S_so.col(j) = S_s.col(j);
					S_ao.col(j) = S_a.col(j);
					for (int k = i; k < j; ++k)
					{
						S_so.col(j) -= S_so.col(k) * (S_so.col(k).adjoint() * S_s.col(j));
						S_ao.col(j) -= S_ao.col(k) * (S_ao.col(k).adjoint() * S_a.col(j));
					}
					//std::cout << "E=" << en(i) << ", orth: i=" << i << ", j=" << j << ": " << S_so.col(j).norm() << " " << S_ao.col(j).norm() << std::endl;
					if (S_so.col(j).norm() > param.epsilon)
					{
						S_so.col(j) /= S_so.col(j).norm();
						S_f.col(cnt) = S_so.col(j);
						++cnt;
					}
					if (S_ao.col(j).norm() > param.epsilon)
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
		
		dmatrix_t ph_symmetrize_EV(const dmatrix_t& S, const dmatrix_t& pm, const dmatrix_t& inv_pm)
		{
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
				if (S_s.col(i).norm() > param.epsilon)
					S_s.col(i) /= S_s.col(i).norm();
				else
					S_s.col(i) *= 0.;
				if (S_a.col(i).norm() > param.epsilon)
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
						S_so.col(j) -= S_so.col(k) * (S_so.col(k).adjoint() * S_s.col(j));
						S_ao.col(j) -= S_ao.col(k) * (S_ao.col(k).adjoint() * S_a.col(j));
					}
					//std::cout << "orth: i=" << i << ", j=" << j << ": " << S_so.col(j).norm() << " " << S_ao.col(j).norm() << std::endl;
					if (S_so.col(j).norm() > param.epsilon)
					{
						S_so.col(j) /= S_so.col(j).norm();
						S_sf.col(cnt_s) = S_so.col(j);
						++cnt_s;
					}
					if (S_ao.col(j).norm() > param.epsilon)
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
		
		std::vector<std::vector<int>> get_energy_blocks(const Eigen::VectorXd& en)
		{
			std::vector<std::vector<int>> energy_blocks;
			energy_blocks.push_back({0, n_matrix_size-1});
			for (int i = 1; i < n_matrix_size/2; ++i)
			{
				if (std::abs(en(i) - en(energy_blocks.back()[0])) > param.epsilon)
					energy_blocks.push_back(std::vector<int>());
				energy_blocks.back().push_back(i);
				energy_blocks.back().push_back(n_matrix_size-1-i);
			}
			for (int i = 0; i < energy_blocks.size(); ++i)
				std::sort(energy_blocks[i].begin(), energy_blocks[i].end());
			return energy_blocks;
		}
		
		dmatrix_t symmetrize_ph_blocks(const dmatrix_t& S, const std::vector<std::vector<int>>& energy_blocks, const Eigen::VectorXd& en, const dmatrix_t& pm)
		{
			dmatrix_t S_s = S + pm * S;
			dmatrix_t S_a = S - pm * S;
			dmatrix_t S_so(n_matrix_size, S_s.cols());
			dmatrix_t S_ao(n_matrix_size, S_s.cols());
			dmatrix_t S_f = dmatrix_t::Zero(n_matrix_size, 2*S_s.cols());

			for (int i = 0; i < S_s.cols(); ++i)
			{
				if (S_s.col(i).norm() > param.epsilon)
					S_s.col(i) /= S_s.col(i).norm();
				else
					S_s.col(i) *= 0.;
				if (S_a.col(i).norm() > param.epsilon)
					S_a.col(i) /= S_a.col(i).norm();
				else
					S_a.col(i) *= 0.;
			}
			/*
			for (int i = 0; i < S_s.cols(); ++i)
				std::cout << "i = " << i << ", E = " << en(i) << std::endl;
			for (int i = 0; i < energy_blocks.size(); ++i)
			{
				std::cout << "block " << i << " : ";
				for (int j = 0; j < energy_blocks[i].size(); ++j)
					std::cout << energy_blocks[i][j] << ", ";
				std::cout << std::endl;
			}
			*/

			int cnt_s = 0, cnt_a = 0;
			for (int i = 0; i < energy_blocks.size(); ++i)
				for (int j = 0; j < energy_blocks[i].size(); ++j)
				{
					int b = energy_blocks[i][j];
					S_so.col(b) = S_s.col(b);
					S_ao.col(b) = S_a.col(b);
					for (int k = 0; k < j; ++k)
					{
						int a = energy_blocks[i][k];
						S_so.col(b) -= S_so.col(a) * (S_so.col(a).adjoint() * S_s.col(b));
						S_ao.col(b) -= S_ao.col(a) * (S_ao.col(a).adjoint() * S_a.col(b));
					}
					std::cout << "E=" << en(b) << ", orth: i=" << i << ", b=" << b << ": " << S_so.col(b).norm() << " " << S_ao.col(b).norm() << std::endl;
					if (S_so.col(b).norm() > param.epsilon)
					{
						S_so.col(b) /= S_so.col(b).norm();
						S_f.col(cnt_s) = S_so.col(b);
						++cnt_s;
					}
					if (S_ao.col(b).norm() > param.epsilon)
					{
						S_ao.col(b) /= S_ao.col(b).norm();
						S_f.col(n_matrix_size/2+cnt_a) = S_ao.col(b);
						++cnt_a;
					}
				}
			if (cnt_s != S.cols()/2 || cnt_a != S.cols()/2)
			{
				std::cout << "Error! Found " << cnt_s << " symmetric states out of " << 2*S.cols() << std::endl;
				std::cout << "Error! Found " << cnt_a << " antisymmetric states out of " << 2*S.cols() << std::endl;
				throw(std::runtime_error("Error in symmetrization. Wrong number of states."));
			}
			return S_f.leftCols(S.cols());
		}
		
		/*
		void select_ph_states(const dmatrix_t& S, const dmatrix_t& ev, const std::vector<std::vector<int>>& energy_blocks, const dmatrix_t& H, const dmatrix_t& pm)
		{
			for (int i = 0; i < n_matrix_size/2; ++i)
			{
				Eigen::VectorXcd u_s = (S.col(i) + S.col(n_matrix_size/2+i))/std::sqrt(2.);
				Eigen::VectorXcd u_a = (S.col(i) - S.col(n_matrix_size/2+i))/std::sqrt(2.);
				std::cout << "i = " << i << ", P_s = " << std::real(u_s.adjoint() * H * u_s) << std::endl;
				std::cout << "i = " << i << ", P_a = " << std::real(u_a.adjoint() * H * u_a) << std::endl;
			}
		}
		*/
		
		std::vector<std::vector<int>> get_energy_levels(const Eigen::VectorXd& en)
		{
			std::vector<std::vector<int>> energy_levels;
			energy_levels.push_back({0});
			for (int i = 1; i < n_matrix_size; ++i)
			{
				if (std::abs(en(i) - en(energy_levels.back()[0])) > param.epsilon)
					energy_levels.push_back(std::vector<int>());
				energy_levels.back().push_back(i);
			}
			for (int i = 0; i < energy_levels.size(); ++i)
				std::sort(energy_levels[i].begin(), energy_levels[i].end());
			return energy_levels;
		}
		
		dmatrix_t orthonormalize(const dmatrix_t& S)
		{
			dmatrix_t S_o = S, S_f = dmatrix_t::Zero(S.rows(), S.cols());

			std::cout << "start orthogonalize" << std::endl;
			
			for (int i = 0; i < S_o.cols(); ++i)
			{
				if (S_o.col(i).norm() > param.epsilon)
				{
					std::cout << "S_o norm = " << S_o.col(i).norm() << std::endl;
					S_o.col(i) /= S_o.col(i).norm();
				}
				else
					S_o.col(i) *= 0.;
			}

			for (int i = 0; i < S_o.cols(); ++i)
			{
				S_f.col(i) = S_o.col(i);
				for (int k = 0; k < i; ++k)
					S_f.col(i) -= S_f.col(k) * (S_f.col(k).adjoint() * S_o.col(i));
				
				std::cout << "S_f norm = " << S_f.col(i).norm() << std::endl;
				if (S_f.col(i).norm() > param.epsilon)
					S_f.col(i) /= S_f.col(i).norm();
			}
			
			std::cout << "end orthogonalize" << std::endl << std::endl;
			
			return S_f;
		}
		
		void split_quantum_numbers(std::vector<std::vector<int>>& energy_levels, const dmatrix_t& S, const dmatrix_t& pm)
		{
			for (int i = 0; i < energy_levels.size(); ++i)
			{
				std::vector<std::vector<int>> sub_levels;
				std::vector<numeric_t> quantum_numbers;
				sub_levels.push_back({energy_levels[i][0]});
				numeric_t q = S.col(energy_levels[i][0])
					.adjoint() * pm * S.col(energy_levels[i][0]);
				q = std::real(q);
				quantum_numbers.push_back(q);
				//std::cout << "level " << i << ", state " << 0
				//		<< ", q = " << q << std::endl;
				for (int j = 1; j < energy_levels[i].size(); ++j)
				{
					numeric_t q = S.col(energy_levels[i][j])
						.adjoint() * pm * S.col(energy_levels[i][j]);
					q = std::real(q);
					//std::cout << "level " << i << ", state " << j
					//	<< ", q = " << q << std::endl;
					int k;
					for (k = 0; k < quantum_numbers.size();)
						if (std::abs(quantum_numbers[k] - q) > param.epsilon)
							++k;
						else
							break;
					if (k == quantum_numbers.size())
					{
						quantum_numbers.push_back(q);
						sub_levels.push_back({energy_levels[i][j]});
					}
					else
						sub_levels[k].push_back(energy_levels[i][j]);
				}
				energy_levels.erase(energy_levels.begin() + i);
				for (int k = 0; k < quantum_numbers.size(); ++k)
					energy_levels.insert(energy_levels.begin()+i,
						sub_levels[sub_levels.size()-1-k]);
			}
			for (int i = 0; i < energy_levels.size(); ++i)
				std::sort(energy_levels[i].begin(), energy_levels[i].end());
		}
		
		dmatrix_t project_symmetry(const dmatrix_t& S, const std::vector<std::vector<int>>& energy_levels, const dmatrix_t& pm)
		{
			dmatrix_t S_f = dmatrix_t::Zero(n_matrix_size, S.cols());
			
			for (int i = 0; i < energy_levels.size(); ++i)
			{
				int N = energy_levels[i].size();
				dmatrix_t projP(N, N);
				dmatrix_t S_proj = dmatrix_t::Zero(n_matrix_size, N);
				for (int j = 0; j < N; ++j)
					for (int k = 0; k < N; ++k)
						projP(j, k) = S.col(energy_levels[i][j]).adjoint() * pm * S.col(energy_levels[i][k]);
					
				Eigen::ComplexEigenSolver<dmatrix_t> solver(projP);
				std::cout << "Projected eigenvalues: i = " << i << std::endl;
				for (int j = 0; j < N; ++j)
					std::cout << solver.eigenvalues()[j] << std::endl;
				for (int j = 0; j < N; ++j)
					for (int k = 0; k < N; ++k)
						S_proj.col(j) += solver.eigenvectors()(k, j) * S.col(energy_levels[i][k]);
				S_proj = orthonormalize(S_proj);
				for (int j = 0; j < N; ++j)
					S_f.col(energy_levels[i][j]) = S_proj.col(j);
			}
			return S_f;
		}
		
		dmatrix_t project_ph_symmetry(const dmatrix_t& S, const dmatrix_t& pm)
		{
			dmatrix_t S_f = dmatrix_t::Zero(n_matrix_size, S.cols());
			
			int N = S.cols();
			dmatrix_t projP(N, N);
			dmatrix_t S_proj = dmatrix_t::Zero(n_matrix_size, N);
			for (int j = 0; j < N; ++j)
				for (int k = 0; k < N; ++k)
					projP(j, k) = S.col(j).adjoint() * pm * S.col(k);
				
			Eigen::ComplexEigenSolver<dmatrix_t> solver(projP);
			std::cout << "PH Projected eigenvalues:" << std::endl;
			for (int j = 0; j < N; ++j)
				std::cout << solver.eigenvalues()[j] << std::endl;
			for (int j = 0; j < N; ++j)
				for (int k = 0; k < N; ++k)
				{
					S_proj.col(j) += solver.eigenvectors()(k, j) * S.col(k);
					if (std::imag(solver.eigenvectors()(k, j)) > param.epsilon)
						std::cout << "Imag value: " << solver.eigenvectors()(k, j) << std::endl;
				}
			S_proj = orthonormalize(S_proj);
			for (int j = 0; j < N; ++j)
				S_f.col(j) = S_proj.col(j);
			return S_f;
		}

		void print_representations(const dmatrix_t& S, const dmatrix_t& inv_pm, const dmatrix_t& sv_pm, const dmatrix_t& sh_pm,
			const dmatrix_t& rot60_pm, const dmatrix_t& rot120_pm, const dmatrix_t& ph_pm)
		{
			std::vector<dmatrix_t> rep(6, dmatrix_t::Zero(S.cols(), S.cols()));
			for (int i = 0; i < S.cols(); ++i)
				for (int j = 0; j < S.cols(); ++j)
				{
					rep[0](i, j) = S.col(i).adjoint() * inv_pm * S.col(j);
					rep[1](i, j) = S.col(i).adjoint() * sv_pm * S.col(j);
					rep[2](i, j) = S.col(i).adjoint() * sh_pm * S.col(j);
					rep[3](i, j) = S.col(i).adjoint() * rot60_pm * S.col(j);
					rep[4](i, j) = S.col(i).adjoint() * rot120_pm * S.col(j);
					rep[5](i, j) = S.col(i).adjoint() * ph_pm * S.col(j);
					
					for (int k = 0; k < 6; ++k)
						if (std::abs(rep[k](i, j)) < 1E-14)
							rep[k](i, j) = 0.;
				}
			std::cout << "rep inv_pm" << std::endl;
			for (int i = 0; i < rep[0].rows(); ++i)
			{
				for (int j = 0; j < rep[0].rows(); ++j)
					std::cout << rep[0](i, j) << " ";
				std::cout << std::endl;
			}
			std::cout << "rep sv_pm" << std::endl;
			for (int i = 0; i < rep[0].rows(); ++i)
			{
				for (int j = 0; j < rep[0].rows(); ++j)
					std::cout << rep[1](i, j) << " ";
				std::cout << std::endl;
			}
			std::cout << "rep sh_pm" << std::endl;
			for (int i = 0; i < rep[0].rows(); ++i)
			{
				for (int j = 0; j < rep[0].rows(); ++j)
					std::cout << rep[2](i, j) << " ";
				std::cout << std::endl;
			}
			std::cout << "rep rot60_pm" << std::endl;
			for (int i = 0; i < rep[0].rows(); ++i)
			{
				for (int j = 0; j < rep[0].rows(); ++j)
					std::cout << rep[3](i, j) << " ";
				std::cout << std::endl;
			}
			std::cout << "rep rot120_pm" << std::endl;
			for (int i = 0; i < rep[0].rows(); ++i)
			{
				for (int j = 0; j < rep[0].rows(); ++j)
					std::cout << rep[4](i, j) << " ";
				std::cout << std::endl;
			}
			std::cout << "rep ph_pm" << std::endl;
			for (int i = 0; i < rep[0].rows(); ++i)
			{
				for (int j = 0; j < rep[0].rows(); ++j)
					std::cout << rep[5](i, j) << " ";
				std::cout << std::endl;
			}
		}
		
		void print_energy_levels(const dmatrix_t& S, const Eigen::VectorXd& eigen_energies,
			const std::vector<std::vector<int>>& energy_levels,
			const dmatrix_t& inv_pm, const dmatrix_t& sv_pm, const dmatrix_t& sh_pm, const dmatrix_t& rot60_pm, const dmatrix_t& rot120_pm)
		{
			std::cout << "Single particle eigenvalues:" << std::endl;
			for (int k = 0; k < energy_levels.size(); ++k)
			{
				std::cout << "level " << k << ":" << std::endl;
				for (int j = 0; j < energy_levels[k].size(); ++j)
				{
					int i = energy_levels[k][j];
					std::cout << "E(" << i << ") = " << eigen_energies[i]
					<< ", P_inv = " << S.col(i).adjoint() * inv_pm * S.col(i)
					<< ", P_rot60 = " << S.col(i).adjoint() * rot60_pm * S.col(i)
					<< ", P_rot120 = " << S.conjugate().col(i).adjoint() * rot120_pm * S.conjugate().col(i)
					<< ", P_sv = " << S.conjugate().col(i).adjoint() * sv_pm * S.conjugate().col(i) 
					<< ", P_sh = " << S.conjugate().col(i).adjoint() * sh_pm * S.conjugate().col(i) << std::endl;
				}
				std::cout << "---" << std::endl;
			}
		}

		double slater_overlap(const dmatrix_t& p1, const dmatrix_t& p2)
		{
			return std::abs((p1.adjoint() * p2).determinant());
		}

		dmatrix_t get_orbital_basis(const dmatrix_t& H, std::vector<Eigen::Vector2d>& momentum, std::vector<double>& energy, std::vector<std::complex<double>>& form_factor)
		{
			momentum.clear();
			energy.clear();
			form_factor.clear();
			dmatrix_t k_orbital_basis = dmatrix_t::Zero(l.n_sites(), l.n_sites());
			std::complex<double> im = {0., 1.};
			
			for (int n = 0; n < param.Ly; ++n)
				for (int m = 0; m < param.Lx; ++m)
					for (int o = 0; o < 2; ++o)
						for (int j = 0; j < param.Ly; ++j)
							for (int i = 0; i < param.Lx; ++i)
							{
								//auto r = l.a1 * static_cast<double>(i) + l.a2 * static_cast<double>(j);
								auto r = l.real_space_coord(2*(j*param.Ly+i));
								auto k = l.b1 * static_cast<double>(m) / param.Lx + l.b2 * static_cast<double>(n) / param.Ly;
								//std::complex<double> alpha = std::exp(-im * k.dot(l.delta - 2.*l.center));
								std::complex<double> alpha = 1.;
								k_orbital_basis(2*(j*param.Lx+i)+o, 2*(n*param.Lx+m)+o) = std::exp(im * k.dot(r)) / std::sqrt(param.Lx * param.Ly) * alpha;
							}
			for (int n = 0; n < param.Ly; ++n)
				for (int m = 0; m < param.Lx; ++m)
				{
					auto k = l.b1 * static_cast<double>(m) / param.Lx + l.b2 * static_cast<double>(n) / param.Ly;
					std::complex<double> f_k = 1. + std::exp(-im * k.dot(l.a2 - l.a1)) + std::exp(im * k.dot(l.a1));
					momentum.push_back(k);
					energy.push_back(-std::abs(f_k));
					form_factor.push_back(f_k);
					momentum.push_back(k);
					energy.push_back(std::abs(f_k));
					form_factor.push_back(f_k);
				}
			return k_orbital_basis;
		}
		
		dmatrix_t get_band_basis(const dmatrix_t& H, std::vector<Eigen::Vector2d>& momentum, std::vector<double>& energy, std::vector<std::complex<double>>& form_factor)
		{
			dmatrix_t k_orbital_basis = get_orbital_basis(H, momentum, energy, form_factor);
			dmatrix_t k_band_basis = dmatrix_t::Zero(l.n_sites(), l.n_sites());
			std::complex<double> im = {0., 1.};

			for (int n = 0; n < param.Ly; ++n)
				for (int m = 0; m < param.Lx; ++m)
				{
					auto k = l.b1 * static_cast<double>(m) / param.Lx + l.b2 * static_cast<double>(n) / param.Ly;
					std::complex<double> f_k = 1. + std::exp(-im * k.dot(l.a2 - l.a1)) + std::exp(im * k.dot(l.a1));
					
					if (std::abs(f_k) > 1E-14)
					{
						k_band_basis.col(2*(n*param.Lx+m)) = std::sqrt(std::conj(f_k)) * k_orbital_basis.col(2*(n*param.Lx+m))
							+ std::sqrt(f_k) * k_orbital_basis.col(2*(n*param.Lx+m)+1);
						k_band_basis.col(2*(n*param.Lx+m)+1) = -std::sqrt(std::conj(f_k)) * k_orbital_basis.col(2*(n*param.Lx+m))
							+ std::sqrt(f_k) * k_orbital_basis.col(2*(n*param.Lx+m)+1);
					}
					else
					{
						k_band_basis.col(2*(n*param.Lx+m)) = k_orbital_basis.col(2*(n*param.Lx+m)) + k_orbital_basis.col(2*(n*param.Lx+m)+1);
						k_band_basis.col(2*(n*param.Lx+m)+1) = -k_orbital_basis.col(2*(n*param.Lx+m)) + k_orbital_basis.col(2*(n*param.Lx+m)+1);
					}
				}
			for (int i = 0; i < k_band_basis.cols(); ++i)
				k_band_basis.col(i) /= k_band_basis.col(i).norm();
			return k_band_basis;
		}
		
		void get_momentum_trial_wavefunction(const dmatrix_t& H)
		{
			dmatrix_t inv_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				ph_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				rot60_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				rot120_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				sv_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				sh_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (int i = 0; i < n_matrix_size; ++i)
			{
				inv_pm(i, l.inverted_site(i)) = 1.;
				sv_pm(i, l.reflected_v_site(i)) = 1.;
				sh_pm(i, l.reflected_h_site(i)) = 1.;
				rot60_pm(i, l.rotated_site(i, 60.)) = 1.;
				rot120_pm(i, l.rotated_site(i, 120.)) = 1.;
				ph_pm(i, i) = l.parity(i);
			}
			std::vector<numeric_t> total_quantum_numbers = {{1., 1., 1., 1., 1.}};
			std::vector<Eigen::Vector2d> momentum;
			std::vector<double> energy;
			std::vector<std::complex<double>> form_factor;
			
			auto k_orbital_basis = get_orbital_basis(H, momentum, energy, form_factor);
			auto k_band_basis = get_band_basis(H, momentum, energy, form_factor);
			std::vector<int> sort_indices(n_matrix_size/2);
			std::iota (sort_indices.begin(), sort_indices.end(), 0);
			std::sort(sort_indices.begin(), sort_indices.end(), [&](int i1, int i2) { return energy[2*i1] < energy[2*i2]; });
			dmatrix_t sorted_k_orbital_basis = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			dmatrix_t sorted_k_band_basis = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			Eigen::VectorXd en_vector = Eigen::VectorXd::Zero(n_matrix_size);
			for (int i = 0; i < n_matrix_size/2; ++i)
			{
				sorted_k_orbital_basis.col(2*i) = k_orbital_basis.col(2*sort_indices[i]);
				sorted_k_band_basis.col(2*i) = k_band_basis.col(2*sort_indices[i]);
				en_vector[2*i] = energy[2*sort_indices[i]];
				sorted_k_orbital_basis.col(2*i+1) = k_orbital_basis.col(2*sort_indices[i]+1);
				sorted_k_band_basis.col(2*i+1) = k_band_basis.col(2*sort_indices[i]+1);
				en_vector[2*i+1] = energy[2*sort_indices[i]+1];
				//std::cout << "E(" << 2*i << ") = " << energy[2*sort_indices[i]] << ", k = (" << momentum[2*sort_indices[i]][0] << ", " << momentum[2*sort_indices[i]][1]
				//	<< "), form_factor = " << form_factor[2*sort_indices[i]] << std::endl;
			}
			Eigen::Vector2d total_k = {0., 0.};
			for (int i = 0; i < n_matrix_size/2-2; ++i)
				total_k += momentum[2*sort_indices[i]];
			
			//std::cout << "Total momentum modulo dirac levels: (" << total_k[0] << ", " << total_k[1] << ")" << std::endl;
			//std::cout << "b1 = (" << l.b1[0] << ", " << l.b1[1] << "), b2 = (" << l.b2[0] << ", " << l.b2[1] << ")" << std::endl;
			
			P = dmatrix_t::Zero(n_matrix_size, n_matrix_size/2);
			for (int i = 0; i < n_matrix_size/2; ++i)
				P.col(i) = sorted_k_band_basis.col(2*i);
			/*
			dmatrix_t gs_levels = sorted_k_band_basis.block(0, 0, n_matrix_size, 2);
			dmatrix_t mid_levels = sorted_k_band_basis.block(0, 2, n_matrix_size, n_matrix_size-2);
			dmatrix_t dirac_levels = sorted_k_band_basis.block(0, n_matrix_size-4, n_matrix_size, 4);
			
			std::cout << "sorted orbital levels:" << std::endl << std::endl;
			print_representations(sorted_k_orbital_basis, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			std::cout << "----" << std::endl;
			std::cout << "gs_levels all bands:" << std::endl << std::endl;
			print_representations(gs_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			std::cout << "----" << std::endl;
			std::cout << "mid_levels all bands:" << std::endl << std::endl;
			print_representations(mid_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			std::cout << "----" << std::endl;
			std::cout << "dirac_levels both bands:" << std::endl << std::endl;
			//std::vector<std::vector<int>> energy_levels(1, {0, 1, 2, 3});
			//dirac_levels = project_symmetry(dirac_levels, energy_levels, rot120_pm);
			//split_quantum_numbers(energy_levels, dirac_levels, rot120_pm);
			//dirac_levels = project_symmetry(dirac_levels, energy_levels, sv_pm);
			//split_quantum_numbers(energy_levels, dirac_levels, sv_pm);
			print_representations(dirac_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			std::cout << "----" << std::endl;
			
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(H);
			dmatrix_t real_dirac_levels = solver.eigenvectors().block(0, n_matrix_size/2-2, n_matrix_size, 4);
			std::cout << "real_dirac_levels:" << std::endl;
			Eigen::VectorXd ph_ev = Eigen::VectorXd::Zero(4);
			real_dirac_levels = symmetrize_EV(real_dirac_levels, ph_ev, inv_pm);
			print_representations(real_dirac_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			std::cout << "----" << std::endl;
			
			dmatrix_t dirac_levels_ph(n_matrix_size, 4);
			dirac_levels_ph.col(0) = dirac_levels.col(0) - dirac_levels.col(1) + dirac_levels.col(2) - dirac_levels.col(3);
			dirac_levels_ph.col(1) = dirac_levels.col(0) + dirac_levels.col(1) + dirac_levels.col(2) + dirac_levels.col(3);
			dirac_levels_ph.col(2) = dirac_levels.col(2) + dirac_levels.col(3);
			dirac_levels_ph.col(3) = dirac_levels.col(2) - dirac_levels.col(3);
			for (int i = 0; i < 4; ++i)
				dirac_levels_ph.col(i) /= dirac_levels_ph.col(i).norm();
			
			std::cout << "dirac_levels_ph:" << std::endl;
			print_representations(dirac_levels_ph, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			std::cout << "----" << std::endl;
			
			//std::vector<std::vector<int>> energy_levels(1, {0, 1, 2, 3});
			//real_dirac_levels = project_symmetry(real_dirac_levels, energy_levels, rot120_pm);
			//split_quantum_numbers(energy_levels, real_dirac_levels, rot120_pm);
			real_dirac_levels = symmetrize_EV(real_dirac_levels, ph_ev, ph_pm);
			//real_dirac_levels = ph_symmetrize_EV(real_dirac_levels, ph_pm, inv_pm);
			std::cout << "real_dirac_levels after ph:" << std::endl;
			print_representations(real_dirac_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			std::cout << "----" << std::endl;

			dmatrix_t real_dirac_levels_ph(n_matrix_size, 4);
			real_dirac_levels_ph.col(0) = real_dirac_levels.col(0) + real_dirac_levels.col(3);
			real_dirac_levels_ph.col(1) = real_dirac_levels.col(2) + real_dirac_levels.col(1);
			real_dirac_levels_ph.col(2) = real_dirac_levels.col(0) - real_dirac_levels.col(3);
			real_dirac_levels_ph.col(3) = real_dirac_levels.col(2) - real_dirac_levels.col(1);
			for (int i = 0; i < 4; ++i)
				real_dirac_levels_ph.col(i) /= real_dirac_levels_ph.col(i).norm();

			std::cout << "real_dirac_levels_ph:" << std::endl;
			print_representations(real_dirac_levels_ph, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			std::cout << "----" << std::endl;
			
			P.col(n_matrix_size/2-2) = dirac_levels_ph.col(0);
			P.col(n_matrix_size/2-1) = dirac_levels_ph.col(1);
			*/
			
			//print_representations(sorted_k_band_basis, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			//P.col(0) = sorted_k_band_basis.col(1);
			Pt = P.adjoint();
		}
		
		void get_trial_wavefunction(const dmatrix_t& H)
		{
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(H);
			dmatrix_t inv_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				ph_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				rot60_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				rot120_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				sv_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size),
				sh_pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (int i = 0; i < n_matrix_size; ++i)
			{
				inv_pm(i, l.inverted_site(i)) = 1.;
				sv_pm(i, l.reflected_v_site(i)) = 1.;
				sh_pm(i, l.reflected_h_site(i)) = 1.;
				rot60_pm(i, l.rotated_site(i, 60.)) = 1.;
				rot120_pm(i, l.rotated_site(i, 120.)) = 1.;
				ph_pm(i, i) = l.parity(i);
			}
			
			std::vector<numeric_t> total_quantum_numbers = {{1., 1., 1., 1., 1.}};
			std::vector<numeric_t> ph_2p_parity(4);
			std::vector<std::vector<int>> energy_levels = get_energy_levels(solver.eigenvalues());
			
			auto S_f = symmetrize_EV(solver.eigenvectors(), solver.eigenvalues(), inv_pm);
			
			/*
			auto S_f = solver.eigenvectors();
			
			std::cout << "Project symmetry P_inv" << std::endl;
			S_f = project_symmetry(S_f, energy_levels, inv_pm);
			split_quantum_numbers(energy_levels, S_f, inv_pm);
			//print_energy_levels(S_f, solver.eigenvalues(), energy_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm);
			
			std::cout << "Project symmetry P_sv" << std::endl;
			S_f = project_symmetry(S_f, energy_levels, sv_pm);
			split_quantum_numbers(energy_levels, S_f, sv_pm);
			//print_energy_levels(S_f, solver.eigenvalues(), energy_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm);
		
			std::cout << "Project symmetry P_sh" << std::endl;
			S_f = project_symmetry(S_f, energy_levels, sh_pm);
			split_quantum_numbers(energy_levels, S_f, sh_pm);
			//print_energy_levels(S_f, solver.eigenvalues(), energy_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm);
			
			std::cout << "Project symmetry P_rot60" << std::endl;
			S_f = project_symmetry(S_f, energy_levels, rot60_pm);
			split_quantum_numbers(energy_levels, S_f, rot60_pm);
			//print_energy_levels(S_f, solver.eigenvalues(), energy_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm);
			*/
			
			/*
			std::cout << "Project symmetry P_rot120" << std::endl;
			S_f = project_symmetry(S_f, energy_levels, rot120_pm);
			split_quantum_numbers(energy_levels, S_f, rot120_pm);
			print_energy_levels(S_f, solver.eigenvalues(), energy_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm);
			*/
			
			if (l.n_sites() % 3 != 0)
			{
				/*
				print_energy_levels(S_f, solver.eigenvalues(), energy_levels, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm);
				
				P = S_f.leftCols(n_matrix_size/2);
				for (int i = 0; i < n_matrix_size/2; ++i)
				{
					total_quantum_numbers[0] *= (S_f.col(i).adjoint() * inv_pm * S_f.col(i)).trace();
					total_quantum_numbers[1] *= (S_f.col(i).adjoint() * sv_pm * S_f.col(i)).trace();
					total_quantum_numbers[2] *= (S_f.col(i).adjoint() * sh_pm * S_f.col(i)).trace();
					total_quantum_numbers[3] *= (S_f.col(i).adjoint() * rot60_pm * S_f.col(i)).trace();
					total_quantum_numbers[4] *= (S_f.col(i).adjoint() * rot120_pm * S_f.col(i)).trace();
				}
				
				if (std::abs(param.inv_symmetry - total_quantum_numbers[0]) > param.epsilon)
				{
					total_quantum_numbers[0] /= (S_f.col(0).adjoint() * inv_pm * S_f.col(0)).trace();
					total_quantum_numbers[0] *= (S_f.col(n_matrix_size-1).adjoint() * inv_pm * S_f.col(n_matrix_size-1)).trace();
					total_quantum_numbers[1] /= (S_f.col(0).adjoint() * sv_pm * S_f.col(0)).trace();
					total_quantum_numbers[1] *= (S_f.col(n_matrix_size-1).adjoint() * sv_pm * S_f.col(n_matrix_size-1)).trace();
					total_quantum_numbers[2] /= (S_f.col(0).adjoint() * sh_pm * S_f.col(0)).trace();
					total_quantum_numbers[2] *= (S_f.col(n_matrix_size-1).adjoint() * sh_pm * S_f.col(n_matrix_size-1)).trace();
					P.col(0) = S_f.col(n_matrix_size-1);
				}

				print_representations(P, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
				
				Pt = P.adjoint();
				*/
				
				P = S_f.leftCols(n_matrix_size/2);
				//for (int i = 0; i < n_matrix_size/2; ++i)
				//	total_quantum_numbers[0] *= (S_f.col(i).adjoint() * inv_pm * S_f.col(i)).trace();
				//std::cout << "P = " << total_quantum_numbers[0] << std::endl;
				//std::cout << solver.eigenvalues()[n_matrix_size/2-1] << std::endl;
				Pt = P.adjoint();
				return;
			}
			else
			{
				for (int i = 0; i < n_matrix_size/2-2; ++i)
				{
					total_quantum_numbers[0] *= (S_f.col(i).adjoint() * inv_pm * S_f.col(i)).trace();
					total_quantum_numbers[1] *= (S_f.col(i).adjoint() * sv_pm * S_f.col(i)).trace();
					total_quantum_numbers[2] *= (S_f.col(i).adjoint() * sh_pm * S_f.col(i)).trace();
					total_quantum_numbers[3] *= (S_f.col(i).adjoint() * rot60_pm * S_f.col(i)).trace();
					total_quantum_numbers[4] *= (S_f.col(i).adjoint() * rot120_pm * S_f.col(i)).trace();
				}
			
				dmatrix_t ph_1p_block = S_f.block(0, n_matrix_size/2-2, n_matrix_size, 4);
				Eigen::VectorXd ph_ev = Eigen::VectorXd::Zero(4);
				
				ph_1p_block = symmetrize_EV(ph_1p_block, ph_ev, ph_pm);
				//ph_1p_block = project_ph_symmetry(ph_1p_block, ph_pm);
				
				ph_1p_block = ph_symmetrize_EV(ph_1p_block, ph_pm, inv_pm);

				for (int i = 0; i < ph_1p_block.cols(); ++i)
					ph_2p_parity[i] = ph_1p_block.col(i).adjoint() * ph_pm * ph_1p_block.col(i);
				std::vector<dmatrix_t> ph_2p_block(4, dmatrix_t(n_matrix_size, 2));
				
				//PH = -1
				ph_2p_block[0].col(0) = ph_1p_block.col(0);
				ph_2p_block[0].col(1) = ph_1p_block.col(3);
				ph_2p_block[1].col(0) = ph_1p_block.col(1);
				ph_2p_block[1].col(1) = ph_1p_block.col(2);
				//PH = 1
				ph_2p_block[2].col(0) = ph_1p_block.col(0);
				ph_2p_block[2].col(1) = ph_1p_block.col(1);
				ph_2p_block[3].col(0) = ph_1p_block.col(2);
				ph_2p_block[3].col(1) = ph_1p_block.col(3);
				
				ph_2p_parity = {-1., -1., 1., 1.};
				std::vector<std::vector<numeric_t>> e0_quantum_numbers = {4, std::vector<numeric_t>()};
				for (int i = 0; i < ph_2p_block.size(); ++i)
				{
					e0_quantum_numbers[i].push_back(ph_2p_block[i].col(0).adjoint() * inv_pm * ph_2p_block[i].col(0)
						* ph_2p_block[i].col(1).adjoint() * inv_pm * ph_2p_block[i].col(1));
					e0_quantum_numbers[i].push_back(ph_2p_block[i].col(0).adjoint() * sv_pm * ph_2p_block[i].col(0)
						* ph_2p_block[i].col(1).adjoint() * inv_pm * ph_2p_block[i].col(1));
					e0_quantum_numbers[i].push_back(ph_2p_block[i].col(0).adjoint() * sh_pm * ph_2p_block[i].col(0)
						* ph_2p_block[i].col(1).adjoint() * inv_pm * ph_2p_block[i].col(1));
					e0_quantum_numbers[i].push_back(ph_2p_block[i].col(0).adjoint() * rot60_pm * ph_2p_block[i].col(0)
						* ph_2p_block[i].col(1).adjoint() * inv_pm * ph_2p_block[i].col(1));
					e0_quantum_numbers[i].push_back(ph_2p_block[i].col(0).adjoint() * rot120_pm * ph_2p_block[i].col(0)
						* ph_2p_block[i].col(1).adjoint() * inv_pm * ph_2p_block[i].col(1));
				}
				
				/////////////////

				//auto energy_blocks = get_energy_blocks(solver.eigenvalues());
				//dmatrix_t ph_Np_block = S_f.block(0, 0, n_matrix_size, n_matrix_size);
				//dmatrix_t ph_Np_block = solver.eigenvectors().block(0, 0, n_matrix_size, n_matrix_size);
				//ph_Np_block = symmetrize_ph_blocks(ph_Np_block, energy_blocks, solver.eigenvalues(), ph_pm);
				//select_ph_states(ph_Np_block, S_f.block(0, 0, n_matrix_size, n_matrix_size), energy_blocks, H, ph_pm);

				/////////////////

				P.resize(n_matrix_size, n_matrix_size / 2);
				for (int i = 0; i < n_matrix_size/2-2; ++i)
					P.col(i) = S_f.col(i);
				
				/*
				for (int i = 0; i < ph_2p_block.size(); ++i)
					std::cout << "i = " << i << ": E = 0, inv_P = " << total_quantum_numbers[0] * e0_quantum_numbers[i][0]
						<< ", sv_P = " << total_quantum_numbers[1] * e0_quantum_numbers[i][1]
						<< ", sh_P = " << total_quantum_numbers[2] * e0_quantum_numbers[i][2]
						<< ", rot60_P = " << total_quantum_numbers[3] * e0_quantum_numbers[i][3]
						<< ", rot120_P = " << total_quantum_numbers[4] * e0_quantum_numbers[i][4]
						<< ", ph_P = " << ph_2p_parity[i] << std::endl;
				*/
				
				for (int i = 0; i < ph_2p_block.size(); ++i)
					if (std::abs(total_quantum_numbers[0] * e0_quantum_numbers[i][0] - param.inv_symmetry) < param.epsilon)
					{
						//std::cout << "Taken: i=" << i << std::endl;
						P.block(0, n_matrix_size/2-2, n_matrix_size, 2) = ph_2p_block[i];
						for (int j = 0; j < total_quantum_numbers.size(); ++j)
							total_quantum_numbers[j] *= e0_quantum_numbers[i][j];
						break;
					}

				/*
				P.block(0, n_matrix_size/2-2, n_matrix_size, 2) = ph_2p_block[param.slater];
				for (int j = 0; j < 5; ++j)
					total_quantum_numbers[j] *= e0_quantum_numbers[param.slater][j];
				*/

				Pt = P.adjoint();
				//print_representations(P, inv_pm, sv_pm, sh_pm, rot60_pm, rot120_pm, ph_pm);
			}
			if (std::abs(param.inv_symmetry - total_quantum_numbers[0]) > param.epsilon)
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
				for (auto& a : l.bonds("t3_bonds"))
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
				//for (auto& a : l.bonds("nearest neighbors"))
				//	H0(a.first+as, a.second+as) = -param.t;
				for (auto& a : l.bonds("t3_bonds"))
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
				
				/*
				numeric_t c = std::cosh(param.lambda * spin);
				numeric_t s = std::sinh(param.lambda * spin);
				numeric_t cp = std::cosh(-param.lambda * spin);
				numeric_t sp = std::sinh(-param.lambda * spin);
				*/
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
			//std::cout << "Start rebuilding..." << std::endl;
			if (aux_spins.size() == 0) return;
			if (param.use_projector)
			{
				stabilizer.set_proj_l(n_intervals, id);
				for (int n = n_intervals - 1; n >= 0; --n)
				{
					dmatrix_t b = propagator((n + 1) * param.n_delta,
						n * param.n_delta);
					stabilizer.set_proj_l(n, b);
					//if (n % 50 == 0)
					//	std::cout << "n_int = " << n << std::endl;
				}
				stabilizer.set_proj_r(0, id);
				for (int n = 1; n <= n_intervals; ++n)
				{
					dmatrix_t b = propagator(n * param.n_delta,
						(n - 1) * param.n_delta);
					stabilizer.set_proj_r(n, b);
					//if (n % 50 == 0)
					//	std::cout << "n_int = " << n << std::endl;
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
			//std::cout << "Done rebuilding..." << std::endl;
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
				//if (param.multiply_T)
				//	b *= expH0;
				
				
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
				
				
				
				//multiply_T_matrix();
			}
			else if (param.direction == -1)
			{
				//multiply_T_matrix();
				
				
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

		void measure_static_observable(const std::vector<std::string>& names,
			const std::vector<wick_static_base<dmatrix_t>>& obs,
			const std::vector<std::string>& vec_names,
			const std::vector<vector_wick_static_base<dmatrix_t>>& vec_obs)
		{
			if (param.use_projector)
			{
				dmatrix_t wl = proj_W * proj_W_l;
				equal_time_gf = id;
				equal_time_gf.noalias() -= proj_W_r * wl;
			}
			for (int i = 0; i < names.size(); ++i)
				measure.add(names[i], obs[i].get_obs(equal_time_gf));
			for (int i = 0; i < vec_names.size(); ++i)
				measure.add(vec_names[i], vec_obs[i].get_obs(equal_time_gf));

			if (param.mu != 0 || param.stag_mu != 0)
			{
				numeric_t n = 0.;
				for (int i = 0; i < l.n_sites(); ++i)
					n += equal_time_gf(i, i) / static_cast<double>(l.n_sites());
				measure.add("n_re", std::real(n*param.sign_phase));
				measure.add("n_im", std::imag(n*param.sign_phase));
				measure.add("n", std::real(n));
			}
		}

		void measure_dynamical_observable_ft(std::vector<std::vector<double>>& dyn_tau,
			const std::vector<std::string>& names,
			const std::vector<wick_base<dmatrix_t>>& obs,
			const std::vector<std::string>& vec_names,
			const std::vector<vector_wick_base<dmatrix_t>>& vec_obs)
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
					for (int i = 0; i < obs.size(); ++i)
						dyn_tau[i][t] = obs[i].get_obs(et_gf_0, equal_time_gf,
							time_displaced_gf);
					int cnt = 0;
					for (int i = 0; i < vec_obs.size(); ++i)
					{
						auto& values = vec_obs[i].get_obs(et_gf_0, equal_time_gf,
							time_displaced_gf);
						for (int j = 0; j < vec_obs[i].n_values; ++j)
						{
							dyn_tau[obs.size()+cnt][t] = values[j];
							++cnt;
						}
					}
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
		
		void get_obs_values(std::vector<std::vector<double>>& dyn_tau, int tau,
			const dmatrix_t& et_gf_0, const dmatrix_t& et_gf_t, const dmatrix_t& td_gf, 
			const std::vector<wick_base<dmatrix_t>>& obs, const std::vector<vector_wick_base<dmatrix_t>>& vec_obs)
		{
			for (int i = 0; i < obs.size(); ++i)
				dyn_tau[i][tau] = obs[i].get_obs(et_gf_0, et_gf_t, td_gf);
			int cnt = 0;
			for (int i = 0; i < vec_obs.size(); ++i)
			{
				auto& values = vec_obs[i].get_obs(et_gf_0, et_gf_t, td_gf);
				for (int j = 0; j < vec_obs[i].n_values; ++j)
				{
					dyn_tau[obs.size()+cnt][tau] = values[j];
					++cnt;
				}
			}
		}
		
		void measure_dynamical_observable_proj(std::vector<std::vector<double>>& dyn_tau,
			const std::vector<std::string>& names,
			const std::vector<wick_base<dmatrix_t>>& obs,
			const std::vector<std::string>& vec_names,
			const std::vector<vector_wick_base<dmatrix_t>>& vec_obs)
		{
			buffer_equal_time_gf();
			stabilizer.set_buffer();
			std::vector<dmatrix_t> et_gf_L(param.n_discrete_tau);
			std::vector<dmatrix_t> et_gf_R(2*param.n_discrete_tau);
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
			}
			
			get_obs_values(dyn_tau, 0, et_gf_0, et_gf_0, et_gf_0, obs, vec_obs);
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
				get_obs_values(dyn_tau, 2*n-1, et_gf_0, et_gf_R[2*n-2], time_displaced_gf, obs, vec_obs);
				
				time_displaced_gf = time_displaced_gf * g_r;
				get_obs_values(dyn_tau, 2*n, et_gf_0, et_gf_R[2*n-1], time_displaced_gf, obs, vec_obs);
			}

			reset_equal_time_gf_to_buffer();
			stabilizer.restore_buffer();
		}
		
		void measure_dynamical_observable_proj_2(std::vector<std::vector<double>>& dyn_tau,
			const std::vector<std::string>& names,
			const std::vector<wick_base<dmatrix_t>>& obs,
			const std::vector<std::string>& vec_names,
			const std::vector<vector_wick_base<dmatrix_t>>& vec_obs)
		{
			buffer_equal_time_gf();
			stabilizer.set_buffer();
			std::vector<dmatrix_t> et_gf_L(param.n_discrete_tau+1);
			std::vector<dmatrix_t> et_gf_R(param.n_discrete_tau+1);
			time_displaced_gf = id;
			
			if (tau == max_tau/2 + param.n_discrete_tau * param.n_dyn_tau)
				param.direction = -1;
			else if (tau == max_tau/2 - param.n_discrete_tau * param.n_dyn_tau)
				param.direction = 1;

			dmatrix_t wl = proj_W * proj_W_l;
			et_gf_L[0] = id;
			et_gf_L[0].noalias() -= proj_W_r * wl;
			for (int n = 1; n <= param.n_discrete_tau; ++n)
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
			}
			
			dmatrix_t& et_gf_0 = et_gf_L[param.n_discrete_tau];
			
			et_gf_R[0] = et_gf_0;
			for (int n = 1; n <= param.n_discrete_tau; ++n)
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
			}
			
			get_obs_values(dyn_tau, 0, et_gf_0, et_gf_0, et_gf_0, obs, vec_obs);
			for (int n = 1; n <= param.n_discrete_tau; ++n)
			{
				dmatrix_t g_l, g_r;
				if (param.direction == -1)
				{
					g_l = propagator(max_tau/2 + n*param.n_dyn_tau,
						max_tau/2 + (n-1)*param.n_dyn_tau) * et_gf_L[et_gf_L.size() - n];
					g_r = propagator(max_tau/2 - (n-1)*param.n_dyn_tau,
						max_tau/2 - n*param.n_dyn_tau) * et_gf_R[n];
						
					time_displaced_gf = g_l * time_displaced_gf;
					get_obs_values(dyn_tau, 2*n-1, et_gf_R[n-1], et_gf_L[et_gf_L.size() - n - 1], time_displaced_gf, obs, vec_obs);
					time_displaced_gf = time_displaced_gf * g_r;
					get_obs_values(dyn_tau, 2*n, et_gf_R[n], et_gf_L[et_gf_L.size() - n - 1], time_displaced_gf, obs, vec_obs);
				}
				else
				{
					g_l = et_gf_R[n] * propagator(max_tau/2 + n*param.n_dyn_tau,
						max_tau/2 + (n-1)*param.n_dyn_tau);
					g_r = et_gf_L[et_gf_L.size() - n] * propagator(max_tau/2 - (n-1)*param.n_dyn_tau,
						max_tau/2 - n*param.n_dyn_tau);
					
					time_displaced_gf = g_l * time_displaced_gf;
					get_obs_values(dyn_tau, 2*n-1, et_gf_L[et_gf_L.size() - n], et_gf_R[n], time_displaced_gf, obs, vec_obs);
					time_displaced_gf = time_displaced_gf * g_r;
					get_obs_values(dyn_tau, 2*n, et_gf_L[et_gf_L.size() - n - 1], et_gf_R[n], time_displaced_gf, obs, vec_obs);
				}
			}

			reset_equal_time_gf_to_buffer();
			stabilizer.restore_buffer();
		}
		
		void measure_dynamical_observable(std::vector<std::vector<double>>& dyn_tau,
			const std::vector<std::string>& names,
			const std::vector<wick_base<dmatrix_t>>& obs,
			const std::vector<std::string>& vec_names,
			const std::vector<vector_wick_base<dmatrix_t>>& vec_obs)
		{
			//check_td_gf_stability();
			if (param.use_projector)
				measure_dynamical_observable_proj_2(dyn_tau, names, obs, vec_names, vec_obs);
			else
				measure_dynamical_observable_ft(dyn_tau, names, obs, vec_names, vec_obs);
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

