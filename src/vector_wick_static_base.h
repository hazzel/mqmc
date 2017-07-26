#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>

template<typename matrix_t>
class vector_wick_static_base
{
	public:
		template<typename T>
		vector_wick_static_base(T&& functor)
		{
			construct_delegation(new typename std::remove_reference<T>::type(
				std::forward<T>(functor)));
		}
		
		vector_wick_static_base(vector_wick_static_base&& rhs) {*this = std::move(rhs);}
		vector_wick_static_base& operator=(vector_wick_static_base&& rhs) = default;
		vector_wick_static_base(const vector_wick_static_base& rhs) { std::cout << "copy c" << std::endl;}

		std::vector<double>& get_obs(const matrix_t& et_gf) const
		{ return get_obs_fun(et_gf); }
	private:
		template<typename T>
		void construct_delegation (T* functor)
		{
			impl = std::shared_ptr<T>(functor);
			get_obs_fun = [functor](const matrix_t& et_gf) -> std::vector<double>&
				{ return functor->get_obs(et_gf); };
		}
	private:
		std::shared_ptr<void> impl;
		std::function<std::vector<double>&(const matrix_t&)> get_obs_fun;
};
