#pragma once

#include "LocalTerms.hpp"

#include <polyfem/assembler/LocalCapture.hpp>

#include <limits>
#include <memory>
#include <vector>

namespace polyfem::solver
{
	class ElasticForm;

	/// @brief The units of one anchor as the decisions see them: key, local dofs (full coordinates), local gradient and raw Hessian,
	/// and the factor w the gradient and Hessian are to be multiplied with. Pointers into the arrays of the anchor.
	struct UnitTable
	{
		int n = 0;
		int n_fixed = 0; ///< the first n_fixed units (elements, nodes) have the same index at every anchor; the others (collision pairs) are found by key
		std::vector<UnitKey> key;
		std::vector<int> nd;
		std::vector<const int *> dof;
		std::vector<const double *> g;
		std::vector<const double *> H;
		std::vector<double> w;
	};

	/// @brief The local gradient and Hessian of every unit at the current and at the previous anchor (the point the model of an outer
	/// iteration is built at). The elastic elements are captured while the smooth part is assembled (assembler::LocalCapture); the
	/// barrier terms and the node terms are made by the term when the anchor is taken (BlindUnitTerm::blind_capture). The cache lives
	/// as long as the session (across solves).
	class UnitCache
	{
	public:
		using Vec = Eigen::VectorXd;

		struct Anchor
		{
			bool g_ok = false, h_ok = false; ///< the elements' gradients / Hessians are in g / H
			bool projected = false;          ///< the outer iteration started here used the projected smooth Hessian
			double res2 = std::numeric_limits<double>::quiet_NaN(); ///< g . B^-1 g of the outer started here (NaN: no selection ran there)
			double f = 1;                    ///< weight / scale of the elastic form
			Vec xf;                          ///< full point
			std::vector<double> g;           ///< nel * 12, unweighted
			std::vector<double> H;           ///< nel * 144, unweighted, unprojected, column-major
			bool u_ok = false;               ///< the barrier and node terms of xf are in coll / node
			long gen = -1;                   ///< the generation they were made in
			CollData coll;
			NodeData node;
		};

		/// @brief Tells the cache the elastic form (once per solve): sizes the tables; the cache is off for anything but P1 tetrahedra with unit dof weights
		void bind(const std::shared_ptr<ElasticForm> &elastic, const int dim, const Vec &xfull0);
		bool on() const { return nel_ > 0 && !off_; }
		int n_elements() const { return nel_; }
		const int *dofs(const int e) const { return dofs_.data() + std::size_t(e) * 12; }
		const UnitKey &element_key(const int e) const { return ekey_[e]; }

		/// @brief Starts the capture of the gradient (grad) or the Hessian of the smooth part assembled at the full point xf; a new
		/// point makes the last complete anchor the previous one. Returns the capture target for assembler::local_capture() (null: off).
		assembler::LocalCapture *capture_begin(const Vec &xf, const double f, const bool grad);
		void capture_end(const bool grad);
		/// @brief The current anchor holds the gradient / the Hessian of the point xf
		bool cur_has(const Vec &xf, const bool grad) const;
		/// @brief The anchor of the point xf (the previous complete anchor becomes the previous one if the point is new); idempotent
		/// for the same point. new_anchor tells whether the point was new.
		Anchor &anchor_for(const Vec &xf, const double f, bool *new_anchor = nullptr);

		Anchor &cur() { return a_[cur_]; }
		const Anchor &cur() const { return a_[cur_]; }
		const Anchor &prev() const { return a_[1 - cur_]; }
		double cur_f() const { return a_[cur_].f; }

		/// @brief The units of an anchor as a table
		void build_table(const Anchor &a, UnitTable &t) const;

	private:
		Anchor a_[2];
		int cur_ = 0;
		int nel_ = 0;
		bool off_ = false;
		std::vector<int> dofs_;       ///< nel * 12 full dofs of the local dofs
		std::vector<UnitKey> ekey_;   ///< nodes of every element
		assembler::LocalCapture cap_;
	};
} // namespace polyfem::solver
