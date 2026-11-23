#pragma once

#include "BlindSession.hpp"
#include "UnitSelector.hpp"

#include <polyfem/solver/BarrierContactNonsmoothTerm.hpp>

#include <ipc/collisions/normal/normal_collisions.hpp>

namespace polyfem::solver
{
	/// @brief The nonsmooth term of the blind selection (NonsmoothNewton/selection = blind). The problem is split into units (the
	/// local energy terms that depend on the same FE nodes: elastic elements, barrier collision pairs, node terms of the lumped
	/// inertia and the body force); the exact term holds the units of the exact set S of the session (elastic elements and
	/// collision pairs) and every collision pair that is new at a point, the smooth part is the rest: the forms' global assembly minus
	/// the exact elements, plus the collision pairs of the anchor that are not exact (their quadratic expansion at the anchor).
	///
	/// It uses the contact services of its base (collision set update, certified narrow phase, fraction to boundary, step length,
	/// dof maps); the energy split of the base is replaced by the units.
	class BlindUnitTerm : public BarrierContactNonsmoothTerm
	{
	public:
		BlindUnitTerm(NLProblem &problem,
					  const std::vector<std::shared_ptr<Form>> &forms,
					  const Eigen::VectorXd &mass_full,
					  const ipc::CollisionMesh &collision_mesh,
					  const Eigen::SparseMatrix<double> *displacement_map,
					  const std::shared_ptr<const FastTetValidity> &fast_validity,
					  const Options &opts,
					  const std::shared_ptr<BlindSession> &session);
		~BlindUnitTerm() override;

		void smooth_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override;
		void smooth_hessian(const Eigen::VectorXd &x, const bool project_to_psd, StiffnessMatrix &hessian) override;
		double value(const Eigen::VectorXd &y) override;
		void gradient(const Eigen::VectorXd &y, Eigen::VectorXd &grad) override;
		void hessian(const Eigen::VectorXd &y, const bool project_to_psd, StiffnessMatrix &hessian) override;
		bool hessian_blocks(const Eigen::VectorXd &y, Eigen::VectorXd &grad, std::vector<polysolve::nonlinear::HessianBlock> &blocks) override;
		void step_accepted(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		polysolve::nonlinear::FirstLayerPort *first_layer() override { return first_layer_blind_.get(); }
		polysolve::nonlinear::SelectionPort *selection() override { return &port_; }
		polysolve::nonlinear::BasisPort *basis() override { return nullptr; }

		/// @brief The previous anchor made again with the current functions (the energy changed since it was taken)
		bool blind_reeval(const UnitCache::Anchor &prev, UnitCache::Anchor &out, const Eigen::VectorXd &xcur);
		std::shared_ptr<const FastTetValidity> fast_validity() const { return fast_validity_; }

	private:
		using Vec = Eigen::VectorXd;

		/// @brief The selection port the strategy sees
		class Port : public polysolve::nonlinear::SelectionPort
		{
		public:
			explicit Port(BlindUnitTerm &t) : t_(t) {}
			int select(const Context &context, polysolve::nonlinear::NonsmoothDelta &delta) override;
			bool reread_allowed() const override;
			polysolve::nonlinear::JGate &jgate() override { return t_.session_->jgate(); }
			void jratio_exact(const int on) override;
			double min_ratio_exact(const Vec &x, const Vec &y) override;
			void outer_end(const long *cuts, const long inner) override;

		private:
			BlindUnitTerm &t_;
		};

		/// @brief What the first layer promotes into: units of the session
		class Sink : public ::polyfem::solver::UnitSink
		{
		public:
			explicit Sink(BlindUnitTerm &t) : t_(t) {}
			bool is_exact_element(const int e) const override;
			int add_elements(const std::vector<int> &ids) override;
			void record_elements(const std::vector<int> &ids) override { t_.selector_->record_violators(ids); }
			int add_limiting_contacts(const polysolve::nonlinear::LimitEvent &event) override;
			bool take_delta(const Vec &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta) override;

		private:
			BlindUnitTerm &t_;
		};

		/// the anchor of the full point xf with the barrier and node terms (made when it is new or of an earlier generation)
		UnitCache::Anchor &blind_capture(const Vec &xf);
		void fill_collisions(CollData &cd, const Vec &xfull);
		UnitKey collision_key(const ipc::NormalCollisions &cs, const size_t i) const;
		/// the collisions of the state that are exact: in S, or new at the state (not at the anchor)
		void blind_exact_list(const ipc::NormalCollisions &cs, std::vector<int> &idx) const;
		/// ids of the elastic elements in S (sorted), made again when S changed
		const std::vector<int> &exact_elements();
		void rebuild_jratio();
		/// the change of the model that the units made exact since the last call cause, at the anchor x
		bool promotion_delta(const Vec &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta);
		int ftb_limiting(const Vec &a, const Vec &b, const double tau);
		int ccd_limiting();

		std::shared_ptr<BlindSession> session_;
		std::unique_ptr<UnitSelector> selector_;
		std::unique_ptr<FirstLayer> first_layer_blind_;
		Port port_;
		Sink sink_;
		std::vector<std::shared_ptr<Form>> node_local_;
		std::unordered_map<UnitKey, int, UnitKeyHash> element_of_key_;
		std::vector<int> sel_ids_; ///< sorted elastic elements of S
		long sel_set_size_ = -1;   ///< the size of S that sel_ids_ was made for
		long sel_ver_ = -1;
		int jratio_dyn_ = -1;      ///< Jg: -1 not set, 0 the Jacobian bound sees the unselected elements only, 1 all
	};
} // namespace polyfem::solver
