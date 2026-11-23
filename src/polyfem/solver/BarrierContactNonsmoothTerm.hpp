#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/solver/NonsmoothTerm.hpp>
#include <polyfem/solver/nonsmooth/FirstLayer.hpp>

#include <polysolve/nonlinear/Problem.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <memory>
#include <string>
#include <vector>

namespace ipc
{
	class CollisionMesh;
}

namespace polyfem::solver
{
	class Form;
	class FrictionForm;
	class NLProblem;
	class BarrierContactForm;
	class BilateralSegmentCertifier;
	class FastTetValidity;
	class ElasticForm;

	/// @brief True if the nonlinear solver parameters select polysolve's NonsmoothNewton strategy
	bool selects_nonsmooth_newton(const json &nonlinear_params);

	/// @brief The barrier contact form (with the friction form, if any) as the nonsmooth term of an NLProblem (polysolve
	/// Problem::nonsmooth_*), in its current size: the reduced problem, or the full-size augmented Lagrangian subproblems.
	///
	/// Smooth part: all other forms (plus the augmented Lagrangian penalty in the full size); nonsmooth term: the (single)
	/// BarrierContactForm and the FrictionForm, if any. The state of the term is the collision set of the contact form;
	/// nonsmooth_update rebuilds it at the trial point (new and vanished collisions included). The friction form keeps
	/// its lagged pairs (fixed during a solve); its energy is evaluated exactly at the trial points, its Hessian is not
	/// projected, and in B of the first solve of a time step it is taken at the velocities of the previous step
	/// (NLProblem::take_friction_b_lag). With local blocks the contact (and friction) derivatives are formed per
	/// collision (12-vector gradient, 12x12 unprojected Hessian) and scattered to reduced dofs; this needs the reduced
	/// size, a plain selection reduced <- full map and a collision mesh whose vertices each follow exactly one FE node
	/// with weight 1 (checked once from the dof map and the collision displacement map). max_step_size on a segment with
	/// cached candidates goes through BilateralSegmentCertifier (same result as the form's max_step_size); element
	/// validity and the element volume ratios of the Jacobian limits through FastTetValidity where applicable.
	class BarrierContactNonsmoothTerm : public NonsmoothTerm, private UnitSink
	{
	public:
		/// @brief Problem-side switches (solver/nonlinear/NonsmoothNewton/debug)
		struct Options
		{
			bool certificate = true;       ///< AABB / U_tr certificate in front of TI CCD
			bool dedupe = true;            ///< drop provably-far swept candidates per segment
			bool local_blocks = true;      ///< per-collision contact derivatives (selection dof map only)
			bool fast_validity = true;     ///< tet-determinant validity with anchored sigma_min certificate
			int candidate_permutation = 0; ///< null control on the segment candidate order (0 off, 1 reverse, >= 2 seeded shuffle)
			bool first_layer = false;      ///< first layer: elastic elements that cause a cut become exact (first_layer/enabled = on)
			std::string lift_model = "exact";     ///< first_layer/lift_model: how a lifted element enters the model (exact, frozen, frozen_projected)
			std::string lift_basis = "projected"; ///< first_layer/lift_basis: how it enters B (projected, raw)

			static Options from_json(const json &nonlinear_params);
		};

		/// @param forms all forms of the problem (exactly one BarrierContactForm, at most one FrictionForm)
		/// @param mass_full lumped mass per full dof
		/// @param displacement_map collision full vertex <- FE node map of collision_mesh (null or empty: the collision full
		///                         vertices are the FE nodes)
		/// @param fast_validity null: not applicable (native validity check)
		BarrierContactNonsmoothTerm(NLProblem &problem,
									const std::vector<std::shared_ptr<Form>> &forms,
									const Eigen::VectorXd &mass_full,
									const ipc::CollisionMesh &collision_mesh,
									const Eigen::SparseMatrix<double> *displacement_map,
									const std::shared_ptr<const FastTetValidity> &fast_validity,
									const Options &opts,
									const std::shared_ptr<LiftSet> &lift = nullptr);
		~BarrierContactNonsmoothTerm() override;

		// --- polysolve::nonlinear::Problem nonsmooth callbacks (coordinates of the current problem size) ---
		void smooth_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override;
		void smooth_hessian(const Eigen::VectorXd &x, const bool project_to_psd, StiffnessMatrix &hessian) override;
		double smooth_valid_fraction(const Eigen::VectorXd &x, const Eigen::VectorXd &a, const Eigen::VectorXd &b) override;
		/// @brief Smallest element volume ratio det F at x (NaN without the fast element validity table)
		double min_jacobian(const Eigen::VectorXd &x) const override;
		/// @brief min_jacobian(y) (if want_min) and the smallest det F(y) / det F(x) of an element (if want_ratio) in one
		/// sweep; NaN where not requested or without the fast element validity table. The determinants at x are kept for
		/// the next calls with the same full point x.
		void min_jacobians(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const bool want_min, const bool want_ratio,
						   double &min_j, double &min_ratio) override;
		void metric_diagonal(Eigen::VectorXd &diag) const override;
		void update(const Eigen::VectorXd &y) override;
		double value(const Eigen::VectorXd &y) override;
		void gradient(const Eigen::VectorXd &y, Eigen::VectorXd &grad) override;
		void hessian(const Eigen::VectorXd &y, const bool project_to_psd, StiffnessMatrix &hessian) override;
		bool hessian_blocks(const Eigen::VectorXd &y, Eigen::VectorXd &grad, std::vector<polysolve::nonlinear::HessianBlock> &blocks) override;
		int active_size() const override;
		void active_distances(const Eigen::VectorXd &a, const Eigen::VectorXd &b, Eigen::VectorXd &distance, Eigen::VectorXd &rate) override;
		double step_length(const Eigen::VectorXd &a, const Eigen::VectorXd &b) override;
		double length_scale() const override { return h_edge_; }
		double activation_distance() const override;
		void segment_begin(const Eigen::VectorXd &a, const Eigen::VectorXd &b) override;

		/// @brief True if max_step_size should go through the certified narrow phase (cached candidates installed)
		bool certified_max_step_size() const override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		bool local_ok() const { return local_ok_; }

		/// @brief Problem-side counters (certified pairs, TI pairs, local blocks in use; with friction: lagged pairs, friction
		/// blocks of B and how many of them were taken at the previous velocities)
		json stats() const override;

		polysolve::nonlinear::FirstLayerPort *first_layer() override { return first_layer_.get(); }
		polysolve::nonlinear::BasisPort *basis() override;

		/// @brief The friction form of the term (null: the problem has no friction)
		const std::shared_ptr<FrictionForm> &friction() const { return friction_; }

	protected:
		// --- first layer in the energy split: the lifted elastic elements (the lift set of the time step) ---
		friend class LiftBasis;
		bool lift_exact() const;
		void init_lift();
		void sync_lift();
		bool is_exact_element(const int e) const override { return e >= 0 && e < int(lift_mask_.size()) && lift_mask_[e]; }
		int add_elements(const std::vector<int> &ids) override;
		bool take_delta(const Eigen::VectorXd &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta) override;
		/// @brief Elastic part of the exact term at yf: energy, gradient (full dofs) and Hessian blocks of the lifted elements
		void lift_blocks(const Eigen::VectorXd &yf, std::vector<polysolve::nonlinear::HessianBlock> &blocks, Eigen::VectorXd &grad) const;

		std::shared_ptr<LiftSet> lift_;
		std::shared_ptr<ElasticForm> elastic_;
		std::unique_ptr<FirstLayer> first_layer_;
		std::unique_ptr<polysolve::nonlinear::BasisPort> basis_port_;
		std::vector<int> pending_;       ///< lifted since the last take_delta
		std::vector<char> lift_mask_;    ///< per tet: lifted
		std::vector<char> jratio_mask_;  ///< per tet: counts for the Jacobian bound (the exact elements do not)

		Eigen::VectorXd to_reduced(const Eigen::VectorXd &full) const;
		void to_reduced(StiffnessMatrix &hessian) const;
		void init_local_map();
		bool init_node_of_vertex(const int nf);
		void ensure_anchor(const Eigen::VectorXd &x);
		double valid_fraction_native(const Eigen::VectorXd &a, const Eigen::VectorXd &b);

		NLProblem &problem_;
		std::shared_ptr<BarrierContactForm> contact_;
		std::shared_ptr<FrictionForm> friction_; ///< friction form of the term (null without friction)
		std::vector<std::shared_ptr<Form>> other_;
		const ipc::CollisionMesh &mesh_;
		const Eigen::SparseMatrix<double> *displacement_map_;
		std::shared_ptr<const FastTetValidity> fast_validity_;
		const Options opts_;
		Eigen::VectorXd mass_full_;
		std::unique_ptr<BilateralSegmentCertifier> certifier_;
		struct AnchorHolder;
		std::unique_ptr<AnchorHolder> anchor_;
		Eigen::VectorXd jdet_key_; ///< full point of jdet_ (empty: none)
		Eigen::VectorXd jdet_;     ///< det D per tet at jdet_key_ (min_jacobians)
		double h_edge_;
		bool local_ok_ = false;
		std::vector<int> red_of_full_;
		std::vector<int> node_of_vertex_; ///< collision vertex -> FE node
		long b_assemblies_ = 0;           ///< friction blocks assembled for B (hessian with project_to_psd)
		long b_at_lag_ = 0;               ///< of which at the velocities of the previous time step
	};
} // namespace polyfem::solver
