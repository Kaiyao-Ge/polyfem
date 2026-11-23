#pragma once

#include "Form.hpp"

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/utils/MatrixCache.hpp>
#include <polyfem/utils/Types.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace polyfem::solver
{
	enum class ElementInversionCheck
	{
		Discrete,
		Conservative,
	};
	NLOHMANN_JSON_SERIALIZE_ENUM(
		polyfem::solver::ElementInversionCheck,
		{{ElementInversionCheck::Discrete, "Discrete"},
		 {ElementInversionCheck::Conservative, "Conservative"}})

	/// @brief Form of the elasticity potential and forces
	class ElasticForm : public Form
	{
		friend class ElasticForceDerivative;

	public:
		/// @brief Construct a new Elastic Form object
		/// @param state Reference to the simulation state
		ElasticForm(const int n_bases,
					std::vector<basis::ElementBases> &bases,
					const std::vector<basis::ElementBases> &geom_bases,
					const assembler::Assembler &assembler,
					assembler::AssemblyValsCache &ass_vals_cache,
					const double t, const double dt,
					const bool is_volume,
					const double jacobian_threshold = 0.,
					const ElementInversionCheck check_inversion = ElementInversionCheck::Discrete,
					const unsigned conservative_max_iter = 100000);

		std::string name() const override { return "elastic"; }

	protected:
		/// @brief Compute the elastic potential value
		/// @param x Current solution
		/// @return Value of the elastic potential
		virtual double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the value of the form multiplied with the weigth
		/// @param x Current solution
		/// @return Computed value
		Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		/// @brief Determine if a step from solution x0 to solution x1 is allowed
		/// @param x0 Current solution
		/// @param x1 Proposed next solution
		/// @return True if the step is allowed
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Checks if the step is inversion free
		/// @return True if the step is inversion free else false
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Update time-dependent fields
		/// @param t Current time
		/// @param x Current solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override
		{
			t_ = t;
			x_prev_ = x;
		}

		/// @brief Determine the maximum step size allowable between the current and next solution
		/// @param x0 Current solution (step size = 0)
		/// @param x1 Next solution (step size = 1)
		/// @return Maximum allowable step size
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Reset adaptive quadrature refinement after each complete nonlinear solve.
		void finish() override;

		/// @brief Commit the quadrature refinement candidate found by the
		/// most recent max_step_size() call, once the step is accepted.
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		/// @brief Local derivatives of one element, same quadrature, weights and scaling (weight / scale) as the form
		struct ElementLocal
		{
			std::vector<int> dofs;   ///< full dof of each local dof
			std::vector<double> w;   ///< weight of the local dof in its full dof (x_full[dofs[j]] = w[j] * x_local[j] up to the conforming map)
			Eigen::VectorXd g;       ///< local gradient
			Eigen::MatrixXd H;       ///< local Hessian (unprojected), only if requested
			double E = 0;            ///< local energy, only if requested
		};

		/// @brief True if the assembler provides local element derivatives with one full dof per local dof
		bool supports_local() const;
		int n_elements() const { return int(bases_.size()); }
		/// @brief Use the closed-form kernel for linear tetrahedra with a NeoHookean material (FastP1.hpp) for the derivatives, the energy and
		/// the local element quantities. It changes the rounding of the elastic terms (about 1e-13 relative). Falls back to the assembler (with a
		/// warning) where it does not apply: other materials, other elements, material parameters that depend on position or time. Call before
		/// the form is used.
		void enable_fast_p1(const bool on) { fast_requested_ = on; }

		/// @brief Local gradient (and Hessian, energy) of element e at x (full dofs, as a column matrix)
		void element_local(const Eigen::MatrixXd &x, const int e, const bool with_hessian, const bool with_energy,
						   assembler::ElementAssemblyValues &vals, ElementLocal &out) const;

		/// @brief Value (weighted and scaled like the form) of the elements ids only
		double subset_energy(const Eigen::VectorXd &x, const std::vector<int> &ids) const;
		/// @brief Gradient (full dofs, weighted and scaled like the form) of the elements ids only
		void subset_gradient(const Eigen::VectorXd &x, const std::vector<int> &ids, Eigen::VectorXd &grad) const;
		/// @brief Hessian (full dofs, weighted and scaled like the form, local blocks projected to PSD as PolyFEM does if project)
		/// of the elements ids only
		void subset_hessian(const Eigen::VectorXd &x, const std::vector<int> &ids, const bool project, StiffnessMatrix &hessian) const;

	private:
		struct FastP1Data;
		bool fast_requested_ = false;
		mutable std::shared_ptr<FastP1Data> fast_;
		mutable std::once_flag fast_once_;
		mutable bool fast_on_ = false;
		bool fast_active() const;
		void fast_init() const;
		double fast_energy(const Eigen::VectorXd &x) const;
		void fast_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const;
		void fast_hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const;

		const int n_bases_;
		std::vector<basis::ElementBases> &bases_;
		const std::vector<basis::ElementBases> &geom_bases_;

		const assembler::Assembler &assembler_; ///< Reference to the assembler
		assembler::AssemblyValsCache &ass_vals_cache_;
		double t_;
		const double jacobian_threshold_;
		const ElementInversionCheck check_inversion_;
		const unsigned conservative_max_iter_;
		const double dt_;
		const bool is_volume_;

		StiffnessMatrix cached_stiffness_;                      ///< Cached stiffness matrix for linear elasticity
		mutable std::unique_ptr<utils::MatrixCache> mat_cache_; ///< Matrix cache (mutable because it is modified in second_derivative_unweighted)

		/// @brief Compute the stiffness matrix (cached)
		void compute_cached_stiffness();

		/// @brief Merge a subdivision tree into quadrature_hierarchy_ for the
		/// given element and rebuild its live quadrature if it actually refined.
		/// Shared by post_step() (deferred commit, once a step is accepted) and
		/// max_step_size() (immediate commit, only when step == 0 -- such a step
		/// can never reach post_step(), so deferring it would just discard the
		/// refinement candidate and leave the next iteration stuck on the same
		/// too-coarse quadrature).
		void commit_refinement(const int id, utils::Tree &subdivision_tree) const;

		Eigen::VectorXd x_prev_;

		mutable std::vector<utils::Tree> quadrature_hierarchy_;
		int quadrature_order_;

		/// @brief Pending (element id, subdivision tree) candidate from the
		/// last max_step_size() call; replaced (not accumulated) each call,
		/// committed in post_step() once a step is accepted.
		mutable std::optional<std::pair<int, utils::Tree>> pending_refinement_;
	};
} // namespace polyfem::solver
