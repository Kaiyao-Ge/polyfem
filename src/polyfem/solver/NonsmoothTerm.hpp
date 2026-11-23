#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>

#include <polysolve/nonlinear/NonsmoothPorts.hpp>
#include <polysolve/nonlinear/Problem.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <vector>

namespace polyfem::solver
{
	/// @brief What changed in the energy of the problem between two solves (NLProblem::energy_changed)
	enum class EnergyChange
	{
		time_step,    ///< new time step: inertia target, body force, boundary values, barrier stiffness, friction lag
		al_update,    ///< augmented Lagrangian multipliers or weight
		friction_lag, ///< lagged friction pairs, tangent bases and normal forces
		unknown       ///< a solve starts (NLProblem::init)
	};

	/// @brief State of a selection that lives across the solves of a simulation (the terms are rebuilt for every solve).
	class NonsmoothSession
	{
	public:
		virtual ~NonsmoothSession() = default;
		/// @brief The energy of the problem changed; `time_step` is the current time step
		virtual void begin_epoch(const EnergyChange kind, const int time_step) = 0;
	};

	/// @brief The nonsmooth term of an NLProblem (polysolve Problem::nonsmooth_*) in its current size: the splitting of the
	/// energy into a smooth part and a part that the NonsmoothNewton strategy models exactly. Two implementations
	/// (selection energy_split: BarrierContactNonsmoothTerm; selection blind: BlindUnitTerm).
	class NonsmoothTerm
	{
	public:
		virtual ~NonsmoothTerm() = default;

		virtual void smooth_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) = 0;
		virtual void smooth_hessian(const Eigen::VectorXd &x, const bool project_to_psd, StiffnessMatrix &hessian) = 0;
		virtual double smooth_valid_fraction(const Eigen::VectorXd &x, const Eigen::VectorXd &a, const Eigen::VectorXd &b) = 0;
		/// @brief Smallest element volume ratio det F at x (NaN without the fast element validity table)
		virtual double min_jacobian(const Eigen::VectorXd &x) const = 0;
		/// @brief min_jacobian(y) (if want_min) and the smallest det F(y) / det F(x) of an element (if want_ratio) in one sweep
		virtual void min_jacobians(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const bool want_min, const bool want_ratio,
								   double &min_j, double &min_ratio) = 0;
		virtual void metric_diagonal(Eigen::VectorXd &diag) const = 0;
		virtual void update(const Eigen::VectorXd &y) = 0;
		virtual double value(const Eigen::VectorXd &y) = 0;
		virtual void gradient(const Eigen::VectorXd &y, Eigen::VectorXd &grad) = 0;
		virtual void hessian(const Eigen::VectorXd &y, const bool project_to_psd, StiffnessMatrix &hessian) = 0;
		virtual bool hessian_blocks(const Eigen::VectorXd &y, Eigen::VectorXd &grad, std::vector<polysolve::nonlinear::HessianBlock> &blocks) = 0;
		virtual int active_size() const = 0;
		virtual void active_distances(const Eigen::VectorXd &a, const Eigen::VectorXd &b, Eigen::VectorXd &distance, Eigen::VectorXd &rate) = 0;
		virtual double step_length(const Eigen::VectorXd &a, const Eigen::VectorXd &b) = 0;
		virtual double length_scale() const = 0;
		virtual double activation_distance() const = 0;
		virtual void segment_begin(const Eigen::VectorXd &a, const Eigen::VectorXd &b) = 0;
		/// @brief True if max_step_size should go through the certified narrow phase (cached candidates installed)
		virtual bool certified_max_step_size() const = 0;
		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) = 0;
		/// @brief Problem-side counters for sim.json
		virtual json stats() const = 0;

		/// @brief The solver accepted the step x0 -> x1 (the term is at x1)
		virtual void step_accepted(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) {}

		/// @brief Ports of the strategy; null where the implementation has no such part
		virtual polysolve::nonlinear::FirstLayerPort *first_layer() { return nullptr; }
		virtual polysolve::nonlinear::SelectionPort *selection() { return nullptr; }
		virtual polysolve::nonlinear::BasisPort *basis() { return nullptr; }
	};
} // namespace polyfem::solver
