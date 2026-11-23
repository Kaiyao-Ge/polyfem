#pragma once

#include "BlindSession.hpp"

#include <polysolve/nonlinear/NonsmoothPorts.hpp>

namespace polyfem::solver
{
	class NLProblem;
	class BlindUnitTerm;

	/// @brief The second and third layer of the blind selection. At the start of an outer iteration, after the factorization of B, it
	/// sees only anonymous units (key, local dofs, local gradient and Hessian at the anchors):
	///   second layer: the force errors of the units that are not exact over the step from the previous anchor are attributed with
	///   one back-solve with the factorization; the units with the largest shares become exact until the rest is below eta^2 times the
	///   residual (admitted in batches, with the marginal benefit rule);
	///   third layer: exact units stay exact across epochs while they are charged: selected by a shadow accounting that treats all units
	///   as unselected, or a Jacobian violation of the step.
	/// The state is in the session; the units that became exact are put into it (BlindSession::add_keys) and the term reads them off.
	class UnitSelector
	{
	public:
		using Vec = Eigen::VectorXd;

		UnitSelector(NLProblem &problem, BlindSession &session, BlindUnitTerm &term) : P_(problem), S_(session), term_(term) {}

		/// @brief Returns the number of units that became exact (they wait for take_promoted of the session)
		int select(const polysolve::nonlinear::SelectionPort::Context &context);

		/// @brief Violators of the Jacobian limits that the first layer could not promote (no rounds left); selected with the second layer
		void record_violators(const std::vector<int> &elements);

	private:
		NLProblem &P_;
		BlindSession &S_;
		BlindUnitTerm &term_;
		std::vector<int> viol_;
	};
} // namespace polyfem::solver
