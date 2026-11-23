#include "FirstLayer.hpp"

#include <polyfem/solver/FastTetValidity.hpp>
#include <polyfem/solver/NLProblem.hpp>

namespace polyfem::solver
{
	using polysolve::nonlinear::LimitEvent;

	FirstLayer::FirstLayer(NLProblem &problem, const std::shared_ptr<const FastTetValidity> &validity, UnitSink &sink)
		: problem_(problem), validity_(validity), sink_(sink)
	{
	}

	int FirstLayer::identify(const LimitEvent &event)
	{
		if (event.kind == LimitEvent::Kind::ftb || event.kind == LimitEvent::Kind::ccd)
			return event.promote ? sink_.add_limiting_contacts(event) : 0;
		if (event.kind != LimitEvent::Kind::jacobian_floor && event.kind != LimitEvent::Kind::jacobian_bound)
			return 0;
		if (!validity_ || event.a == nullptr || event.b == nullptr)
			return 0;
		// the elements that are not exact and violate the limit at the end point b of the step from the outer iterate a: below
		// the floor (J), or below the bound of the volume ratio to a
		const Eigen::VectorXd xf = problem_.reduced_to_full(*event.a);
		if (!(det_key_.size() == xf.size() && det_key_ == xf))
		{
			dets_ = validity_->dets(xf);
			det_key_ = xf;
		}
		std::vector<double> J, ratio;
		validity_->jacobian_arrays(dets_, problem_.reduced_to_full(*event.b), J, ratio);
		const bool floor = event.kind == LimitEvent::Kind::jacobian_floor;
		std::vector<int> violators;
		for (int e = 0; e < int(J.size()); ++e)
		{
			if (sink_.is_exact_element(e))
				continue;
			if (floor ? !(J[e] >= event.jacobian_floor) : (ratio[e] < event.jacobian_bound))
				violators.push_back(e);
		}
		if (violators.empty())
			return 0;
		if (!event.promote)
		{
			sink_.record_elements(violators);
			return 0;
		}
		return sink_.add_elements(violators);
	}

	bool FirstLayer::take_delta(const Eigen::VectorXd &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta)
	{
		return sink_.take_delta(x, projected, delta);
	}
} // namespace polyfem::solver
