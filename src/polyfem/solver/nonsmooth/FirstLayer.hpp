#pragma once

#include <polyfem/solver/NonsmoothTerm.hpp>

#include <polysolve/nonlinear/NonsmoothPorts.hpp>

#include <Eigen/Core>

#include <memory>
#include <vector>

namespace polyfem::solver
{
	class NLProblem;
	class FastTetValidity;

	/// @brief The elements that the first layer made exact in the energy split, kept for the rest of the time step (it outlives the terms,
	/// which are rebuilt for every solve; AL updates and friction lags do not clear it).
	class LiftSet : public NonsmoothSession
	{
	public:
		std::vector<int> ids; ///< sorted element ids

		void begin_epoch(const EnergyChange kind, const int /*time_step*/) override
		{
			if (kind == EnergyChange::time_step)
				ids.clear();
		}
	};

	/// @brief What the first layer promotes into: the exact units of a selection (the lift set of the energy split, the exact
	/// set of the blind selection). Elements are given by their id in the element table of FastTetValidity.
	class UnitSink
	{
	public:
		virtual ~UnitSink() = default;
		virtual bool is_exact_element(const int e) const = 0;
		/// @brief Makes the elements exact (waiting for take_delta); returns how many were new
		virtual int add_elements(const std::vector<int> &ids) = 0;
		/// @brief Elements that violate a limit but could not be promoted (no rounds left): kept for the selection of the next outer iteration
		virtual void record_elements(const std::vector<int> &ids) {}
		/// @brief The collision pairs behind a cut by the fraction-to-boundary or max_step_size limiter become exact; returns
		/// how many were new (the energy split has all contact exact: none)
		virtual int add_limiting_contacts(const polysolve::nonlinear::LimitEvent &event) { return 0; }
		/// @brief The change of the model that the units made exact since the last call cause, at the outer iterate x
		virtual bool take_delta(const Eigen::VectorXd &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta) = 0;
	};

	/// @brief The first layer: when a limiter cuts an inner step, the units that caused the cut become exact. It reads only the
	/// feasibility services of the platform (element volume ratios, the limiting collision), never what energy a unit has.
	class FirstLayer : public polysolve::nonlinear::FirstLayerPort
	{
	public:
		FirstLayer(NLProblem &problem, const std::shared_ptr<const FastTetValidity> &validity, UnitSink &sink);

		int identify(const polysolve::nonlinear::LimitEvent &event) override;
		bool take_delta(const Eigen::VectorXd &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta) override;

	private:
		NLProblem &problem_;
		std::shared_ptr<const FastTetValidity> validity_;
		UnitSink &sink_;
		Eigen::VectorXd det_key_; ///< full point of dets_ (empty: none)
		Eigen::VectorXd dets_;    ///< det D per tet at det_key_
	};
} // namespace polyfem::solver
