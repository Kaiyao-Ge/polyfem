#pragma once

#include "UnitCache.hpp"

#include <polyfem/solver/NonsmoothTerm.hpp>

#include <polysolve/nonlinear/NonsmoothPorts.hpp>

#include <unordered_map>
#include <unordered_set>

namespace polyfem::solver
{
	/// @brief Parameters of the blind selection (solver/nonlinear/NonsmoothNewton/blind)
	struct BlindParams
	{
		bool second_layer = true;
		double eta = 0.1;
		double marginal_kappa = 0.1;
		bool third_layer = true;
		bool present_only = true;
		bool born_hold = true;
		bool exit_all = true; ///< exit_at = all_changes (else time_step)
		std::string al_stage = "newton";
		bool wall_enabled = false;
		double wall_threshold = 1.2;
		bool wall_reread_off = false;
		unsigned wall_counted = 63; ///< bit i: guard, valid, ftb, ccd, jfloor, jbound

		static BlindParams from_json(const json &nonlinear_params);
	};

	/// @brief State of the blind selection that lives across solves (the terms are rebuilt for every solve): the exact set S (unit
	/// key -> the epoch in which it was last charged), the units born exact, the unit cache, the epoch bookkeeping, the Jg gate and the
	/// counts of the wall switch.
	class BlindSession : public NonsmoothSession
	{
	public:
		using Vec = Eigen::VectorXd;

		explicit BlindSession(const BlindParams &params);

		const BlindParams &params() const { return params_; }

		// --- epochs: stretches of the simulation in which the energy function is one ---
		void begin_epoch(const EnergyChange kind, const int time_step) override;
		long generation() const { return gen_; }
		void bump_generation() { ++gen_; }
		int epoch() const { return epoch_; }
		int time_step() const { return time_step_; }

		// --- the exact set ---
		bool in_S(const UnitKey &k) const { return !S_.empty() && S_.count(k) != 0; }
		int size() const { return int(S_.size()); }
		const std::unordered_map<UnitKey, int, UnitKeyHash> &units() const { return S_; }
		/// @brief Makes the units exact (charged in the current epoch); with pending they wait for the in-place update of the model
		/// (take_promoted). Returns the number of units that were not exact before.
		int add_keys(const std::vector<UnitKey> &keys, const bool pending);
		std::vector<UnitKey> take_promoted();
		void clear_promoted() { promoted_.clear(); }
		void erase(const UnitKey &k)
		{
			if (S_.erase(k) > 0)
				++ver_;
		}
		/// @brief Changes with the membership of S
		long version() const { return ver_; }
		/// @brief charges a unit that is exact in the current epoch
		void charge(const UnitKey &k) { S_[k] = epoch_; }

		// --- born units (SEL3-BL) ---
		void mark_born(const std::vector<UnitKey> &keys);
		bool is_born(const UnitKey &k) const { return born_.count(k) != 0; }
		/// @brief Removes the born units that are not units of the anchor any more (with them from S); returns how many
		int prune_born(const CollData &anchor_coll);

		UnitCache &cache() { return cache_; }

		// --- the first layer's units that wait for the in-place update ---

		// --- Jg ---
		polysolve::nonlinear::JGate &jgate() { return jgate_; }

		// --- wall switch ---
		/// @brief From the report of an outer iteration (cuts by limiter, accepted inner steps) of the current time step
		void on_outer_end(const long *cuts, const long inner);
		bool wall_now() const { return wall_now_; }

	private:
		BlindParams params_;
		std::unordered_map<UnitKey, int, UnitKeyHash> S_;
		std::unordered_set<UnitKey, UnitKeyHash> born_;
		std::vector<UnitKey> promoted_;
		UnitCache cache_;
		polysolve::nonlinear::JGate jgate_;
		long gen_ = 0;
		long ver_ = 0;
		int epoch_ = 0;
		int time_step_ = -1;
		bool fresh_ = false; ///< a real change of the energy was reported since the last solve started
		long cuts_ = 0, inner_ = 0; ///< wall switch: cuts by limiter and accepted inner steps in the current time step
		bool wall_now_ = false;
	};
} // namespace polyfem::solver
