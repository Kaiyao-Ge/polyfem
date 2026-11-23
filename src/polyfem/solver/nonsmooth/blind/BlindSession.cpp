#include "BlindSession.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	namespace
	{
		template <typename T>
		T value_or(const json &j, const char *key, const T &def)
		{
			return j.is_object() && j.contains(key) ? j[key].get<T>() : def;
		}
	} // namespace

	BlindParams BlindParams::from_json(const json &nonlinear_params)
	{
		BlindParams o;
		if (!nonlinear_params.is_object() || !nonlinear_params.contains("NonsmoothNewton"))
			return o;
		const json &p = nonlinear_params["NonsmoothNewton"];
		if (!p.is_object())
			return o;
		if (p.contains("blind"))
		{
			const json &b = p["blind"];
			if (b.contains("second_layer"))
			{
				o.second_layer = value_or(b["second_layer"], "enabled", o.second_layer);
				o.eta = value_or(b["second_layer"], "eta", o.eta);
				o.marginal_kappa = value_or(b["second_layer"], "marginal_kappa", o.marginal_kappa);
			}
			if (b.contains("third_layer"))
				o.third_layer = value_or(b["third_layer"], "enabled", o.third_layer);
			o.present_only = value_or(b, "present_only", o.present_only);
			o.born_hold = value_or(b, "born_hold", o.born_hold);
			o.exit_all = value_or(b, "exit_at", std::string("all_changes")) == "all_changes";
			o.al_stage = value_or(b, "al_stage", o.al_stage);
			if (value_or(b, "unit_partition", std::string("local_term")) != "local_term")
				log_and_throw_error("NonsmoothNewton: blind/unit_partition must be local_term");
			if (b.contains("wall_switch"))
			{
				const json &w = b["wall_switch"];
				o.wall_enabled = value_or(w, "enabled", o.wall_enabled);
				o.wall_threshold = value_or(w, "threshold", o.wall_threshold);
				o.wall_reread_off = value_or(w, "reread_off", o.wall_reread_off);
				if (w.contains("counted"))
				{
					o.wall_counted = 0;
					static const char *names[6] = {"guard", "valid", "ftb", "ccd", "jfloor", "jbound"};
					for (const auto &c : w["counted"])
						for (int i = 0; i < 6; ++i)
							if (c.get<std::string>() == names[i])
								o.wall_counted |= 1u << i;
				}
			}
		}
		if (o.second_layer && !(o.eta > 0))
			log_and_throw_error("NonsmoothNewton: blind/second_layer/eta must be > 0");
		return o;
	}

	BlindSession::BlindSession(const BlindParams &params) : params_(params)
	{
		jgate_ = polysolve::nonlinear::JGate();
	}

	void BlindSession::begin_epoch(const EnergyChange kind, const int time_step)
	{
		if (kind == EnergyChange::unknown)
		{
			// a solve starts: if no change of the energy was reported since the last solve, the functions may still have changed
			// (the previous anchor is made again with the current ones)
			if (!fresh_)
				++gen_;
			fresh_ = false;
			return;
		}
		if (kind == EnergyChange::time_step && time_step == time_step_)
			return; // the same time step again
		fresh_ = true;
		const bool exits = kind == EnergyChange::time_step || params_.exit_all;
		if (kind == EnergyChange::time_step)
		{
			// the wall switch: the cuts of the time step that ends decide about the one that starts
			wall_now_ = params_.wall_enabled && inner_ > 0 && double(cuts_) / double(inner_) >= params_.wall_threshold;
			cuts_ = inner_ = 0;
			time_step_ = time_step;
		}
		if (exits)
		{
			// a unit that was not charged in the epoch that ends goes back to the quadratic model, unless it was born exact and born_hold
			for (auto it = S_.begin(); it != S_.end();)
				if (!(params_.third_layer && it->second == epoch_) && !(params_.born_hold && born_.count(it->first) != 0))
				{
					it = S_.erase(it);
					++ver_;
				}
				else
					++it;
			++epoch_;
		}
		promoted_.clear();
		++gen_;
	}

	int BlindSession::add_keys(const std::vector<UnitKey> &keys, const bool pending)
	{
		int n = 0;
		for (const UnitKey &k : keys)
		{
			const auto it = S_.find(k);
			if (it == S_.end())
			{
				S_.emplace(k, epoch_);
				++ver_;
				++n;
				if (pending)
					promoted_.push_back(k);
			}
			else
				it->second = epoch_;
		}
		return n;
	}

	std::vector<UnitKey> BlindSession::take_promoted()
	{
		std::vector<UnitKey> out;
		out.swap(promoted_);
		return out;
	}

	void BlindSession::mark_born(const std::vector<UnitKey> &keys)
	{
		// the born units stay exact across epochs only with born_hold (and leave when they are not units of the anchor)
		if (params_.born_hold)
			for (const UnitKey &k : keys)
				born_.insert(k);
	}

	int BlindSession::prune_born(const CollData &anchor_coll)
	{
		int n = 0;
		for (auto it = born_.begin(); it != born_.end();)
		{
			if (anchor_coll.find(*it) < 0)
			{
				S_.erase(*it);
				++ver_;
				it = born_.erase(it);
				++n;
			}
			else
				++it;
		}
		return n;
	}

	void BlindSession::on_outer_end(const long *cuts, const long inner)
	{
		for (int i = 0; i < 6; ++i)
			if (params_.wall_counted >> i & 1)
				cuts_ += cuts[i];
		inner_ += inner;
	}
} // namespace polyfem::solver
