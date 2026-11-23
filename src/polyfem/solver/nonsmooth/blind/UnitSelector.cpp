#include "UnitSelector.hpp"

#include "BlindUnitTerm.hpp"

#include <polyfem/solver/FastTetValidity.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <ipc/utils/eigen_ext.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace polyfem::solver
{
	namespace
	{
		constexpr int NL = 12;
		// a force error below the rounding of its own terms is zero
		constexpr double FLOOR = 16 * std::numeric_limits<double>::epsilon();
	} // namespace

	void UnitSelector::record_violators(const std::vector<int> &elements)
	{
		viol_.insert(viol_.end(), elements.begin(), elements.end());
	}

	int UnitSelector::select(const polysolve::nonlinear::SelectionPort::Context &ctx)
	{
		using Anchor = UnitCache::Anchor;
		const BlindParams &par = S_.params();
		UnitCache &cache = S_.cache();
		const Vec &x = ctx.x;
		const double res2 = ctx.res2;
		const bool projected = ctx.projected;
		const bool l2 = par.second_layer, l3 = par.third_layer;
		const bool marginal = par.marginal_kappa > 0;

		std::vector<int> viol;
		viol.swap(viol_);
		std::sort(viol.begin(), viol.end());
		viol.erase(std::unique(viol.begin(), viol.end()), viol.end());
		std::vector<UnitKey> viol_keys;
		for (const int e : viol)
			viol_keys.push_back(cache.element_key(e)); // the Jacobian limit is the domain of the elements: its violators are reported by element

		const Vec xf = P_.reduced_to_full(x);
		Anchor &cur = cache.cur();
		const Anchor &prev = cache.prev();
		const bool have_cur = cache.cur_has(xf, true) && cache.cur_has(xf, false) && cur.u_ok;
		const bool have_prev = have_cur && prev.g_ok && prev.h_ok && prev.u_ok && prev.xf.size() == xf.size();
		// the units born exact leave when they are not units of the anchor any more
		if (par.born_hold && have_cur)
			S_.prune_born(cur.coll);

		if (!have_prev)
		{
			int n_new = 0;
			if (l2 && !viol_keys.empty())
				n_new = S_.add_keys(viol_keys, true);
			if (have_cur)
			{
				cur.projected = projected;
				cur.res2 = res2;
			}
			return n_new;
		}
		if (!l2 && !l3)
		{
			// only the units born exact and the first layer: nothing is selected here
			cur.projected = projected;
			cur.res2 = res2;
			return 0;
		}

		const bool prev_projected = prev.projected;
		cur.projected = projected;
		// the budget: eta^2 times the larger of the residuals of the two anchors
		const double budget_res2 = std::isfinite(prev.res2) ? std::max(prev.res2, res2) : res2;
		cur.res2 = res2;
		const double T = par.eta * par.eta * budget_res2;

		// the previous anchor made again with the functions of now if the energy changed since it was taken
		Anchor prev_re;
		bool reeval = false;
		if (prev.gen != S_.generation())
		{
			reeval = term_.blind_reeval(prev, prev_re, xf);
			if (!reeval)
				throw std::runtime_error("UnitSelector: boundary re-evaluation not available");
		}
		if (reeval)
		{
		}

		UnitTable T0, T1;
		cache.build_table(reeval ? prev_re : prev, T0);
		cache.build_table(cur, T1);
		const int n0 = T0.n;
		std::vector<char> isx(n0);
		for (int u = 0; u < n0; ++u)
			isx[u] = S_.in_S(T0.key[u]);
		const auto counterpart = [&](const int u) -> int {
			if (u < T0.n_fixed)
				return u;
			const int j = cur.coll.find(T0.key[u]);
			return j < 0 ? -1 : T1.n_fixed + j;
		};
		// units that are at both anchors only: a unit that left is in no defect sum and no candidate list and leaves S
		std::vector<char> present(n0, 1);
		if (par.present_only)
		{
			for (int u = 0; u < n0; ++u)
				if (counterpart(u) < 0)
				{
					present[u] = 0;
					if (isx[u])
					{
						S_.erase(T0.key[u]);
						isx[u] = 0;
					}
				}
		}

		// the force error of every unit over the step from the previous anchor: r = g(x) - g(x0) - H(x0) p, a gradient at the anchor
		// minus the prediction of its frozen quadratic
		const Vec pf = xf - prev.xf;
		std::vector<std::size_t> roff(n0 + 1, 0);
		for (int u = 0; u < n0; ++u)
			roff[u + 1] = roff[u] + std::size_t(T0.nd[u]);
		std::vector<double> R(roff[n0], 0.0);
		utils::maybe_parallel_for(n0, [&](int start, int end, int /*thread_id*/) {
			for (int u = start; u < end; ++u)
			{
				const int nd = T0.nd[u];
				const int c = counterpart(u);
				Eigen::Matrix<double, NL, 1> pl, g1, g0;
				for (int j = 0; j < nd; ++j)
				{
					const int d = T0.dof[u][j];
					pl[j] = d >= 0 ? pf[d] : 0.0;
					g0[j] = T0.g[u][j] * T0.w[u];
					g1[j] = c >= 0 ? T1.g[c][j] * T1.w[c] : 0.0;
					// the consistent-mass inertia terms of the element
					if (T0.xg[u] != nullptr)
						g0[j] += T0.xg[u][j];
					if (c >= 0 && T1.xg[c] != nullptr)
						g1[j] += T1.xg[c][j];
				}
				Eigen::MatrixXd H = Eigen::Map<const Eigen::MatrixXd>(T0.H[u], nd, nd) * T0.w[u];
				if (prev_projected)
				{
					// the barrier part is projected, the friction part is not
					if (T0.Hf[u] != nullptr)
						H -= Eigen::Map<const Eigen::MatrixXd>(T0.Hf[u], nd, nd);
					H = ipc::project_to_psd(H);
					if (T0.Hf[u] != nullptr)
						H += Eigen::Map<const Eigen::MatrixXd>(T0.Hf[u], nd, nd);
				}
				if (T0.xm[u] != nullptr)
				{
					const auto tri = [](const int i, const int jj) { return i >= jj ? i * (i + 1) / 2 + jj : jj * (jj + 1) / 2 + i; };
					for (int i = 0; i < 4; ++i)
						for (int jj = 0; jj < 4; ++jj)
							for (int k = 0; k < 3; ++k)
								H(3 * i + k, 3 * jj + k) += T0.xw[u] * T0.xm[u][tri(i, jj)];
				}
				const Eigen::VectorXd Hp = H * pl.head(nd);
				for (int j = 0; j < nd; ++j)
				{
					double r = g1[j] - g0[j] - Hp[j];
					if (std::abs(r) <= FLOOR * (std::abs(g1[j]) + std::abs(g0[j]) + std::abs(Hp[j])))
						r = 0.0; // below the rounding of its terms
					R[roff[u] + j] = r;
				}
			}
		});

		const auto scatter = [&](Vec &c, const int u, const double sign) {
			for (int j = 0; j < T0.nd[u]; ++j)
				if (const int d = T0.dof[u][j]; d >= 0)
					c[d] += sign * R[roff[u] + j];
		};
		Vec c_un = l2 ? Vec(Vec::Zero(xf.size())) : Vec(), c_all = l3 ? Vec(Vec::Zero(xf.size())) : Vec();
		for (int u = 0; u < n0; ++u)
		{
			if (!present[u])
				continue;
			if (l3)
				scatter(c_all, u, 1.0);
			if (l2 && !isx[u])
				scatter(c_un, u, 1.0);
		}

		const Vec zero_full = P_.reduced_to_full(Vec::Zero(x.size()));
		const auto back = [&](const Vec &cfull, Vec &ufull, double &q) {
			const Vec cr = P_.uses_full_size() ? cfull : P_.full_to_reduced_grad(cfull);
			Vec ur;
			if (!ctx.solve(cr, ur))
				throw std::runtime_error("UnitSelector: back-solve not available");
			q = cr.dot(ur);
			ufull = P_.reduced_to_full(ur) - zero_full;
		};
		const auto share_of = [&](const int u, const Vec &ufull) {
			double s = 0;
			for (int j = 0; j < T0.nd[u]; ++j)
				if (const int d = T0.dof[u][j]; d >= 0)
					s += R[roff[u] + j] * ufull[d];
			return s;
		};

		struct Adm
		{
			std::vector<int> sel;
			double q0 = 0;
		};
		// Admission of the candidates (units) by their shares of the defect ctot: B^-1 ctot attributes the error q0 = ctot . B^-1 ctot to
		// the units (share r_u . B^-1 ctot); the units with the largest positive shares are taken until the error that is left is
		// below the threshold T (doubling batches, checked with a back-solve each, at most 5), or, with the marginal benefit rule, until
		// a batch lowers the error norm by less than kappa
		const auto admit = [&](const std::vector<int> &cand, const Vec &ctot) -> Adm {
			Adm a;
			Vec u;
			back(ctot, u, a.q0);
			std::vector<std::pair<double, int>> sh;
			sh.reserve(cand.size());
			for (const int e : cand)
				sh.emplace_back(share_of(e, u), e);
			if (!(a.q0 > T))
				return a;
			std::sort(sh.begin(), sh.end(), [](const auto &p, const auto &q) { return p.first != q.first ? p.first > q.first : p.second < q.second; });
			int npos = 0;
			while (npos < int(sh.size()) && sh[npos].first > 0)
				++npos;
			int n = 0;
			double run = a.q0;
			while (n < npos && run > T)
				run -= sh[n++].first;
			int n_ret = n;  // the number of units returned
			int n_prev = 0; // the previous batch (0: nothing admitted)
			double e_prev = std::sqrt(std::max(a.q0, 0.0));
			for (int check = 1; check <= 5 && n > 0; ++check)
			{
				Vec cp = ctot;
				for (int k = 0; k < n; ++k)
					scatter(cp, sh[k].second, -1.0);
				Vec up;
				double q;
				back(cp, up, q);
				n_ret = n;
				if (q <= T)
					break;
				if (marginal)
				{
					const double e_now = std::sqrt(std::max(q, 0.0));
					const double drop = e_prev > 0 ? (e_prev - e_now) / e_prev : 0.0;
					if (!(drop >= par.marginal_kappa))
					{
						n_ret = n_prev;
						break;
					}
					n_prev = n;
					e_prev = e_now;
				}
				if (check == 5)
					break;
				const int n2 = std::min(2 * n, npos);
				if (n2 == n)
					break;
				n = n2;
			}
			if (marginal)
				n = n_ret; // without the marginal rule the returned set is the last one checked
			a.sel.reserve(n);
			for (int k = 0; k < n; ++k)
				a.sel.push_back(sh[k].second);
			return a;
		};

		Adm real;
		if (l2)
		{
			std::vector<int> cand;
			cand.reserve(n0);
			for (int u = 0; u < n0; ++u)
				if (!isx[u] && present[u])
					cand.push_back(u);
			real = admit(cand, c_un);
		}

		// third layer: the exact units that a shadow accounting (all present units as unselected) selects, or that violate a Jacobian
		// limit over the step, are charged in this epoch and stay in S
		std::vector<char> charged(n0, 0);
		Adm shadow;
		if (l3 && S_.size() > 0)
		{
			std::vector<int> cand;
			cand.reserve(n0);
			for (int u = 0; u < n0; ++u)
				if (present[u])
					cand.push_back(u);
			shadow = admit(cand, c_all);
			for (const int u : shadow.sel)
				if (isx[u])
				{
					charged[u] = 1;
				}
			if (const auto fv = term_.fast_validity(); fv)
			{
				const Vec det0 = fv->dets(prev.xf);
				std::vector<double> J, ratio;
				fv->jacobian_arrays(det0, xf, J, ratio);
				const double inf = std::numeric_limits<double>::infinity();
				for (int e = 0; e < cache.n_elements() && e < int(J.size()); ++e)
					if (isx[e] && !charged[e] && ((ctx.jacobian_bound > 0 && ratio[e] < ctx.jacobian_bound) || (ctx.jacobian_floor > -inf && !(J[e] >= ctx.jacobian_floor))))
					{
						charged[e] = 1;
					}
			}
			for (int u = 0; u < n0; ++u)
				if (charged[u])
					S_.charge(T0.key[u]);
		}

		std::vector<UnitKey> admitted;
		if (l2)
		{
			for (const int u : real.sel)
				admitted.push_back(T0.key[u]);
			admitted.insert(admitted.end(), viol_keys.begin(), viol_keys.end());
		}
		const int n_new = l2 ? S_.add_keys(admitted, true) : 0;

		return n_new;
	}
} // namespace polyfem::solver
