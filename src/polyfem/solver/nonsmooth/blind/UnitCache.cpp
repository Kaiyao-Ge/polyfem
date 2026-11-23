#include "UnitCache.hpp"

#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	namespace
	{
		constexpr int NL = 12;
	}

	void UnitCache::bind(const std::shared_ptr<ElasticForm> &elastic, const int dim, const Vec &xfull0)
	{
		const int nel = elastic->n_elements();
		if (nel_ == nel && int(dofs_.size()) == nel * NL)
			return;
		nel_ = nel;
		off_ = false;
		dofs_.assign(std::size_t(nel) * NL, -1);
		ekey_.assign(nel, UnitKey());
		const Eigen::MatrixXd xm = xfull0;
		assembler::ElementAssemblyValues vals;
		ElasticForm::ElementLocal loc;
		for (int e = 0; e < nel && !off_; ++e)
		{
			elastic->element_local(xm, e, false, false, vals, loc);
			if (int(loc.dofs.size()) != NL)
			{
				off_ = true;
				break;
			}
			int nodes[4];
			for (int j = 0; j < NL; ++j)
			{
				if (loc.w[j] != 1.0 || loc.dofs[j] < 0)
				{
					off_ = true;
					break;
				}
				dofs_[std::size_t(e) * NL + j] = loc.dofs[j];
				if (j % dim == 0)
					nodes[j / dim] = loc.dofs[j] / dim;
			}
			if (!off_)
				ekey_[e] = make_unit_key(nodes, 4);
		}
		if (off_)
			logger().warn("[NonsmoothNewton] the unit cache is off: the elastic elements are not P1 tetrahedra with unit dof weights");
		for (auto &a : a_)
			a = Anchor();
		cur_ = 0;
	}

	assembler::LocalCapture *UnitCache::capture_begin(const Vec &xf, const double f, const bool grad)
	{
		if (!on())
			return nullptr;
		Anchor *c = &a_[cur_];
		const bool same = c->xf.size() == xf.size() && (c->g_ok || c->h_ok) && c->xf == xf;
		if (!same)
		{
			if (c->g_ok && c->h_ok)
				cur_ = 1 - cur_;
			c = &a_[cur_];
			c->xf = xf;
			c->g_ok = c->h_ok = false;
			c->projected = false;
			c->res2 = std::numeric_limits<double>::quiet_NaN();
		}
		c->f = f;
		if (c->g.size() != std::size_t(nel_) * NL)
			c->g.assign(std::size_t(nel_) * NL, 0.0);
		if (c->H.size() != std::size_t(nel_) * NL * NL)
			c->H.assign(std::size_t(nel_) * NL * NL, 0.0);
		cap_.n = nel_;
		cap_.bad = 0;
		cap_.g = grad ? c->g.data() : nullptr;
		cap_.H = grad ? nullptr : c->H.data();
		if (grad)
			c->g_ok = false;
		else
			c->h_ok = false;
		return &cap_;
	}

	void UnitCache::capture_end(const bool grad)
	{
		Anchor &c = a_[cur_];
		if (cap_.bad.load() != 0)
		{
			off_ = true;
			logger().warn("[NonsmoothNewton] the unit cache is off: an element did not fit");
			return;
		}
		(grad ? c.g_ok : c.h_ok) = true;
	}

	bool UnitCache::cur_has(const Vec &xf, const bool grad) const
	{
		const Anchor &c = a_[cur_];
		return on() && (grad ? c.g_ok : c.h_ok) && c.xf.size() == xf.size() && c.xf == xf;
	}

	UnitCache::Anchor &UnitCache::anchor_for(const Vec &xf, const double f, bool *new_anchor)
	{
		Anchor *c = &a_[cur_];
		const bool same = c->xf.size() == xf.size() && (c->g_ok || c->h_ok || c->u_ok) && c->xf == xf;
		if (!same)
		{
			if (c->g_ok && c->h_ok)
				cur_ = 1 - cur_;
			c = &a_[cur_];
			c->xf = xf;
			c->g_ok = c->h_ok = false;
			c->projected = false;
			c->res2 = std::numeric_limits<double>::quiet_NaN();
			c->u_ok = false;
		}
		if (new_anchor)
			*new_anchor = !same;
		c->f = f;
		return *c;
	}

	void UnitCache::build_table(const Anchor &a, UnitTable &t) const
	{
		const int nel = nel_, nno = a.u_ok ? a.node.size() : 0, nc = a.u_ok ? a.coll.size() : 0;
		t.n = nel + nno + nc;
		t.n_fixed = nel + nno;
		t.key.resize(t.n);
		t.nd.resize(t.n);
		t.dof.resize(t.n);
		t.g.resize(t.n);
		t.H.resize(t.n);
		t.w.resize(t.n);
		const bool has_mass = mass_ != nullptr && a.ig.size() == std::size_t(nel) * NL;
		t.xg.assign(t.n, nullptr);
		t.xm.assign(t.n, nullptr);
		t.xw.assign(t.n, 0.0);
		t.Hf.assign(t.n, nullptr);
		for (int e = 0; e < nel; ++e)
		{
			t.key[e] = ekey_[e];
			t.nd[e] = NL;
			t.dof[e] = dofs_.data() + std::size_t(e) * NL;
			t.g[e] = a.g.data() + std::size_t(e) * NL;
			t.H[e] = a.H.data() + std::size_t(e) * NL * NL;
			t.w[e] = a.f;
			if (has_mass)
			{
				t.xg[e] = a.ig.data() + std::size_t(e) * NL;
				t.xm[e] = mass_->m.data() + std::size_t(e) * 10;
				t.xw[e] = a.wm;
			}
		}
		for (int k = 0; k < nno; ++k)
		{
			const int u = nel + k;
			t.key[u] = a.node.key[k];
			t.nd[u] = 3;
			t.dof[u] = a.node.dof.data() + std::size_t(k) * 3;
			t.g[u] = a.node.g.data() + std::size_t(k) * 3;
			t.H[u] = a.node.H.data() + std::size_t(k) * 9;
			t.w[u] = 1.0;
		}
		for (int j = 0; j < nc; ++j)
		{
			const int u = nel + nno + j;
			t.key[u] = a.coll.key[j];
			t.nd[u] = a.coll.nd[j];
			t.dof[u] = a.coll.dof.data() + std::size_t(j) * NL;
			t.g[u] = a.coll.g.data() + std::size_t(j) * NL;
			t.H[u] = a.coll.H.data() + std::size_t(j) * NL * NL;
			t.w[u] = 1.0;
			if (a.coll.Hf.size() == a.coll.H.size())
				t.Hf[u] = a.coll.Hf.data() + std::size_t(j) * NL * NL;
		}
	}
} // namespace polyfem::solver
