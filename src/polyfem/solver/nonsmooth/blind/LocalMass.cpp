#include "LocalMass.hpp"

#include <polyfem/assembler/Mass.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <algorithm>
#include <cmath>
#include <map>

namespace polyfem::solver
{
	std::shared_ptr<LocalMass> LocalMass::build(const assembler::Mass &mass, const bool is_volume, const std::vector<basis::ElementBases> &bases,
												const std::vector<basis::ElementBases> &geom_bases, const assembler::AssemblyValsCache &cache)
	{
		auto out = std::make_shared<LocalMass>();
		const int n = int(bases.size());
		out->n = n;
		out->m.assign(std::size_t(n) * 10, 0.0);
		out->node.assign(std::size_t(n) * 4, -1);
		const int size = mass.size();
		bool ok = true;
		utils::maybe_parallel_for(n, [&](int start, int end, int /*thread_id*/) {
			assembler::ElementAssemblyValues vals;
			for (int e = start; e < end; ++e)
			{
				// the inputs of LinearAssembler::assemble: the cached assembly values of the element, da = det * quadrature weights, time 0
				cache.compute(e, is_volume, bases[e], geom_bases[e], vals);
				const QuadratureVector da = vals.det.array() * vals.quadrature.weights.array();
				if (vals.basis_values.size() != 4 || !is_volume || size != 3)
				{
					ok = false;
					continue;
				}
				for (int i = 0; i < 4; ++i)
				{
					const auto &global = vals.basis_values[i].global;
					if (global.size() != 1 || global[0].val != 1.0)
					{
						ok = false;
						continue;
					}
					out->node[std::size_t(e) * 4 + i] = int(global[0].index);
					for (int j = 0; j <= i; ++j)
						out->m[std::size_t(e) * 10 + tri(i, j)] = mass.assemble(assembler::LinearAssemblerData(vals, 0, i, j, da))(0);
				}
			}
		});
		if (!ok)
			log_and_throw_error("NonsmoothNewton: selection = blind with the consistent mass needs linear tetrahedra with unit dof weights; use solver/advanced/lump_mass_matrix = true or selection = energy_split");
		return out;
	}

	void LocalMass::gradient(const int e, const double w, const Eigen::VectorXd &x, const Eigen::VectorXd &xt, double *g) const
	{
		double d[12];
		for (int i = 0; i < 4; ++i)
			for (int k = 0; k < 3; ++k)
				d[3 * i + k] = x[dof(e, i, k)] - xt[dof(e, i, k)];
		for (int i = 0; i < 4; ++i)
			for (int k = 0; k < 3; ++k)
			{
				double s = 0;
				for (int j = 0; j < 4; ++j)
					s += scalar(e, i, j) * d[3 * j + k];
				g[3 * i + k] = w * s;
			}
	}

	double LocalMass::energy(const int e, const double w, const Eigen::VectorXd &x, const Eigen::VectorXd &xt) const
	{
		double g[12], d[12], s = 0;
		gradient(e, 1.0, x, xt, g);
		for (int i = 0; i < 4; ++i)
			for (int k = 0; k < 3; ++k)
				d[3 * i + k] = x[dof(e, i, k)] - xt[dof(e, i, k)];
		for (int j = 0; j < 12; ++j)
			s += d[j] * g[j];
		return 0.5 * w * s;
	}

	void LocalMass::add_hessian(const int e, const double w, double *H) const
	{
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				for (int k = 0; k < 3; ++k)
					H[(3 * i + k) + 12 * (3 * j + k)] += w * scalar(e, i, j);
	}

	double LocalMass::mismatch(const StiffnessMatrix &M, const int dim) const
	{
		std::map<std::pair<int, int>, double> sum; // (row node, column node) -> scalar entry (the mass is m (x) I)
		for (int e = 0; e < n; ++e)
			for (int i = 0; i < 4; ++i)
				for (int j = 0; j < 4; ++j)
					sum[{node[std::size_t(e) * 4 + i], node[std::size_t(e) * 4 + j]}] += scalar(e, i, j);
		double scale = 0, err = 0;
		for (int c = 0; c < M.outerSize(); ++c)
			for (StiffnessMatrix::InnerIterator it(M, c); it; ++it)
				scale = std::max(scale, std::abs(it.value()));
		// every entry of M against the sum of the blocks; entries of the blocks that are not in M count as errors
		std::map<std::pair<int, int>, double> seen;
		for (int c = 0; c < M.outerSize(); ++c)
			for (StiffnessMatrix::InnerIterator it(M, c); it; ++it)
			{
				const int r = int(it.row()), cc = int(it.col());
				const std::pair<int, int> key{r / dim, cc / dim};
				const double want = r % dim == cc % dim ? sum.count(key) ? sum[key] : 0.0 : 0.0;
				err = std::max(err, std::abs(it.value() - want));
				if (r % dim == cc % dim)
					seen[key] = 1;
			}
		for (const auto &kv : sum)
			if (!seen.count(kv.first))
				err = std::max(err, std::abs(kv.second));
		return scale > 0 ? err / scale : err;
	}
} // namespace polyfem::solver
