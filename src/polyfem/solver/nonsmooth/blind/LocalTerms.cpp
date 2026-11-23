#include "LocalTerms.hpp"

#include <polyfem/solver/forms/Form.hpp>
#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <cmath>

namespace polyfem::solver
{
	UnitKey make_unit_key(const int *nodes, const int n)
	{
		UnitKey k;
		std::array<int32_t, 4> v{{-1, -1, -1, -1}};
		for (int i = 0; i < n && i < 4; ++i)
			v[i] = nodes[i];
		std::sort(v.begin(), v.begin() + std::min(n, 4));
		k.node = v;
		return k;
	}

	namespace
	{
		/// True if every entry of the Hessian lies in the 3 x 3 block of one node
		bool node_block_diagonal(const StiffnessMatrix &H, const int dim)
		{
			for (int c = 0; c < H.outerSize(); ++c)
				for (StiffnessMatrix::InnerIterator it(H, c); it; ++it)
					if (it.value() != 0 && it.row() / dim != it.col() / dim)
						return false;
			return true;
		}
	} // namespace

	BlindForms BlindForms::classify(const std::vector<std::shared_ptr<Form>> &forms, const std::shared_ptr<Form> &elastic,
									const std::shared_ptr<Form> &contact, const std::shared_ptr<Form> &friction, const Eigen::VectorXd &x0, const int dim)
	{
		BlindForms out;
		std::string refused;
		for (const auto &f : forms)
		{
			if (f == elastic || f == contact || f == friction)
				continue;
			if (!f->enabled())
				continue;
			// a node-local form: its Hessian couples only the dofs of one node (checked here at the first point; the
			// lumped inertia has the diagonal mass, the body force no Hessian)
			bool ok = true;
			try
			{
				StiffnessMatrix H;
				const bool flag = f->is_project_to_psd();
				f->set_project_to_psd(false);
				f->second_derivative(x0, H);
				f->set_project_to_psd(flag);
				ok = node_block_diagonal(H, dim);
			}
			catch (const std::exception &)
			{
				ok = false;
			}
			if (ok)
				out.node_local.push_back(f);
			else
				refused += (refused.empty() ? "" : ", ") + f->name();
		}
		if (!refused.empty())
			log_and_throw_error("NonsmoothNewton: selection = blind cannot split the form(s) [{}] into units (blind/unsupported_forms = error). Supported: the elastic form, barrier contact, and node-local forms (lumped mass inertia, body force).", refused);
		return out;
	}

	void fill_node_data(const std::vector<std::shared_ptr<Form>> &node_local, const Eigen::VectorXd &xf, const int dim, NodeData &out)
	{
		const int nn = int(xf.size()) / dim;
		out.key.resize(nn);
		out.dof.assign(std::size_t(nn) * 3, -1);
		out.g.assign(std::size_t(nn) * 3, 0.0);
		out.H.assign(std::size_t(nn) * 9, 0.0);
		for (int i = 0; i < nn; ++i)
		{
			const int nd[1] = {i};
			out.key[i] = make_unit_key(nd, 1);
			for (int d = 0; d < dim; ++d)
				out.dof[std::size_t(i) * 3 + d] = i * dim + d;
		}
		for (const auto &f : node_local)
		{
			Eigen::VectorXd g;
			f->first_derivative(xf, g);
			for (Eigen::Index r = 0; r < g.size(); ++r)
				out.g[std::size_t(r / dim) * 3 + r % dim] += g[r];
			StiffnessMatrix H;
			const bool flag = f->is_project_to_psd();
			f->set_project_to_psd(false);
			f->second_derivative(xf, H);
			f->set_project_to_psd(flag);
			for (int c = 0; c < H.outerSize(); ++c)
				for (StiffnessMatrix::InnerIterator it(H, c); it; ++it)
					out.H[std::size_t(it.col() / dim) * 9 + (it.row() % dim) + 3 * (it.col() % dim)] += it.value();
		}
	}
} // namespace polyfem::solver
