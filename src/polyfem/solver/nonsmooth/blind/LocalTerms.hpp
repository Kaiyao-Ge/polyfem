#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace polyfem::solver
{
	class Form;

	/// @brief Identity of a unit across points: the sorted FE nodes that its local energy terms depend on. All local terms that depend
	/// on the same nodes (whatever energy they come from) form one unit. A tetrahedron has 4 nodes, a node unit 1, a collision pair
	/// the nodes of its (up to 4) vertices.
	struct UnitKey
	{
		std::array<int32_t, 4> node{{-1, -1, -1, -1}}; ///< ascending, -1 padded
		bool operator==(const UnitKey &o) const { return node == o.node; }
		int size() const
		{
			int n = 0;
			while (n < 4 && node[n] >= 0)
				++n;
			return n;
		}
	};

	struct UnitKeyHash
	{
		size_t operator()(const UnitKey &k) const
		{
			uint64_t h = 1469598103934665603ull;
			for (const int32_t x : k.node)
			{
				h ^= uint64_t(uint32_t(x));
				h *= 1099511628211ull;
			}
			return size_t(h);
		}
	};

	/// @brief The key of the unit made of the given nodes (any order; duplicates are kept)
	UnitKey make_unit_key(const int *nodes, const int n);

	/// @brief The contact terms at one point, in full dofs: weighted local gradient and raw Hessian. One entry per unit of the pairs: the
	/// barrier term of a collision pair, together with the friction term of the lagged pair on the same vertices (the same stencil, so the
	/// same local dofs), if there is one; friction pairs without a collision pair are entries of their own.
	struct CollData
	{
		std::vector<UnitKey> key;
		std::vector<signed char> nd;
		std::vector<int> dof;     ///< 12 per collision (-1 pads)
		std::vector<double> g;    ///< 12 per collision
		std::vector<double> H;    ///< 144 per collision, nd x nd column-major
		std::vector<double> e;    ///< energy
		std::vector<double> Hf;   ///< 144 per entry: the friction part of H (not projected where the barrier part is); empty without friction
		std::unordered_map<UnitKey, int, UnitKeyHash> index; ///< first entry of a key
		int size() const { return int(key.size()); }
		void clear()
		{
			key.clear();
			nd.clear();
			dof.clear();
			g.clear();
			H.clear();
			e.clear();
			Hf.clear();
			index.clear();
		}
		int find(const UnitKey &k) const
		{
			const auto it = index.find(k);
			return it == index.end() ? -1 : it->second;
		}
	};

	/// @brief The node-local terms (lumped inertia, body force) at one point: per node the local gradient (3) and Hessian (3 x 3)
	struct NodeData
	{
		std::vector<UnitKey> key; ///< one node per unit
		std::vector<int> dof;     ///< 3 per node
		std::vector<double> g;    ///< 3 per node
		std::vector<double> H;    ///< 9 per node, column-major
		int size() const { return int(key.size()); }
	};

	/// @brief True if every entry of the matrix lies in the dim x dim block of one node
	bool is_node_block_diagonal(const StiffnessMatrix &H, const int dim);

	/// @brief The forms of the problem sorted by what the blind selection can do with them. A form is a source of local terms if the
	/// units can be read off its derivatives: the elastic form (elements), the barrier contact form (collision pairs), and forms
	/// whose Hessian is block diagonal by node (the lumped inertia, the body force: one unit per node). Anything else cannot be
	/// split into units and is refused at startup (blind/unsupported_forms = error).
	struct BlindForms
	{
		std::vector<std::shared_ptr<Form>> node_local; ///< node-local forms
		/// @brief Sorts the forms (those that are enabled) given the full point x0 of the problem. Throws with the names of the
		/// forms that cannot be split.
		/// @param elastic, contact, friction the forms that the term handles itself (null: none)
		/// @param skip forms that the caller handles itself (the consistent-mass inertia)
		/// @param remainder true: forms that cannot be split are not an error: they stay in the frozen part (their gradient and Hessian at the anchor) and are not units
		static BlindForms classify(const std::vector<std::shared_ptr<Form>> &forms, const std::shared_ptr<Form> &elastic,
								   const std::shared_ptr<Form> &contact, const std::shared_ptr<Form> &friction, const Eigen::VectorXd &x0, const int dim,
								   const std::vector<std::shared_ptr<Form>> &skip = {}, const bool remainder = false);
	};

	/// @brief The node units at the full point xf: derivatives of the node-local forms (weighted, raw), per node
	void fill_node_data(const std::vector<std::shared_ptr<Form>> &node_local, const Eigen::VectorXd &xf, const int dim, NodeData &out);
} // namespace polyfem::solver
