#pragma once

#include <polysolve/nonlinear/NonsmoothPorts.hpp>

#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <array>
#include <functional>
#include <unordered_map>
#include <vector>

namespace polyfem::solver
{
	/// @brief A unit of the anchor as the repair sees it: the nodes it depends on, its local dofs (in the unknowns of the problem; -1: fixed)
	/// and the block of the terms that the basis matrix projects (the weighted raw Hessian of the elastic or barrier terms), and whether the
	/// model keeps it frozen (not exact).
	struct RepairUnit
	{
		int n_nodes = 0;
		std::array<int, 4> node{{-1, -1, -1, -1}};
		int nd = 0;
		std::array<int, 12> dof{};
		Eigen::MatrixXd H;
		bool frozen = true;
	};

	/// @brief The basis port of a term: the static correction of the lifted units (first layer, lift_model frozen) and the local repair R_ad of
	/// an indefinite B. R_ad: when the factorization of B fails at an unknown, the frozen units with negative curvature whose nodes are within a
	/// radius (in the graph of nodes that share a unit) of the node of the failure get their projected block, dB = sum (proj(K) - K); a round
	/// that fails at the same node again enlarges the radius by one, a round that fails at another node starts around that node with the
	/// initial radius; at most basis_repair/max_rounds rounds (counted by the strategy). It reads only the blocks and the nodes of the units,
	/// never what energy a unit has.
	class BasisRepair : public polysolve::nonlinear::BasisPort
	{
	public:
		using Vec = Eigen::VectorXd;
		/// @brief The units of the anchor x
		using UnitSource = std::function<void(const Vec &x, std::vector<RepairUnit> &units)>;
		/// @brief The node of an unknown (index in the unknowns of the problem)
		using NodeOfUnknown = std::function<int(const long index)>;
		using Correction = std::function<bool(const Vec &x, const bool projected, StiffnessMatrix &dB)>;

		BasisRepair(UnitSource units, NodeOfUnknown node_of, Correction correction = nullptr)
			: units_(std::move(units)), node_of_(std::move(node_of)), correction_(std::move(correction))
		{
		}

		bool correction(const Vec &x, const bool projected, StiffnessMatrix &dB) override;
		bool repair(const polysolve::nonlinear::RepairQuery &query, StiffnessMatrix &dB, bool &final_round) override;

	private:
		void start(const Vec &x);
		const std::vector<int> &distances(const int node);

		UnitSource units_;
		NodeOfUnknown node_of_;
		Correction correction_;

		std::vector<RepairUnit> all_;
		std::vector<int> negs_;                          ///< indices (in all_) of the frozen units with a negative eigenvalue
		std::unordered_map<int, std::vector<int>> adj_;  ///< node -> neighbouring nodes
		std::unordered_map<int, int> node_index_;
		std::vector<int> dist_;
		std::vector<char> in_;                           ///< per neg: in the repaired set
		int cur_node_ = -1, radius_ = 2;
		std::size_t n_prev_ = 0;
		Vec x_;
	};
} // namespace polyfem::solver
