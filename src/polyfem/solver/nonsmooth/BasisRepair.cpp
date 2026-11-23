#include "BasisRepair.hpp"

#include <ipc/utils/eigen_ext.hpp>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <queue>

namespace polyfem::solver
{
	bool BasisRepair::correction(const Vec &x, const bool projected, StiffnessMatrix &dB)
	{
		return correction_ ? correction_(x, projected, dB) : false;
	}

	void BasisRepair::start(const Vec &x)
	{
		x_ = x;
		all_.clear();
		units_(x, all_);
		negs_.clear();
		adj_.clear();
		node_index_.clear();
		for (int u = 0; u < int(all_.size()); ++u)
		{
			const RepairUnit &r = all_[u];
			// nodes that share a unit are adjacent
			for (int a = 0; a < r.n_nodes; ++a)
				for (int b = 0; b < r.n_nodes; ++b)
					if (a != b)
						adj_[r.node[a]].push_back(r.node[b]);
			if (r.frozen && r.nd > 0)
			{
				const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(r.H, Eigen::EigenvaluesOnly);
				if (es.info() == Eigen::Success && es.eigenvalues()(0) < 0)
					negs_.push_back(u);
			}
		}
		for (auto &kv : adj_)
		{
			std::sort(kv.second.begin(), kv.second.end());
			kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
		}
		in_.assign(negs_.size(), 0);
		cur_node_ = -1;
		n_prev_ = 0;
	}

	/// distance (number of shared units) of every node from `node`, by breadth first search; -1: not reached
	const std::vector<int> &BasisRepair::distances(const int node)
	{
		node_index_.clear();
		std::vector<int> nodes;
		nodes.reserve(adj_.size());
		for (const auto &kv : adj_)
		{
			node_index_[kv.first] = int(nodes.size());
			nodes.push_back(kv.first);
		}
		dist_.assign(nodes.size(), -1);
		const auto it = node_index_.find(node);
		if (it == node_index_.end())
			return dist_;
		std::queue<int> q;
		dist_[it->second] = 0;
		q.push(node);
		while (!q.empty())
		{
			const int v = q.front();
			q.pop();
			for (const int w : adj_[v])
			{
				const int wi = node_index_[w];
				if (dist_[wi] < 0)
				{
					dist_[wi] = dist_[node_index_[v]] + 1;
					q.push(w);
				}
			}
		}
		return dist_;
	}

	bool BasisRepair::repair(const polysolve::nonlinear::RepairQuery &query, StiffnessMatrix &dB, bool &final_round)
	{
		final_round = false;
		if (query.round == 1 || x_.size() != query.x.size() || x_ != query.x)
			start(query.x);
		if (negs_.empty() || query.fail_index < 0)
			return false;
		const int fail_node = node_of_(query.fail_index);
		if (fail_node < 0)
			return false;
		if (cur_node_ < 0)
		{
			cur_node_ = fail_node;
			radius_ = query.radius;
		}
		else
		{
			// the last round failed again: at the same node the radius grows, at another node the repair starts around it
			radius_ = fail_node == cur_node_ ? radius_ + 1 : query.radius;
			cur_node_ = fail_node;
		}
		const std::vector<int> &dist = distances(cur_node_);
		for (std::size_t k = 0; k < negs_.size(); ++k)
		{
			const RepairUnit &r = all_[negs_[k]];
			int md = 0;
			for (int a = 0; a < r.n_nodes && md >= 0; ++a)
			{
				const auto it = node_index_.find(r.node[a]);
				const int d = it == node_index_.end() ? -1 : dist[it->second];
				md = d < 0 ? -1 : std::max(md, d);
			}
			if (md >= 0 && md <= radius_)
				in_[k] = 1;
		}
		std::size_t n = 0;
		for (const char c : in_)
			n += c != 0;
		if (n == 0 || n == n_prev_)
			return false; // nothing new to project: the factorization would fail the same way
		n_prev_ = n;
		std::vector<Eigen::Triplet<double>> trip;
		for (std::size_t k = 0; k < negs_.size(); ++k)
		{
			if (!in_[k])
				continue;
			const RepairUnit &r = all_[negs_[k]];
			const Eigen::MatrixXd D = Eigen::MatrixXd(ipc::project_to_psd(r.H)) - r.H;
			for (int a = 0; a < r.nd; ++a)
				for (int b = 0; b < r.nd; ++b)
					if (r.dof[a] >= 0 && r.dof[b] >= 0 && D(a, b) != 0)
						trip.emplace_back(r.dof[a], r.dof[b], D(a, b));
		}
		dB.resize(query.x.size(), query.x.size());
		dB.setFromTriplets(trip.begin(), trip.end());
		return true;
	}
} // namespace polyfem::solver
