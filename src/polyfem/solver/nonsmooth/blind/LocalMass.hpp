#pragma once

#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

#include <memory>
#include <vector>

namespace polyfem::assembler
{
	class Mass;
}

namespace polyfem::solver
{
	/// @brief The mass matrix of the inertia as element blocks: M_e = m_e (x) I_3 with the 4 x 4 symmetric scalar block m_e of every linear
	/// tetrahedron e. They come from the same call of Mass::assemble with the same inputs as the assembly loop of the global mass matrix
	/// (LinearAssembler::assemble), so each entry is the number that the loop adds to the global matrix.
	struct LocalMass
	{
		int n = 0;                  ///< elements
		std::vector<double> m;      ///< 10 per element: the lower triangle of m_e, row by row (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) (3,3)
		std::vector<int> node;      ///< 4 per element: the FE nodes

		static int tri(const int i, const int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }
		double scalar(const int e, const int i, const int j) const { return m[std::size_t(e) * 10 + tri(i, j)]; }

		/// @brief Builds the blocks. Throws unless every element is a linear tetrahedron whose local basis functions are single FE nodes with
		/// weight 1 (blind selection with the consistent mass supports only those).
		static std::shared_ptr<LocalMass> build(const assembler::Mass &mass, const bool is_volume, const std::vector<basis::ElementBases> &bases,
												const std::vector<basis::ElementBases> &geom_bases, const assembler::AssemblyValsCache &cache);

		/// @brief Local gradient w M_e (x_e - xt_e) (12 values) of element e at the full point x with the inertia target xt
		void gradient(const int e, const double w, const Eigen::VectorXd &x, const Eigen::VectorXd &xt, double *g) const;
		/// @brief Local energy 1/2 w (x_e - xt_e)^T M_e (x_e - xt_e)
		double energy(const int e, const double w, const Eigen::VectorXd &x, const Eigen::VectorXd &xt) const;
		/// @brief Adds the local Hessian w M_e (12 x 12, column-major) to H
		void add_hessian(const int e, const double w, double *H) const;
		/// @brief Full dof of local dof 3 i + k of element e
		int dof(const int e, const int i, const int k) const { return 3 * node[std::size_t(e) * 4 + i] + k; }

		/// @brief max |sum_e L_e^T M_e L_e - M| over the entries of the sparse matrix M, relative to max |M| (the blocks and the global matrix are the same mass)
		double mismatch(const StiffnessMatrix &M, const int dim) const;
	};
} // namespace polyfem::solver
