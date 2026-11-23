#pragma once

// Closed-form neo-Hookean kernel for linear tetrahedra with one quadrature point (an option of ElasticForm, off by default).
//
// Same energy as assembler::NeoHookeanElasticity:
//   W(F) = mu/2 (|F|^2 - 3 - 2 log J) + lambda/2 log^2 J,   F = I + u^T B,   J = det F,
//   E = da * W,  da = det * quadrature weight of the element,  B(i,k) = (reference gradient of shape function i * jac_it)(k)
// and the same local dof order (node i, component m -> 3 i + m) and layout (Hessian column-major 12 x 12).
//
// Closed form (cof = dJ/dF, a_i = cof B_i, c1 = (mu + lambda (1 - log J)) / J^2, c2 = (lambda log J - mu) / J); F, J, cof, |F|^2 and the stress
// P = mu F - mu/J cof + lambda log J / J cof are formed in the order of the assembler's code (the same rounding up to the compiler's contraction):
//   g_i       = da sum_k B(i, k) P(:, k)
//   H_(i,j)   = da (mu (B_i . B_j) I + c1 a_i a_j^T + c2 [eps_(m n r) (F (B_i x B_j))_r]_(m n))      (3 x 3 block, i != j; the last term is 0 for i == j)
// B_i . B_j and B_i x B_j are precomputed per element.

#include <Eigen/Core>

#include <cmath>
#include <vector>

namespace polyfem::solver::fastp1
{
	/// pair index of the local nodes i < j (0..5): (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
	inline int pair_index(const int i, const int j)
	{
		static constexpr int t[4][4] = {{-1, 0, 1, 2}, {-1, -1, 3, 4}, {-1, -1, -1, 5}, {-1, -1, -1, -1}};
		return t[i][j];
	}

	/// per element constants
	struct Elem
	{
		double B[12];  ///< B(i, k) at i * 3 + k
		double BB[16]; ///< B_i . B_j at i * 4 + j
		double W[18];  ///< B_i x B_j (i < j) at 3 * pair_index(i, j) + s
		double da = 0, lambda = 0, mu = 0;
		int dofs[12];  ///< full dof of each local dof
	};

	inline void fill_derived(Elem &el)
	{
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				el.BB[i * 4 + j] = el.B[i * 3] * el.B[j * 3] + el.B[i * 3 + 1] * el.B[j * 3 + 1] + el.B[i * 3 + 2] * el.B[j * 3 + 2];
		for (int i = 0; i < 4; ++i)
			for (int j = i + 1; j < 4; ++j)
			{
				const double *a = el.B + 3 * i, *b = el.B + 3 * j;
				double *w = el.W + 3 * pair_index(i, j);
				w[0] = a[1] * b[2] - a[2] * b[1];
				w[1] = a[2] * b[0] - a[0] * b[2];
				w[2] = a[0] * b[1] - a[1] * b[0];
			}
	}

	template <typename T>
	struct Id
	{
		using type = T;
	};

	/// Energy of the element at the displacement x (full dof vector); the gradient g (12) if G, the Hessian Hm (144, column-major, unprojected) if Hh.
	/// Both include the factor da. Any of them is only written if requested. T = double (long double only for the K3 reference).
	template <bool G, bool Hh, typename T = double>
	inline T eval(const Elem &el, const T *x, typename Id<T>::type *g, typename Id<T>::type *Hm)
	{
		using std::log;
		const double *Bd = el.B;
		T B[12];
		for (int i = 0; i < 12; ++i)
			B[i] = T(Bd[i]);
		T u[4][3];
		for (int i = 0; i < 4; ++i)
			for (int m = 0; m < 3; ++m)
				u[i][m] = x[el.dofs[3 * i + m]];
		// F(m, k) = delta + sum_i u(i, m) B(i, k)
		T F[3][3];
		for (int m = 0; m < 3; ++m)
			for (int k = 0; k < 3; ++k)
				F[m][k] = T(m == k ? 1.0 : 0.0) + u[0][m] * B[k] + u[1][m] * B[3 + k] + u[2][m] * B[6 + k] + u[3][m] * B[9 + k];
		// columns of F: c0, c1, c2; cof = dJ/dF with columns (c1 x c2, c2 x c0, c0 x c1)
		T cof[3][3];
		cof[0][0] = F[1][1] * F[2][2] - F[2][1] * F[1][2];
		cof[1][0] = F[2][1] * F[0][2] - F[0][1] * F[2][2];
		cof[2][0] = F[0][1] * F[1][2] - F[1][1] * F[0][2];
		cof[0][1] = F[1][2] * F[2][0] - F[2][2] * F[1][0];
		cof[1][1] = F[2][2] * F[0][0] - F[0][2] * F[2][0];
		cof[2][1] = F[0][2] * F[1][0] - F[1][2] * F[0][0];
		cof[0][2] = F[1][0] * F[2][1] - F[2][0] * F[1][1];
		cof[1][2] = F[2][0] * F[0][1] - F[0][0] * F[2][1];
		cof[2][2] = F[0][0] * F[1][1] - F[1][0] * F[0][1];
		const T J = F[0][0] * cof[0][0] + F[1][0] * cof[1][0] + F[2][0] * cof[2][0];
		const T lj = log(J);
		const T lam = T(el.lambda), mu = T(el.mu), da = T(el.da);
		// |F|^2 summed in the order Eigen's squaredNorm uses for a 3 x 3 matrix without vectorization (column-major tree)
		T e2[9];
		for (int m = 0; m < 3; ++m)
			for (int k = 0; k < 3; ++k)
				e2[m + 3 * k] = F[m][k] * F[m][k];
		const T sq = ((e2[0] + e2[1]) + (e2[2] + e2[3])) + ((e2[4] + e2[5]) + (e2[6] + (e2[7] + e2[8])));
		const T energy = da * (mu / 2 * (sq - 3 - 2 * lj) + lam / 2 * lj * lj);
		if constexpr (G || Hh)
		{
			const T iJ = T(1) / J;
			T a[4][3];
			for (int i = 0; i < 4; ++i)
				for (int m = 0; m < 3; ++m)
				{
					a[i][m] = cof[m][0] * B[3 * i] + cof[m][1] * B[3 * i + 1] + cof[m][2] * B[3 * i + 2];
				}
			if constexpr (G)
			{
				// PK1 stress P as the assembler writes it: mu F - mu (1/J) cof + lambda log J (1/J) cof; g(i, m) = sum_k B(i, k) P(m, k) times da
				T P[3][3];
				const T c4 = mu * iJ, c3 = (lam * lj) * iJ;
				for (int m = 0; m < 3; ++m)
					for (int k = 0; k < 3; ++k)
						P[m][k] = (mu * F[m][k] - c4 * cof[m][k]) + c3 * cof[m][k];
				for (int i = 0; i < 4; ++i)
					for (int m = 0; m < 3; ++m)
						g[3 * i + m] = (B[3 * i] * P[m][0] + B[3 * i + 1] * P[m][1] + B[3 * i + 2] * P[m][2]) * da;
			}
			if constexpr (Hh)
			{
				const T c2 = (lam * lj - mu) / J;
				const T c1 = (mu + lam * (1 - lj)) / (J * J);
				for (int i = 0; i < 4; ++i)
				{
					// diagonal block (i, i): no antisymmetric part
					{
						const T d = da * mu * T(el.BB[i * 4 + i]);
						for (int m = 0; m < 3; ++m)
							for (int n = 0; n < 3; ++n)
								Hm[(3 * i + m) + 12 * (3 * i + n)] = da * c1 * a[i][m] * a[i][n] + (m == n ? d : T(0));
					}
					for (int j = i + 1; j < 4; ++j)
					{
						const double *wd = el.W + 3 * pair_index(i, j);
						const T w[3] = {T(wd[0]), T(wd[1]), T(wd[2])};
						// v = F w
						const T v0 = F[0][0] * w[0] + F[0][1] * w[1] + F[0][2] * w[2];
						const T v1 = F[1][0] * w[0] + F[1][1] * w[1] + F[1][2] * w[2];
						const T v2 = F[2][0] * w[0] + F[2][1] * w[1] + F[2][2] * w[2];
						const T M[3][3] = {{0, v2, -v1}, {-v2, 0, v0}, {v1, -v0, 0}};
						const T d = da * mu * T(el.BB[i * 4 + j]);
						for (int m = 0; m < 3; ++m)
							for (int n = 0; n < 3; ++n)
							{
								const T val = da * (c1 * a[i][m] * a[j][n] + c2 * M[m][n]) + (m == n ? d : T(0));
								Hm[(3 * i + m) + 12 * (3 * j + n)] = val;
								Hm[(3 * j + n) + 12 * (3 * i + m)] = val;
							}
					}
				}
			}
		}
		return energy;
	}
} // namespace polyfem::solver::fastp1
