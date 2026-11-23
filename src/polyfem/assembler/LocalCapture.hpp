#pragma once

// The elastic assembly loops (NLAssembler::assemble_gradient / assemble_hessian) can store the local (per element) gradient and
// unprojected Hessian they compute anyway (the unit cache of the blind selection). The pointer is null unless a unit cache is
// capturing; a null pointer leaves the loops bitwise unchanged. P1 tetrahedra only (12 local dofs): any other element size counts
// into `bad` and the capture is dropped.

#include <atomic>

namespace polyfem::assembler
{
	struct LocalCapture
	{
		static constexpr int NLOC = 12;
		int n = 0;                 ///< number of elements
		double *g = nullptr;       ///< n * 12 (unweighted element gradients); null: gradient capture off
		double *H = nullptr;       ///< n * 144 (unweighted, unprojected element Hessians, column-major 12 x 12); null: off
		std::atomic<int> bad{0};   ///< elements that did not fit
	};

	/// @brief The capture target of the current assembly (null: none). Set only by the thread that calls the assembly.
	inline LocalCapture *&local_capture()
	{
		static LocalCapture *p = nullptr;
		return p;
	}
} // namespace polyfem::assembler
