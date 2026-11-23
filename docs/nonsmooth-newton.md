# NonsmoothNewton in PolyFEM

`solver/nonlinear/solver = "NonsmoothNewton"` uses polysolve's strategy of the same name (see
`polysolve/nonlinear/descent_strategies/nonsmooth/README.md`). PolyFEM gives it the split f = s + n through a `NonsmoothTerm`
(`src/polyfem/solver/NonsmoothTerm.hpp`); `NonsmoothNewton/selection` chooses the term.

## `selection = "energy_split"` (default): the paper configuration

`BarrierContactNonsmoothTerm`: n is the barrier contact form (with the friction form), s is every other form. With
`first_layer/enabled = on` the elastic elements that cause a Jacobian cut become exact (`FirstLayer`, `LiftSet`: kept for the time step;
`lift_model`, `lift_basis` choose how they enter the model and B).

## `selection = "blind"`: units

`BlindUnitTerm` (`src/polyfem/solver/nonsmooth/blind/`) does not split by the kind of energy. Every local energy term depends on some FE
nodes; the terms that depend on the same nodes form one unit (`UnitKey` = the sorted nodes):

| terms | unit |
|---|---|
| elastic energy of a tetrahedron, its share of a consistent mass (`LocalMass`) | the tetrahedron |
| lumped inertia, body force, lagged regularization (forms whose Hessian couples only the dofs of one node), the augmented Lagrangian penalty (full size) | the node |
| barrier term of a collision pair and the friction term of the lagged pair on the same vertices | the pair |

The exact set S (`BlindSession`) is a set of keys. `UnitSelector` chooses S at the start of every outer iteration from the force errors of the
units over the step from the previous anchor (second layer, with the marginal benefit rule), keeps the units that are charged (third layer: the
shadow accounting and Jacobian violations of the step; exit rules at `energy_changed` epochs, `blind/exit_at`), and `BlindUnitTerm` makes new
collision pairs exact (born units, `blind/born_hold`). `UnitCache` keeps the local gradients and Hessians of the units at the current and the
previous anchor (the elastic ones are captured while the smooth part is assembled: `assembler/LocalCapture.hpp`).

Forms that cannot be split into units stop the run at startup (`blind/unsupported_forms = error`) or stay in the frozen part (`remainder`).
The augmented Lagrangian subproblems are solved by the Newton chain unless `blind/al_stage = blind`.

## Platform options (default off)

`solver/advanced/fast_p1_kernel` (closed-form NeoHookean P1 kernel), `solver/advanced/initial_perturbation_seed`,
`solver/nonlinear/line_search/RobustArmijo/gradient_norm_accept`, `NonsmoothNewton/reduced_matrix/mode`, `NonsmoothNewton/model/reduced_algebra` (the reduced model gradient sums only the rows of the nonsmooth gradient; changes rounding) and `NonsmoothNewton/model/row_major_q` (row-major copy of the basis for row gathers; same values).

## Time steps and epochs

`NLProblem::begin_time_step(t)` and `NLProblem::energy_changed(kind)` tell the session when the energy function changes (time step, augmented
Lagrangian update, friction lag, a solve starts): the previous anchor is made again with the new functions and units that were not charged in
the epoch leave S.

Tests: `tests/test_nonsmooth_newton_contact.cpp`.

## Notes for macOS

- Nothing in the method reads platform-specific APIs. The only platform-dependent calls are `mkl_set_num_threads_local` (guarded by `EIGEN_USE_MKL`) and the CHOLMOD failure pivot (guarded by `POLYSOLVE_WITH_CHOLMOD`; without CHOLMOD `basis_repair/mode = r_ad` finds no failure index and falls back to projecting the whole smooth part, and the strategy needs `Eigen::SimplicialLLT`).
- Results are not bitwise equal across machines: MKL (Linux/x86) and Accelerate (Apple silicon), and the fused multiply-add rules of the compilers, round differently. Compare a build with the reference build of the same machine, and chaotic scenes by the distribution over seeds (`solver/advanced/initial_perturbation_seed`), not by single runs.
- Build with `-DPOLYFEM_WITH_MISO=OFF` (the generated sources need much memory) and a modest `ninja -j` (the Ogden/MR sources need about 3.5 GB each).
