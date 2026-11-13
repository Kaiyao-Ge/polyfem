# Benchmark scenes for NonsmoothNewton

Four contact scenes from the NonsmoothNewton benchmark, with their meshes. Mesh paths are relative, so the folder can live anywhere.

| File | Scene | Deformable DOFs | Frames (dt) | Notes |
|---|---|---|---|---|
| `golf_ball.json` | golf ball at 67 m/s hits a fixed steel plate | 86,523 | 200 (2e-5 s) | full length; frames 1–59 have no contact |
| `octocat_bowl.json` | three octocats fall into a bowl | 55,197 | 22 (0.07 s) | first 22 of 100 frames |
| `mat_twist_20k.json` | mat twisted at both ends | 60,000 | 100 (0.04 s) | full length |
| `ball_rollers.json` | ball on four rotating rollers | 21,102 | 150 (0.01 s) | friction μ = 0.5; first 150 of 1000 frames |

DOFs are 3 × the vertex count of the deformable meshes, Dirichlet nodes included. Obstacles (bowl, rollers, plane) carry no DOFs.

## Build

Use the polyfem branch `parallel-nonsmooth` from the fork. It contains the method (polyfem#553) plus the parallel sweeps (polyfem#563), and pins the matching polysolve branch.

```sh
git clone -b parallel-nonsmooth https://github.com/Kaiyao-Ge/polyfem.git
cd polyfem
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target PolyFEM_bin
```

## Run

```sh
mkdir -p out/golf
./build/PolyFEM_bin -j golf_ball.json -o out/golf --max_threads 8
```

Timings and Newton statistics go to `out/golf/sim.json`.

Settings in every scene:

- `solver.nonlinear.solver = "NonsmoothNewton"` runs the method. Set it to `"Newton"` for the baseline; everything else stays the same.
- Linear solver `Eigen::CholmodSupernodalLLT`, as in the benchmark. On a Mac, `Eigen::AccelerateLDLT` also works but gives different timings.
- `common.json` carries the shared defaults: contact on, lumped mass, RobustArmijo line search.
- ParaView output is off (`output.paraview.file_name` is empty). Set it to `"sim.pvd"` to write frames.
- `octocat_bowl` and `ball_rollers` raise `solver.nonlinear.max_iterations` to 10000 so the baseline does not stop early.
