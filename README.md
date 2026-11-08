# mdao-su2

`mdao-su2` connects SU2 flow and discrete-adjoint solves to OpenMDAO and MPhys for aerodynamic shape optimization. You can parameterize geometry, prepare meshes, configure and optimize runs, and inspect recorded results. The package includes seven reference cases and example SU2-native runs.

## Install and activate

The package requires Python 3.14. The runtime stack includes SU2, Open MPI, and Python packages that `setup.sh` installs. A plain `pip install` does not provision that stack. From the checkout root, run the setup once and activate the environment in each new shell:

```bash
bash setup.sh --all
source activate.sh
```

`setup.sh` uses the checkout's parent directory for its installation root. Run `bash setup.sh --help` to choose components or paths. The [human guide](docs/human-guide.pdf) gives the full installation and workflow instructions.

## Inspect and run a case

Each runnable case has a TOML file under `cases/<case>/input/configs/`. Inspect the resolved configuration before starting a solver run:

```bash
CFG=cases/rae2822_rans_sa_hh/input/configs/run_sp.toml
python run/optimize.py --config "$CFG" --print-config
```

To run the task selected by that file, give it a run ID and output location:

```bash
python run/optimize.py --config "$CFG" --run-id baseline --output-root /tmp/mdao-su2-runs
```

The run writes its status, resolved configuration, logs, results, and final artifacts to `/tmp/mdao-su2-runs/baseline/`. `run/optimize.py --help` lists tasks such as model evaluation, derivative checks, and optimization. The other entry points are `run/mesh.py` for geometry and meshes, `run/plot.py` for figures, and `run/archive.py` for run archives.

## Reference cases

The seven case definitions include three Aerodynamic Design Optimisation Discussion Group (ADODG) benchmarks:

- [ADODG Case 1: NACA 0012 airfoil, Euler flow](cases/adodg1_naca0012_euler_ffd/case.toml)
- [ADODG Case 2: RAE 2822 airfoil, RANS–SA flow](cases/adodg2_rae2822_rans_sa_ffd/case.toml)
- [ADODG Case 3: NACA 0012 wing, Euler flow](cases/adodg3_naca0012_euler_ffd/case.toml)
- [RAE 2822 airfoil with Hicks–Henne geometry](cases/rae2822_rans_sa_hh/case.toml)
- [ONERA M6 wing, Euler flow](cases/oneram6_euler_ffd/case.toml)
- [E387 airfoil, RANS–SA flow](cases/e387_rans_sa_bcm/case.toml)
- [NLF 0416 airfoil, RANS–SA flow](cases/nlf0416_rans_sa_bcm/case.toml)

The [ADODG validation study](docs/validation-study.md) reports recorded outcomes and their evidence limits. Historical results do not establish grid-independent or experimentally validated absolute drag for these cases.

## Verify and read more

After `source activate.sh`, run the bounded checks with:

```bash
bash verify.sh --lane fast
```

The solver regression lane starts SU2 work and takes longer. If you need that evidence, run `bash verify.sh 4 --lane solver`.

The [human guide](docs/human-guide.pdf) covers workflows. The [code manual](docs/llm-guides/code_manual.md) maps package behavior, and the [reference manual](docs/llm-guides/reference_manual.md) lists configuration keys and command flags.

## License

This project is licensed under [LGPL-2.1-only](LICENSE).
