#!/usr/bin/env python3
"""
Benchmark comparatif de solveurs QP sur des problèmes de taille croissante.

Exemples :
    python run_bench.py --check                       # auto-test des conventions
    python run_bench.py --sizes 10,20,50,100 --seeds 5
    python run_bench.py --sizes 50 --solvers ipquadprog,quadprog,osqp
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from qpbench import SOLVER_CLASSES, SOLVERS_BY_NAME, available_solvers, evaluate, make_random_qp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    sizes: Sequence[int] = (10, 20, 50, 100, 200, 300)  # tailles n des problèmes
    m_ratio: float = 2.0                      # m = ratio * n contraintes
    n_seeds: int = 5                          # instances par taille
    cond: float = 10.0                        # conditionnement visé de G
    solver_names: Sequence[str] = ()           # vide = tous les disponibles
    outdir: str = "results"
    feas_tol: float = 1e-6                    # tolérance de faisabilité
    warmup: bool = True


class BenchmarkRunner:
    """Orchestration : génération des problèmes, appels des solveurs,
    agrégation en DataFrame pandas."""

    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        self.solvers = self._select_solvers()

    def _select_solvers(self):
        wanted = list(self.cfg.solver_names)
        unknown = [s for s in wanted if s not in SOLVERS_BY_NAME]
        if unknown:
            raise SystemExit(f"Solveurs inconnus : {unknown}. "
                             f"Choix possibles : {list(SOLVERS_BY_NAME)}")
        chosen = []
        print("Solveurs :")
        for cls in SOLVER_CLASSES:
            if wanted and cls.name not in wanted:
                continue
            s = cls()
            if s.available():
                chosen.append(s)
                print(f"  [x] {cls.name:18s}")
            else:
                print(f"  [ ] {cls.name:18s} indisponible — {cls.pip_hint}")
        if not chosen:
            raise SystemExit("Aucun solveur disponible.")
        return chosen

    def _problem(self, n: int, seed: int):
        m = max(1, int(round(self.cfg.m_ratio * n)))
        pb = make_random_qp(n, m, seed=1000 * n + seed, cond=self.cfg.cond)
        return pb, f"n{n}m{m}s{seed}"

    def _warmup(self):
        """Un problème minuscule résolu par chacun : import / JIT / cache
        hors du chronométrage."""
        if not self.cfg.warmup:
            return
        pb, _ = self._problem(5, -1)
        for s in self.solvers:
            s.solve(pb)

    def run(self) -> pd.DataFrame:
        self._warmup()
        rows = []
        for n in self.cfg.sizes:
            for seed in range(self.cfg.n_seeds):
                pb, pid = self._problem(n, seed)
                for solver in self.solvers:
                    res = solver.solve(pb)
                    row = dict(problem=pid, n=n, m=pb.m, seed=seed,
                               solver=solver.name, time_s=res.wall_time,
                               iterations=res.iterations, status=res.status)
                    if res.x is not None:
                        q = evaluate(pb, res.x)
                        row.update(objective=q.objective,
                                   primal_violation=q.primal_violation,
                                   dual_residual=q.dual_residual,
                                   complementarity=q.complementarity)
                    else:
                        row.update(objective=np.nan, primal_violation=np.nan,
                                   dual_residual=np.nan, complementarity=np.nan)
                    rows.append(row)
        df = pd.DataFrame(rows)

        # Une solution est "valide" si elle existe et respecte les contraintes.
        df["valid"] = df.objective.notna() & (df.primal_violation <= self.cfg.feas_tol)
        best = (df[df.valid].groupby("problem")["objective"]
                .min().rename("best_obj").reset_index())
        df = df.merge(best, on="problem", how="left")
        df["gap"] = (df.objective - df.best_obj) / np.maximum(1.0, df.best_obj.abs())
        df["success"] = df.valid & (df.status == "ok")
        return df


# ---------------------------------------------------------------------------
# Synthèse et figures
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["solver", "n"])
              .agg(problems=("problem", "nunique"),
                   success=("success", "mean"),
                   time_med=("time_s", "median"),
                   gap_med=("gap", "median"),
                   viol_med=("primal_violation", "median"),
                   iter_med=("iterations", "median"))
              .reset_index())


def make_plots(df: pd.DataFrame, outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non installé — figures ignorées.")
        return

    specs = [("time_s", "median", "temps médian (s)", True),
             ("gap", "median", "gap relatif médian vs meilleur", True),
             ("success", "mean", "taux de succès", False)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, (col, agg, label, log) in zip(axes, specs):
        for solver, g in df.groupby("solver"):
            gg = g.groupby("n")[col].agg(agg).reset_index()
            ax.plot(gg["n"], gg[col], "o-", label=solver)
        ax.set_xlabel("n (variables de contrôle)")
        ax.set_ylabel(label)
        ax.set_title(label)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "benchmark.png")
    fig.savefig(out, dpi=150)
    print(f"Figure : {out}")


# ---------------------------------------------------------------------------
# Auto-test des conventions (à lancer une fois avant tout benchmark)
# ---------------------------------------------------------------------------

def self_test() -> None:
    print("=== Auto-test des conventions (problème n=6, m=12, cond=10) ===")
    pb = make_random_qp(6, 12, seed=42, cond=10.0)
    solvers = available_solvers()
    if not solvers:
        raise SystemExit("Aucun solveur disponible.")
    qp_objs = []    # solveurs dédiés QP : doivent coïncider à ~1e-8
    nlp_objs = []   # solveurs NLP généraux : précision ~1e-6 attendue
    for s in solvers:
        res = s.solve(pb)
        if res.x is None:
            print(f"{s.name:18s} ÉCHEC ({res.status})")
            continue
        q = evaluate(pb, res.x)
        print(f"{s.name:18s} status={res.status!s:14.14s} f={q.objective: .10e} "
              f"viol={q.primal_violation:.1e} dual={q.dual_residual:.1e} "
              f"iters={res.iterations}")
        if res.status == "ok" and q.primal_violation <= 1e-7:
            (qp_objs if s.expect == "qp" else nlp_objs).append((s.name, q.objective))
    if len(qp_objs) >= 2:
        ref = min(f for _, f in qp_objs)
        spread = max(f for _, f in qp_objs) - ref
        tol = 1e-8 * max(1.0, abs(ref))
        verdict = ("OK — conventions cohérentes" if spread <= tol
                   else "ATTENTION — les objectifs divergent entre solveurs QP")
        print(f"Écart max entre solveurs QP dédiés : {spread:.3e}  →  {verdict}")
    else:
        print("Moins de deux solveurs QP ont réussi — vérifier les installations.")
    ref = min((f for _, f in qp_objs), default=float("nan"))
    for name, f in nlp_objs:
        print(f"{name:18s} écart vs référence QP : {abs(f - ref):.1e} "
              f"(~1e-6 est normal pour un solveur NLP général)")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="10,20,50,100,200,300,500",
                    help="tailles n, séparées par des virgules (déf. 10,20,50,100)")
    ap.add_argument("--m-ratio", type=float, default=2.0,
                    help="nombre de contraintes m = ratio * n (déf. 2)")
    ap.add_argument("--seeds", type=int, default=5,
                    help="instances aléatoires par taille (déf. 5)")
    ap.add_argument("--cond", type=float, default=10.0,
                    help="conditionnement visé de G (déf. 10)")
    ap.add_argument("--solvers", default="",
                    help="sous-ensemble de solveurs, séparés par des virgules")
    ap.add_argument("--out", default="results", help="dossier de sortie")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="auto-test de conventions puis sortie")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        self_test()
        return

    cfg = BenchConfig(
        sizes=[int(s) for s in args.sizes.split(",") if s.strip()],
        m_ratio=args.m_ratio,
        n_seeds=args.seeds,
        cond=args.cond,
        solver_names=[s.strip() for s in args.solvers.split(",") if s.strip()],
        outdir=args.out,
        warmup=not args.no_warmup,
    )
    runner = BenchmarkRunner(cfg)
    df = runner.run()

    os.makedirs(cfg.outdir, exist_ok=True)
    df.to_csv(os.path.join(cfg.outdir, "results.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(cfg.outdir, "summary.csv"), index=False)

    print("\n=== Synthèse (par solveur et taille) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if not args.no_plots:
        make_plots(df, cfg.outdir)
    print(f"\nDétail complet : {os.path.join(cfg.outdir, 'results.csv')}")


if __name__ == "__main__":
    main()
