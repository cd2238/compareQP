#!/usr/bin/env python3
"""
build_ipquadprog.py — compile IPQuadProg en bibliothèque partagée.

Produit `libipquadprog.so` à côté de ce script, chargée par qpbench.py via
ctypes. Seule la chaîne de résolution est compilée (quadprosimp et ses
sous-programmes) ; les programmes `main` sont exclus. Les `.F90` passent
par le préprocesseur de gfortran : les blocs `#ifdef DEBUG` disparaissent.

Prérequis : gfortran, LAPACK, BLAS (Ubuntu : `sudo apt install gfortran
liblapack-dev libblas-dev`).

Usage :
    python build_ipquadprog.py                 # clone le dépôt si absent
    python build_ipquadprog.py --repo chemin   # dépôt déjà présent localement
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPO_URL = "https://github.com/cd2238/IPQuadProg"
EXCLUDED = {"main.F90", "main_modchol.F90"}  # programmes complets → exclus
EXPECTED = {"quadprosimp.F90", "initqp.F90", "solvesysqp.F90",
            "calcalpha.F90", "modchol.F90", "modchol2.F90"}


def find_sources(repo: str) -> list:
    src = os.path.join(repo, "src")
    if not os.path.isdir(src):
        raise SystemExit(f"Répertoire src introuvable dans {repo!r} "
                         "(ce n'est pas le dépôt IPQuadProg ?)")
    names = sorted(f for f in os.listdir(src)
                   if f.endswith(".F90") and f not in EXCLUDED)
    missing = EXPECTED - set(names)
    if missing:
        raise SystemExit(f"Fichiers sources manquants dans {src} : {sorted(missing)}")
    return [os.path.join(src, f) for f in names]


def get_repo(cli_repo: str | None, clone_dir: str) -> str:
    if cli_repo:
        return cli_repo
    if os.path.isdir(os.path.join("IPQuadProg", "src")):
        return "IPQuadProg"
    if not shutil.which("git"):
        raise SystemExit("git est requis (ou passez --repo vers une copie locale).")
    print(f"Clonage de {REPO_URL} (shallow)…")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, clone_dir], check=True)
    return clone_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=None,
                    help="chemin vers une copie locale du dépôt IPQuadProg")
    ap.add_argument("--output",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "libipquadprog.so"),
                    help="chemin de la bibliothèque produite")
    args = ap.parse_args()

    if not shutil.which("gfortran"):
        raise SystemExit("gfortran introuvable — installez-le "
                         "(ex. `sudo apt install gfortran`).")

    clone_dir = os.path.join(tempfile.gettempdir(), "IPQuadProg-bench")
    repo = get_repo(args.repo, clone_dir)
    sources = find_sources(repo)
    print("Sources compilées :")
    for s in sources:
        print(f"  - {os.path.basename(s)}")

    cmd = ["gfortran", "-shared", "-fPIC", "-O2",
           *sources, "-llapack", "-lblas", "-o", args.output]
    print("Compilation : " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit("Échec de la compilation gfortran.")
    print(f"OK — bibliothèque écrite : {args.output}")
    print("Vérification rapide : python run_bench.py --check")


if __name__ == "__main__":
    main()