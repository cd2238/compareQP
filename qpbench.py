"""
qpbench - mini-framework objet pour comparer des solveurs de programmation
quadratique convexe (QP) sur des problèmes de taille croissante.

Formulation canonique (identique à celle d'IPQuadProg, cd2238/IPQuadProg) :

    min  1/2 x'Gx + c'x
    s.c.  A x >= b            avec G symétrique définie positive.

Chaque bibliothèque a ses propres conventions d'appel (sens des inégalités,
signe du terme linéaire, position des égalités...) : chaque sous-classe de
QPSolver est responsable de la traduction depuis/vers la formulation
canonique, et les résultats sont ramenés à un dénominateur commun.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Problème
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QPProblem:
    """Un QP convexe :  min 1/2 x'Gx + c'x   s.c.   A x >= b."""

    G: np.ndarray          # (n, n) symétrique définie positive
    c: np.ndarray          # (n,)
    A: np.ndarray          # (m, n)
    b: np.ndarray          # (m,)

    @property
    def n(self) -> int:
        return self.G.shape[0]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    def objective(self, x: np.ndarray) -> float:
        return 0.5 * float(x @ self.G @ x) + float(self.c @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.G @ x + self.c

    def slack(self, x: np.ndarray) -> np.ndarray:
        """Ax - b : strictement positif si la contrainte est satisfaite."""
        return self.A @ x - self.b

    def primal_violation(self, x: np.ndarray) -> float:
        return float(max(0.0, np.max(self.b - self.A @ x)))


def make_random_qp(n: int, m: int, seed: int = 0, cond: float = 10.0,
                   tight_frac: float = 0.3) -> QPProblem:
    """Génère un QP convexe faisable et borné, reproductible par `seed`.

    - G est construite avec un spectre imposé : valeurs propres en échelle
      log entre 1 et `cond` (contrôle du conditionnement).
    - b est construit à partir d'un point faisable x_ref : b = A x_ref - s,
      avec s >= 0 ; une fraction `tight_frac` des marges s est prise très
      petite pour que plusieurs contraintes soient (quasi) actives autour
      de la zone optimale.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigvals = np.logspace(0.0, np.log10(cond), n)
    G = (Q * eigvals) @ Q.T
    G = 0.5 * (G + G.T)
    c = rng.standard_normal(n)
    A = rng.standard_normal((m, n))
    x_ref = rng.standard_normal(n)
    s = rng.uniform(0.2, 2.0, size=m)
    n_tight = int(round(tight_frac * m))
    if n_tight > 0:
        s[:n_tight] = 10.0 ** rng.uniform(-6.0, -2.0, size=n_tight)
    b = A @ x_ref - s
    return QPProblem(G, c, A, b)


# ---------------------------------------------------------------------------
# Résultats et métriques
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Sortie normalisée d'un solveur."""
    x: Optional[np.ndarray]                # None si échec
    status: str = "ok"
    iterations: Optional[int] = None
    lam: Optional[np.ndarray] = None       # duaux (si fournis par le solveur)
    wall_time: float = float("nan")        # rempli par QPSolver.solve
    info: Optional[dict] = None            # champs bruts spécifiques


@dataclass
class QualityMetrics:
    """Métriques KKT recalculées en Python, indépendamment du solveur."""
    objective: float
    primal_violation: float   # max(0, max(b - Ax))
    dual_residual: float      # ||Gx + c - A'lambda||_inf (lambda estimé)
    complementarity: float    # max |lambda_i * (Ax - b)_i|


def evaluate(pb: QPProblem, x: np.ndarray) -> QualityMetrics:
    """Recalcule objectif + métriques KKT pour une solution candidate.

    Les duaux sont estimés par moindres carrés sur l'ensemble des contraintes
    jugées actives, de façon uniforme pour tous les solveurs (certains ne
    retournent pas de duaux).
    """
    x = np.asarray(x, dtype=float)
    slack = pb.slack(x)
    grad = pb.gradient(x)
    viol = float(max(0.0, np.max(pb.b - pb.A @ x)))
    # Tolérance d'activité volontairement un peu large (1e-6 relatif) : les
    # solveurs imprécis s'arrêtent à ~1e-6 de la frontière, et il faut
    # capturer ces contraintes quasi actives pour estimer les duaux.
    tol = 1e-6 * max(1.0, float(np.max(np.abs(pb.b))))
    active = slack <= tol
    lam = np.zeros(pb.m)
    if np.any(active):
        lam[active], *_ = np.linalg.lstsq(pb.A[active].T, grad, rcond=None)
    dual_res = float(np.max(np.abs(grad - pb.A.T @ lam)))
    comp = float(np.max(np.abs(lam * slack))) if pb.m else 0.0
    return QualityMetrics(objective=pb.objective(x), primal_violation=viol,
                          dual_residual=dual_res, complementarity=comp)


# ---------------------------------------------------------------------------
# Interface commune des solveurs
# ---------------------------------------------------------------------------

class QPSolver(ABC):
    """Interface objet commune à tous les solveurs comparés.

    Contrat : `available()` dit si la bibliothèque est installée ; `_solve`
    traduit le problème canonique vers les conventions de la bibliothèque
    et retourne un SolverResult. `solve` chronomètre et capture les erreurs.
    """

    name: str = "?"
    pip_hint: str = ""
    # "qp" : solveur dédié aux QP (précision attendue ~1e-9 au check) ;
    # "nlp" : solveur d'optimisation générale (~1e-6 attendu).
    expect: str = "qp"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def _solve(self, pb: QPProblem) -> SolverResult: ...

    def solve(self, pb: QPProblem) -> SolverResult:
        if not self.available():
            raise RuntimeError(f"{self.name} indisponible ({self.pip_hint})")
        t0 = time.perf_counter()
        try:
            res = self._solve(pb)
        except Exception as exc:  # noqa: BLE001 - on capture tout pour le bench
            res = SolverResult(x=None, status=f"error: {exc.__class__.__name__}: {exc}")
        res.wall_time = time.perf_counter() - t0
        return res


@lru_cache(maxsize=None)
def _import(module: str):
    return importlib.import_module(module)


def _try_import(module: str) -> bool:
    try:
        _import(module)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 1. IPQuadProg (Fortran, points intérieurs) — appelé via ctypes
# ---------------------------------------------------------------------------

class IPQuadProgSolver(QPSolver):
    """IPQuadProg (https://github.com/cd2238/IPQuadProg) : points intérieurs
    primal-dual avec predictor-corrector de Mehrotra et Cholesky modifiée
    (Gill-Murray-Wright). Compilé en bibliothèque partagée par
    build_ipquadprog.py, appelée ici via ctypes. Conventions identiques à
    la formulation canonique (Ax >= b).

    Nommage gfortran : quadprosimp_ (Linux / macOS).
    """

    name = "ipquadprog"
    pip_hint = "compiler avec python build_ipquadprog.py (gfortran + lapack + blas)"

    _CANDIDATE_LIBS = ("libipquadprog.so", "libipquadprog.dylib")

    def __init__(self, itermax: int = 1000, mutol: float = 1e-10):
        self.itermax = itermax
        self.mutol = mutol
        self._fn = None

    def _lib_path(self) -> Optional[str]:
        env = os.environ.get("IPQUADPROG_LIB")
        if env and os.path.exists(env):
            return env
        here = os.path.dirname(os.path.abspath(__file__))
        for name in self._CANDIDATE_LIBS:
            p = os.path.join(here, name)
            if os.path.exists(p):
                return p
        return None

    def available(self) -> bool:
        return self._lib_path() is not None

    def _entry(self):
        if self._fn is None:
            path = self._lib_path()
            if path is None:
                raise RuntimeError("libipquadprog introuvable (lancer build_ipquadprog.py)")
            lib = ctypes.CDLL(path)
            fn = getattr(lib, "quadprosimp_")
            d = ctypes.POINTER(ctypes.c_double)
            i = ctypes.POINTER(ctypes.c_int)
            # quadprosimp(n, G, c, m, A, b, isx0, x0, itermax, mutol,
            #             x, y, lambda, fobj, iter, info)
            fn.argtypes = [i, d, d, i, d, d, i, d, i, d, d, d, d, d, i, i]
            fn.restype = None
            self._fn = fn
        return self._fn

    @staticmethod
    def _dp(a: np.ndarray):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def _solve(self, pb: QPProblem) -> SolverResult:
        fn = self._entry()
        n = ctypes.c_int(pb.n)
        m = ctypes.c_int(pb.m)
        G = np.asfortranarray(pb.G, dtype=np.float64)
        c = np.ascontiguousarray(pb.c, dtype=np.float64)
        A = np.asfortranarray(pb.A, dtype=np.float64)
        b = np.ascontiguousarray(pb.b, dtype=np.float64)
        isx0 = ctypes.c_int(0)
        x0 = np.zeros(pb.n)
        itermax = ctypes.c_int(self.itermax)
        mutol = ctypes.c_double(self.mutol)
        x = np.zeros(pb.n)
        y = np.zeros(pb.m)
        lam = np.zeros(pb.m)
        fobj = ctypes.c_double()
        itc = ctypes.c_int()
        info = ctypes.c_int()
        fn(ctypes.byref(n), self._dp(G), self._dp(c),
           ctypes.byref(m), self._dp(A), self._dp(b),
           ctypes.byref(isx0), self._dp(x0),
           ctypes.byref(itermax), ctypes.byref(mutol),
           self._dp(x), self._dp(y), self._dp(lam),
           ctypes.byref(fobj), ctypes.byref(itc), ctypes.byref(info))
        status = "ok" if info.value == 0 else f"info={info.value}"
        return SolverResult(x=x, status=status, iterations=itc.value, lam=lam,
                            info={"fobj_fortran": fobj.value})


# ---------------------------------------------------------------------------
# 2. quadprog (Goldfarb & Idnani, dual active set)
# ---------------------------------------------------------------------------

class QuadprogSolver(QPSolver):
    """Paquet `quadprog` : méthode duale d'ensemble actif de Goldfarb &
    Idnani (1983). Résout : min 1/2 x'Gx - a'x  s.c.  C'x >= b, d'où
    a = -c et C = A'.

    solve_qp retourne (x, f, a, iters, lagrangian, iact) : `iters` est un
    tableau à 2 éléments (itérations principales, contraintes désactivées)
    et `lagrangian` contient les duaux des contraintes.
    """

    name = "quadprog"
    pip_hint = "pip install quadprog"

    def available(self) -> bool:
        return _try_import("quadprog")

    def _solve(self, pb: QPProblem) -> SolverResult:
        quadprog = _import("quadprog")
        sol = quadprog.solve_qp(np.ascontiguousarray(pb.G), -pb.c,
                                np.ascontiguousarray(pb.A.T), pb.b, 0)
        x = np.asarray(sol[0], dtype=float)
        iterations = None
        if len(sol) > 3 and sol[3] is not None:
            try:
                iterations = int(np.asarray(sol[3])[0])
            except (TypeError, ValueError, IndexError):
                iterations = None
        lam = np.asarray(sol[4], dtype=float) if len(sol) > 4 else None
        return SolverResult(x=x, iterations=iterations, lam=lam)


# ---------------------------------------------------------------------------
# 3. CVXOPT (points intérieurs)
# ---------------------------------------------------------------------------

class CvxoptSolver(QPSolver):
    """CVXOPT : points intérieurs primal-dual. Résout min 1/2 x'Px + q'x
    s.c. Gx <= h, d'où G = -A et h = -b."""

    name = "cvxopt"
    pip_hint = "pip install cvxopt"

    def available(self) -> bool:
        return _try_import("cvxopt")

    def _solve(self, pb: QPProblem) -> SolverResult:
        cvxopt = _import("cvxopt")
        cvxopt.solvers.options["show_progress"] = False
        from cvxopt import matrix

        res = cvxopt.solvers.qp(matrix(np.ascontiguousarray(pb.G)),
                                matrix(pb.c),
                                matrix(np.ascontiguousarray(-pb.A)),
                                matrix(-pb.b))
        status = str(res["status"])
        x = np.asarray(res["x"]).ravel() if "optimal" in status else None
        return SolverResult(x=x, status="ok" if x is not None else status,
                            iterations=res.get("iterations"))


# ---------------------------------------------------------------------------
# 4. OSQP (ADMM, première ordre)
# ---------------------------------------------------------------------------

class OsqpSolver(QPSolver):
    """OSQP : splitting opérateur duale de première ordre (ADMM). Résout
    min 1/2 x'Px + q'x s.c. l <= Mx <= u, d'où l = b et u = +inf.

    Réglages resserrés (eps_abs = eps_rel = 1e-9, polissage actif) : sans
    cela, l'ADMM est artificiellement imprécis face aux méthodes du second
    ordre. La compatibilité des noms d'options selon la version d'OSQP est
    gérée ci-dessous."""

    name = "osqp"
    pip_hint = "pip install osqp"

    def available(self) -> bool:
        return _try_import("osqp")

    @staticmethod
    def _setup(prob, P, q, M, l, u):
        settings = {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 100000}
        for polish_key in ("polishing", "polish"):   # nom selon la version
            try:
                prob.setup(P, q, M, l, u, verbose=False,
                           **settings, **{polish_key: True})
                return
            except TypeError:
                continue
        prob.setup(P, q, M, l, u, verbose=False)

    def _solve(self, pb: QPProblem) -> SolverResult:
        osqp = _import("osqp")
        from scipy import sparse

        P = sparse.csc_matrix(pb.G)
        M = sparse.csc_matrix(pb.A)
        l = pb.b
        u = np.full(pb.m, np.inf)
        prob = osqp.OSQP()
        self._setup(prob, P, pb.c, M, l, u)
        res = prob.solve()

        status = str(getattr(res, "status", None) or res.info.status)
        niter = getattr(res, "iter", None)
        if niter is None:
            niter = getattr(res.info, "iter", None)
        x = np.asarray(res.x) if res.x is not None else None
        solved = "solved" in status.lower()
        return SolverResult(x=x, status="ok" if solved else status,
                            iterations=int(niter) if niter is not None else None)


# ---------------------------------------------------------------------------
# 5. Clarabel (points intérieurs coniques)
# ---------------------------------------------------------------------------

class ClarabelSolver(QPSolver):
    """Clarabel : points intérieurs coniques (Rust). Résout
    min 1/2 x'Px + q'x s.c. Mx + s = b, s dans le cône non négatif ;
    avec M = -A et b -> -b, cela revient à A x >= b."""

    name = "clarabel"
    pip_hint = "pip install clarabel"

    def available(self) -> bool:
        return _try_import("clarabel")

    def _solve(self, pb: QPProblem) -> SolverResult:
        clarabel = _import("clarabel")
        from scipy import sparse

        P = sparse.csc_matrix(pb.G)
        M = sparse.csc_matrix(-pb.A)
        # API actuelle : NonnegativeConeT ; anciennes versions : NonnegativeCone
        cone_cls = (getattr(clarabel, "NonnegativeConeT", None)
                    or getattr(clarabel, "NonnegativeCone"))
        cones = [cone_cls(pb.m)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, pb.c, M, -pb.b, cones, settings)
        sol = solver.solve()
        status = str(sol.status).split(".")[-1]
        solved = "solved" in status.lower()
        x = np.asarray(sol.x) if solved else None
        return SolverResult(x=x, status="ok" if solved else status,
                            iterations=int(sol.iterations))


# ---------------------------------------------------------------------------
# 6-7. SciPy (SLSQP et trust-constr)
# ---------------------------------------------------------------------------

class ScipySlsqpSolver(QPSolver):
    """SciPy `SLSQP` : SQP (approximations quadratiques séquentielles).
    Contraintes 'ineq' de la forme fun(x) >= 0, d'où fun(x) = A x - b."""

    name = "slsqp"
    pip_hint = "pip install scipy"
    expect = "nlp"

    def available(self) -> bool:
        return _try_import("scipy.optimize")

    def _solve(self, pb: QPProblem) -> SolverResult:
        opt = _import("scipy.optimize")
        cons = [{"type": "ineq",
                 "fun": lambda x: pb.A @ x - pb.b,
                 "jac": lambda x: pb.A}]
        res = opt.minimize(pb.objective, np.zeros(pb.n),
                           jac=pb.gradient, constraints=cons, method="SLSQP",
                           options={"ftol": 1e-12, "maxiter": 2000})
        return SolverResult(x=res.x,
                            status="ok" if res.success else f"exit({res.status})",
                            iterations=int(res.nit))


class ScipyTrustConstrSolver(QPSolver):
    """SciPy `trust-constr` : régions de confiance (sous-problèmes QP).
    LinearConstraint(A, lb = b, ub = +inf)."""

    name = "trustconstr"
    pip_hint = "pip install scipy"
    expect = "nlp"

    def available(self) -> bool:
        return _try_import("scipy.optimize")

    def _solve(self, pb: QPProblem) -> SolverResult:
        opt = _import("scipy.optimize")
        lc = opt.LinearConstraint(pb.A, lb=pb.b, ub=np.full(pb.m, np.inf))
        res = opt.minimize(pb.objective, np.zeros(pb.n), jac=pb.gradient,
                           hess=lambda x: pb.G, constraints=[lc],
                           method="trust-constr",
                           options={"gtol": 1e-10, "xtol": 1e-14,
                                    "barrier_tol": 1e-12, "maxiter": 5000})
        return SolverResult(x=res.x,
                            status="ok" if res.success else f"exit({res.status})",
                            iterations=int(res.niter))


# ---------------------------------------------------------------------------
# 8. quapro de Casas & Pola (cœur Fortran local, appelé via ctypes)
# ---------------------------------------------------------------------------

class QuaproSolver(QPSolver):

    """Signature exacte (src/quaproc.f90) :

        subroutine quaproc(n, H, p, mi, md, C, d, ci, cs, ira,
             modo, x0, itermax, imp, io, x, f, lagr, iter, info)


    D'où la traduction depuis le canonique (Ax >= b) :
        H = G,  p = c,  mi = 0,  md = m,  C = -A'  (i.e. -Ax <= -b),
        d = -b,  ira = 0,  modo = 1.
    """

    name = "quapro"
    pip_hint = "compiler le cœur quapro en libquapro.so (build local)"

    _CANDIDATE_LIBS = ("libquapro.so")
    # répertoires (relatifs au dossier du script) où chercher le cœur,
    # dans l'ordre : à côté du script, puis build local du projet quapro.
    _CANDIDATE_DIRS = (".")

    BIG = 1.0e30          # convention Scilab pour une borne infinie
    # BLAS/LAPACK que le cœur peut laisser non résolues (ex. dnrm2_) si la
    # lib a été éditée sans lier -llapack -lblas : on les précharge alors
    # en global avant de réessayer (cf. _load_lib).
    _RUNTIME_LIBS = ("liblapack.so.3", "liblapack.so", "libopenblas.so.0",
                     "libblas.so.3", "libblas.so", "liblapack.dylib",
                     "libopenblas.0.dylib", "libblas.dylib")

    def __init__(self, entry: str = "quaproc", itermax: int = 0,
                 imp: int = 0, io: int = 6):
        # routine d'entrée exportée (sans l'underscore gfortran)
        self.itermax = itermax   # <= 0 -> défaut du cœur : 14*(n+mi+md)
        self.imp = imp           # niveau d'impression (0 = silencieux)
        self.io = io             # canal de sortie des impressions
        self.entry = entry
        self._fn = None

    def _lib_path(self) -> Optional[str]:
        env = os.environ.get("QUAPRO_LIB")
        if env and os.path.exists(env):
            return env
        here = os.path.dirname(os.path.abspath(__file__))
        for d in self._CANDIDATE_DIRS:
            base = os.path.normpath(os.path.join(here, d))
            for name in self._CANDIDATE_LIBS:
                p = os.path.join(base, name)
                if os.path.exists(p):
                    return p
        return None

    def available(self) -> bool:
        return self._lib_path() is not None

    def _load_lib(self, path: str):
        """Charge libquapro ; si des symboles BLAS/LAPACK restent non
        résolus (ex. dnrm2_), précharge une BLAS en global et réessaie."""
        try:
            return ctypes.CDLL(path)
        except OSError:
            for name in self._RUNTIME_LIBS:
                try:
                    ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
            return ctypes.CDLL(path)

    def _entry(self):
        if self._fn is None:
            path = self._lib_path()
            if path is None:
                raise RuntimeError("libquapro introuvable (variable QUAPRO_LIB ?)")
            lib = self._load_lib(path)
            try:
                fn = getattr(lib, self.entry + "_")
            except AttributeError as exc:
                raise RuntimeError(
                    f"symbole {self.entry}_ introuvable dans {path} — lister "
                    "les symboles avec `nm -g` et ajuster "
                    "QuaproSolver(entry=...)") from exc
            d = ctypes.POINTER(ctypes.c_double)
            i = ctypes.POINTER(ctypes.c_int)
            # quaproc(n, H, p, mi, md, C, d, ci, cs, ira, modo, x0,
            #             itermax, imp, io, x, f, lagr, iter, info)
            fn.argtypes = [i, d, d, i, i, d, d, d, d, i, i, d,
                           i, i, i, d, d, d, i, i]
            fn.restype = None
            self._fn = fn
        return self._fn

    @staticmethod
    def _dp(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def _solve(self, pb: QPProblem) -> SolverResult:
        fn = self._entry()
        n, m = pb.n, pb.m
        # --- traduction canonique (Ax >= b) -> quaproc ----------------
        H = np.asfortranarray(pb.G, dtype=np.float64)
        p = np.ascontiguousarray(pb.c, dtype=np.float64)
        C = np.asfortranarray((-pb.A).T, dtype=np.float64)  # (n, m) : -Ax <= -b
        d = np.ascontiguousarray(-pb.b, dtype=np.float64)
        ci = np.full(n, -self.BIG)   # ignorés (ira = 0) mais transmis
        cs = np.full(n, self.BIG)
        x0 = np.zeros(n)             # ignoré (modo = 1)
        # sorties
        x = np.zeros(n)
        fobj = ctypes.c_double()
        # lagr : bornes (n, non écrites avec ira = 0) puis contraintes (m)
        lagr = np.zeros(n + m)
        itc = ctypes.c_int()
        info = ctypes.c_int()
        fn(ctypes.byref(ctypes.c_int(n)), self._dp(H), self._dp(p),
           ctypes.byref(ctypes.c_int(0)),            # mi : aucune égalité
           ctypes.byref(ctypes.c_int(m)),            # md : toutes inégalités
           self._dp(C), self._dp(d), self._dp(ci), self._dp(cs),
           ctypes.byref(ctypes.c_int(0)),            # ira : pas de bornes
           ctypes.byref(ctypes.c_int(1)),            # modo : pas de x0
           self._dp(x0), ctypes.byref(ctypes.c_int(self.itermax)),
           ctypes.byref(ctypes.c_int(self.imp)),
           ctypes.byref(ctypes.c_int(self.io)),
           self._dp(x), ctypes.byref(fobj), self._dp(lagr),
           ctypes.byref(itc), ctypes.byref(info))
        status = "ok" if info.value == 0 else f"info={info.value}"
        lam = lagr[n:]   # duaux des contraintes (informatifs : les
        # métriques KKT sont recalculées indépendamment par evaluate())
        return SolverResult(x=x, status=status, iterations=itc.value,
                            lam=lam,
                            info={"fobj_fortran": fobj.value,
                                  "info_quapro": info.value})


# ---------------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------------

SOLVER_CLASSES: Tuple[type, ...] = (
    IPQuadProgSolver,
    QuaproSolver,
    QuadprogSolver,
    CvxoptSolver,
    OsqpSolver,
    ClarabelSolver,
    ScipySlsqpSolver,
    ScipyTrustConstrSolver,
)

SOLVERS_BY_NAME: Dict[str, type] = {cls.name: cls for cls in SOLVER_CLASSES}


def available_solvers() -> List[QPSolver]:
    """Toutes les instances de solveurs dont la bibliothèque est présente."""
    return [cls() for cls in SOLVER_CLASSES if cls().available()]


__all__ = [
    "QPProblem", "make_random_qp", "SolverResult", "QualityMetrics",
    "evaluate", "QPSolver", "IPQuadProgSolver", "QuadprogSolver",
    "CvxoptSolver", "OsqpSolver", "ClarabelSolver", "ScipySlsqpSolver",
    "ScipyTrustConstrSolver", "QuaproSolver",
    "SOLVER_CLASSES", "SOLVERS_BY_NAME", "available_solvers",
]
