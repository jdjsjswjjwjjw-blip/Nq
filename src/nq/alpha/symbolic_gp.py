"""اكتشاف معادلات/برامج رمزية بلا ``if`` — DEAP (GP) + gplearn (انحدار رمزي).

الهدف: توسيع فضاء المرشّحين خارج شبكة القواعد اليدوية، مع الإبقاء على
**نفس حكم المشروع**: لياقة على التدريب فقط، ثم walk-forward purged + IC/perm.

* **لا شروط if** في مجموعة العوامل: ``+ - * / neg abs`` فقط (برامج حسابية ناعمة).
* الاعتماد اختياري: ``pip install 'nq[gp]'`` → deap + gplearn + scikit-learn.
* الناتج أعمدة إشارة + نصّ المعادلة؛ يُمرَّر لـ ``walk_forward_select_hypotheses``.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.alpha.signals import align_forward_returns, evaluate_signal
from nq.contracts.temporal import AVAILABILITY_TS
from nq.models.splitting import purged_walk_forward_split
from nq.research.progress import ProgressLike
from nq.statistics.metrics import information_coefficient

FloatArray = npt.NDArray[np.float64]
Backend = Literal["deap", "gplearn", "both"]

_MIN_ROWS = 40
_MIN_FEATURES = 1
_MIN_IC_SAMPLES = 8
_MATRIX_NDIM = 2
_EPS = 1e-8
_CX_PROB = 0.5
_MUT_PROB = 0.2

# مجموعة عوامل بلا if / بلا دوال مثلث/لوغ تؤدّي لتفجير رقمي سهل
_ARITH_OPS = ("add", "sub", "mul", "div", "neg", "abs")


def _ephemeral_uniform() -> float:
    return float(random.uniform(-1.0, 1.0))


@dataclass(frozen=True, slots=True)
class SymbolicProgram:
    """برنامج/معادلة مكتشفة مع الإشارة المادية."""

    name: str
    backend: str
    expression: str
    values: FloatArray
    train_ic: float


@dataclass(frozen=True, slots=True)
class SymbolicSearchResult:
    """نتائج بحث رمزي: برامج + إطار أعمدة + طيات walk-forward إن وُجدت."""

    programs: tuple[SymbolicProgram, ...]
    frame: pl.DataFrame
    fold_selections: pl.DataFrame
    oos_ic: float
    oos_pvalue: float
    oos_n: int
    best_name: str | None


def require_gp_deps() -> None:
    """يرفع ImportError واضح إن نقصت حزمة ``[gp]``."""
    missing: list[str] = []
    try:
        import deap  # noqa: F401
    except ImportError:
        missing.append("deap")
    try:
        import gplearn  # noqa: F401
    except ImportError:
        missing.append("gplearn")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        raise ImportError(
            "Symbolic GP requires optional deps: "
            f"pip install 'nq[gp]' (missing: {', '.join(missing)})"
        )


def protected_div(a: float | FloatArray, b: float | FloatArray) -> float | FloatArray:
    """قسمة محمية: عند |b|≈0 تُرجع 1 (تمنع انفجار الأشجار)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    out = np.ones_like(a_arr, dtype=np.float64)
    mask = np.abs(b_arr) > _EPS
    np.divide(a_arr, b_arr, out=out, where=mask)
    if out.shape == ():
        return float(out)
    return out


def feature_matrix(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[FloatArray, tuple[str, ...]]:
    """يستخرج مصفوفة ميزات محدودة ومتناهية من إطار سببي."""
    cols = [c for c in feature_columns if c in frame.columns]
    if len(cols) < _MIN_FEATURES:
        raise ValueError(f"need >= {_MIN_FEATURES} feature columns present, got {cols}")
    mats = [frame[c].cast(pl.Float64).fill_null(0.0).to_numpy() for c in cols]
    x = np.column_stack(mats).astype(np.float64, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, tuple(cols)


def _spearman_ic(pred: FloatArray, y: FloatArray) -> float:
    mask = np.isfinite(pred) & np.isfinite(y)
    if int(mask.sum()) < _MIN_IC_SAMPLES or float(np.std(pred[mask])) == 0:
        return 0.0
    return float(information_coefficient(pred[mask], y[mask], method="spearman"))


def _unique_name(prefix: str, expression: str, used: set[str]) -> str:
    """اسم مستقر عبر العمليات — ``hash()`` بايثون مملّح لكل عملية فلا يُستخدم."""
    digest = hashlib.sha1(f"{prefix}|{expression}".encode()).hexdigest()[:10]
    name = f"{prefix}__{digest}"
    i = 0
    while name in used:
        i += 1
        digest = hashlib.sha1(f"{prefix}|{expression}|{i}".encode()).hexdigest()[:10]
        name = f"{prefix}__{digest}"
    used.add(name)
    return name


def evolve_deap(  # noqa: PLR0915
    x: FloatArray,
    y: FloatArray,
    feature_names: Sequence[str],
    *,
    x_full: FloatArray | None = None,
    population_size: int = 80,
    generations: int = 12,
    tournament_size: int = 4,
    max_depth: int = 4,
    seed: int = 0,
    n_hof: int = 3,
    progress: ProgressLike | None = None,
) -> list[tuple[str, FloatArray, float]]:
    """GP حسابي (بلا if) عبر DEAP — يُعيد أفضل برامج على عيّنة التدريب.

    إن مُرِّر ``x_full`` تُحسب قيم الإشارة على كل الصفوف (للتطبيق خارج التدريب).
    """
    require_gp_deps()
    from deap import base, creator, gp, tools

    if x.ndim != _MATRIX_NDIM or x.shape[0] != y.shape[0]:
        raise ValueError("x/y shape mismatch for DEAP")
    n_feat = x.shape[1]
    if n_feat != len(feature_names):
        raise ValueError("feature_names length must match x columns")
    x_out = x if x_full is None else x_full
    if x_out.ndim != _MATRIX_NDIM or x_out.shape[1] != n_feat:
        raise ValueError("x_full must match feature width of x")

    # عزل creators داخل استدعاء واحد لتجنّب تعارض الاسم عبر الاستدعاءات
    fit_name = f"FitnessIC_{seed}_{id(x)}"
    ind_name = f"Individual_{seed}_{id(x)}"
    for name in (fit_name, ind_name):
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create(fit_name, base.Fitness, weights=(1.0,))
    creator.create(ind_name, gp.PrimitiveTree, fitness=getattr(creator, fit_name))

    random.seed(int(seed))

    pset = gp.PrimitiveSet("MAIN", n_feat)
    pset.renameArguments(**{f"ARG{i}": feature_names[i] for i in range(n_feat)})
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="div")
    pset.addPrimitive(np.negative, 1, name="neg")
    pset.addPrimitive(np.abs, 1, name="abs")
    # اسم فريد لكل استدعاء — DEAP يسجّل ephemeral عالميًا بالاسم
    eph_name = f"rand_{seed}_{n_feat}"
    with suppress(Exception):
        pset.addEphemeralConstant(eph_name, partial(_ephemeral_uniform))

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, getattr(creator, ind_name), toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def _predict(individual: Any, matrix: FloatArray) -> FloatArray:
        func = toolbox.compile(expr=individual)
        cols = [matrix[:, i] for i in range(n_feat)]
        pred = np.asarray(func(*cols), dtype=np.float64)
        if pred.ndim == 0:
            pred = np.full(matrix.shape[0], float(pred), dtype=np.float64)
        return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    def _eval(individual: Any) -> tuple[float,]:
        try:
            pred = _predict(individual, x)
            return (abs(_spearman_ic(pred, y)),)
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError, TypeError):
            return (0.0,)

    toolbox.register("evaluate", _eval)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator_len, max_value=2**max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator_len, max_value=2**max_depth))

    rng = np.random.default_rng(seed)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(max(1, n_hof))
    for gen in range(generations):
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses, strict=True):
            ind.fitness.values = fit
        hof.update(pop)
        if progress is not None:
            best = float(hof[0].fitness.values[0]) if len(hof) else 0.0
            progress.op(f"DEAP gen {gen + 1}/{generations} · best_|IC|={best:.4g}")
            progress.heartbeat(gen + 1, generations, label="DEAP", force=True)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for i in range(1, len(offspring), 2):
            if rng.random() < _CX_PROB:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
        for i in range(len(offspring)):
            if rng.random() < _MUT_PROB:
                (offspring[i],) = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid), strict=True):
            ind.fitness.values = fit
        pop[:] = offspring

    results: list[tuple[str, FloatArray, float]] = []
    seen: set[str] = set()
    for ind in hof:
        expr = str(ind)
        if expr in seen:
            continue
        seen.add(expr)
        try:
            pred_full = _predict(ind, x_out)
            pred_tr = _predict(ind, x)
            results.append((expr, pred_full, abs(_spearman_ic(pred_tr, y))))
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
    return results


def operator_len(individual: Any) -> int:
    return len(individual)


def evolve_gplearn(
    x: FloatArray,
    y: FloatArray,
    feature_names: Sequence[str],
    *,
    population_size: int = 80,
    generations: int = 12,
    tournament_size: int = 10,
    max_depth: int = 4,
    seed: int = 0,
    n_hof: int = 3,
    progress: ProgressLike | None = None,
) -> list[tuple[str, FloatArray, float]]:
    """انحدار رمزي gplearn (Spearman) — عوامل حسابية بلا if."""
    require_gp_deps()
    from gplearn.genetic import SymbolicRegressor

    if progress is not None:
        progress.op(
            f"gplearn: pop={population_size} · gens={generations} · "
            f"features={len(feature_names)} · rows={x.shape[0]:,}"
        )

    # عدّة بذور → برامج متنوّعة (قاعة شرف يدوية)
    results: list[tuple[str, FloatArray, float]] = []
    seen: set[str] = set()
    for k in range(max(1, n_hof)):
        model = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=0.0,
            const_range=(-1.0, 1.0),
            init_depth=(1, max_depth),
            function_set=_ARITH_OPS,
            metric="spearman",
            parsimony_coefficient=0.001,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=1.0,
            feature_names=list(feature_names),
            verbose=0,
            random_state=int(seed) + 17 * k,
            n_jobs=1,
            low_memory=True,
        )
        if progress is not None:
            progress.op(f"gplearn fit {k + 1}/{n_hof}: pop={population_size} · gens={generations}…")
        # gplearn يُعظّم الارتباط؛ نمرّر y كما هو
        y_fit = np.nan_to_num(y.astype(np.float64), nan=0.0)
        model.fit(x, y_fit)
        expr = str(model._program)
        if expr in seen:
            continue
        seen.add(expr)
        pred = np.asarray(model.predict(x), dtype=np.float64)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        ic = abs(_spearman_ic(pred, y))
        results.append((expr, pred, ic))
        if progress is not None:
            progress.op(f"gplearn program {k + 1}/{n_hof}: |IC|={ic:.4g} · {expr[:80]}")
            progress.heartbeat(k + 1, n_hof, label="gplearn", force=True)
    return results


def materialize_programs_on_frame(
    frame: pl.DataFrame,
    programs: Sequence[SymbolicProgram],
) -> pl.DataFrame:
    """يلصق قيم البرامج كأعمدة على الإطار (نفس الطول/الترتيب)."""
    if frame.height == 0 or not programs:
        return frame
    exprs = [pl.Series(p.name, p.values).alias(p.name) for p in programs]
    return frame.with_columns(exprs)


def discover_symbolic_on_train(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
    *,
    price_col: str = "nq_close",
    horizon: int = 1,
    backend: Backend = "both",
    population_size: int = 80,
    generations: int = 12,
    max_depth: int = 4,
    seed: int = 0,
    n_programs: int = 3,
    train_idx: npt.NDArray[np.int_] | None = None,
    label_cutoff_idx: int | None = None,
    progress: ProgressLike | None = None,
) -> list[SymbolicProgram]:
    """يكتشف معادلات على شريحة تدريب فقط، ثم يطبّقها على كل الصفوف.

    ``label_cutoff_idx``: أقصى فهرس مسموح لهدف العائد الأمامي (عادة بداية الاختبار).
    أي صف تدريب يحتاج ``t+horizon >= cutoff`` يُستبعَد من اللياقة — يمنع تسرّب التسمية.
    """
    require_gp_deps()
    work = frame.sort(AVAILABILITY_TS)
    x_all, names = feature_matrix(work, feature_columns)
    y_all = align_forward_returns(work[price_col].to_numpy().astype(np.float64), horizon=horizon)
    if train_idx is None:
        train_idx = np.arange(work.height, dtype=np.int64)
    tr = np.asarray(train_idx, dtype=np.int64)
    if label_cutoff_idx is not None:
        # العائد الأمامي عند t يستخدم السعر حتى t+horizon — يجب أن يبقى داخل الماضي
        tr = tr[tr + int(horizon) < int(label_cutoff_idx)]
    if tr.size < _MIN_ROWS:
        if progress is not None:
            progress.op(f"symbolic: تخطّي — train rows={tr.size} < {_MIN_ROWS}")
        return []

    x_tr, y_tr = x_all[tr], y_all[tr]
    mask = np.isfinite(y_tr)
    x_tr, y_tr = x_tr[mask], y_tr[mask]
    if x_tr.shape[0] < _MIN_ROWS:
        return []

    raw: list[tuple[str, str, FloatArray, float]] = []
    if backend in ("deap", "both"):
        if progress is not None:
            progress.op("symbolic: تشغيل DEAP (بلا if)")
        for expr, pred_full, ic in evolve_deap(
            x_tr,
            y_tr,
            names,
            x_full=x_all,
            population_size=population_size,
            generations=generations,
            max_depth=max_depth,
            seed=seed,
            n_hof=n_programs,
            progress=progress,
        ):
            raw.append(("deap", expr, pred_full, ic))

    if backend in ("gplearn", "both"):
        if progress is not None:
            progress.op("symbolic: تشغيل gplearn (انحدار رمزي)")
        from gplearn.genetic import SymbolicRegressor

        for k in range(max(1, n_programs)):
            model = SymbolicRegressor(
                population_size=population_size,
                generations=generations,
                tournament_size=10,
                const_range=(-1.0, 1.0),
                init_depth=(1, max_depth),
                function_set=_ARITH_OPS,
                metric="spearman",
                parsimony_coefficient=0.001,
                feature_names=list(names),
                verbose=0,
                random_state=int(seed) + 17 * k,
                n_jobs=1,
                low_memory=True,
            )
            if progress is not None:
                progress.op(
                    f"gplearn[{k + 1}/{n_programs}] fit: "
                    f"pop={population_size} · gens={generations}…"
                )
            model.fit(x_tr, np.nan_to_num(y_tr, nan=0.0))
            expr = str(model._program)
            full = np.nan_to_num(
                np.asarray(model.predict(x_all), dtype=np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            pred_tr = np.asarray(model.predict(x_tr), dtype=np.float64)
            ic = abs(_spearman_ic(pred_tr, y_tr))
            raw.append(("gplearn", expr, full, ic))
            if progress is not None:
                progress.op(f"gplearn[{k + 1}]: |IC_train|={ic:.4g} · {expr[:80]}")

    used: set[str] = set()
    out: list[SymbolicProgram] = []
    for backend_name, expr, values, ic in raw:
        name = _unique_name(f"sym_{backend_name}", expr, used)
        out.append(
            SymbolicProgram(
                name=name,
                backend=backend_name,
                expression=expr,
                values=values.astype(np.float64, copy=False),
                train_ic=float(ic),
            )
        )
    return out


def _apply_deap_expression(
    expression: str,
    x: FloatArray,
    feature_names: Sequence[str],
) -> FloatArray:
    """يفسّر تعبير DEAP بسيط (بدون ephemeral) على مصفوفة كاملة — للاختبارات."""
    require_gp_deps()
    from deap import gp

    n_feat = len(feature_names)
    pset = gp.PrimitiveSet("MAIN", n_feat)
    pset.renameArguments(**{f"ARG{i}": feature_names[i] for i in range(n_feat)})
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="div")
    pset.addPrimitive(np.negative, 1, name="neg")
    pset.addPrimitive(np.abs, 1, name="abs")
    try:
        tree = gp.PrimitiveTree.from_string(expression, pset)
        func = gp.compile(tree, pset)
        cols = [x[:, i] for i in range(n_feat)]
        pred = np.asarray(func(*cols), dtype=np.float64)
        if pred.ndim == 0:
            pred = np.full(x.shape[0], float(pred), dtype=np.float64)
        return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    except (ValueError, TypeError, MemoryError, RecursionError):
        return np.zeros(x.shape[0], dtype=np.float64)


def search_symbolic_hypotheses(  # noqa: PLR0915
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
    *,
    price_col: str = "nq_close",
    horizon: int = 1,
    backend: Backend = "both",
    n_splits: int = 3,
    embargo: int = 0,
    purge_samples: int = 0,
    population_size: int = 60,
    generations: int = 8,
    max_depth: int = 3,
    n_programs: int = 2,
    n_permutations: int = 100,
    selection_aware_null: bool = True,
    seed: int = 0,
    progress: ProgressLike | None = None,
) -> SymbolicSearchResult:
    """بحث رمزي داخل purged walk-forward: تطوّر على train → قياس على test.

    لكل طيّة يُكتشف برنامج/برامج على التدريب فقط، تُلصق كأعمدة، ثم يُختار
    الأفضل بـ |IC| تدريب ويُقاس على الاختبار (نفس فلسفة شبكة FVG/Breakout).

    افتراضيًا ``purge_samples >= horizon`` لعزل أهداف العائد الأمامي عن كتلة الاختبار.

    ``selection_aware_null=True`` (افتراضي): p-value عبر إعادة اختيار أفضل برنامج
    مكتشف في كل طيّة تحت عوائد مخلوطة — دون إعادة تطوّر GP (مكلف).
    """
    require_gp_deps()
    work = frame.sort(AVAILABILITY_TS)
    empty_folds = pl.DataFrame(
        schema={
            "fold": pl.Int64(),
            "selected": pl.Utf8(),
            "train_ic": pl.Float64(),
            "test_ic": pl.Float64(),
            "employed_sign": pl.Float64(),
            "expression": pl.Utf8(),
            "backend": pl.Utf8(),
        }
    )
    if work.height < _MIN_ROWS or price_col not in work.columns:
        return SymbolicSearchResult((), work, empty_folds, 0.0, 1.0, 0, None)

    # عزل تسمية العائد الأمامي عن المستقبل (الحد الأدنى = أفق التقييم)
    effective_purge = max(int(purge_samples), int(horizon))
    times = work[AVAILABILITY_TS].to_numpy()
    folds = purged_walk_forward_split(
        times,
        n_splits=n_splits,
        embargo=embargo,
        purge_samples=effective_purge,
        min_train_size=max(20, work.height // (n_splits + 2)),
    )
    if progress is not None:
        progress.op(
            f"symbolic WF: {len(folds)} folds · backend={backend} · "
            f"pop={population_size} · gens={generations} · "
            f"embargo={embargo} · purge={effective_purge} · "
            f"selection_null={selection_aware_null}"
        )

    all_programs: list[SymbolicProgram] = []
    fold_programs: list[list[SymbolicProgram]] = []
    fold_rows: list[dict[str, float | int | str]] = []
    oos_values = np.full(work.height, np.nan, dtype=np.float64)
    oos_fwd = np.full(work.height, np.nan, dtype=np.float64)
    forward = align_forward_returns(work[price_col].to_numpy().astype(np.float64), horizon=horizon)
    rng = np.random.default_rng(seed)

    for fold_i, fold in enumerate(folds):
        if progress is not None:
            progress.op(
                f"symbolic fold {fold_i + 1}/{len(folds)} "
                f"(train={len(fold.train_idx):,} · test={len(fold.test_idx):,})"
            )
        programs = discover_symbolic_on_train(
            work,
            feature_columns,
            price_col=price_col,
            horizon=horizon,
            backend=backend,
            population_size=population_size,
            generations=generations,
            max_depth=max_depth,
            seed=seed + fold_i * 101,
            n_programs=n_programs,
            train_idx=fold.train_idx,
            label_cutoff_idx=int(fold.test_idx.min()),
            progress=progress,
        )
        if not programs:
            fold_programs.append([])
            fold_rows.append(
                {
                    "fold": fold_i,
                    "selected": "",
                    "train_ic": 0.0,
                    "test_ic": 0.0,
                    "employed_sign": 1.0,
                    "expression": "",
                    "backend": backend,
                }
            )
            continue
        # اختر أفضل |IC| تدريب ثم وظّف بعلامة train_ic
        best = max(programs, key=lambda p: abs(p.train_ic))
        all_programs.extend(programs)
        fold_programs.append(list(programs))
        employed_sign = 1.0 if best.train_ic >= 0.0 else -1.0
        employed = employed_sign * best.values
        test_ic = _spearman_ic(employed[fold.test_idx], forward[fold.test_idx])
        oos_values[fold.test_idx] = employed[fold.test_idx]
        oos_fwd[fold.test_idx] = forward[fold.test_idx]
        fold_rows.append(
            {
                "fold": fold_i,
                "selected": best.name,
                "train_ic": float(best.train_ic),
                "test_ic": float(test_ic),
                "employed_sign": float(employed_sign),
                "expression": best.expression,
                "backend": best.backend,
            }
        )
        if progress is not None:
            progress.op(
                f"symbolic fold {fold_i + 1}: {best.name} · "
                f"train|IC|={best.train_ic:.4g} · employed_sign={employed_sign:+.0f} · "
                f"test_ic={test_ic:.4g}"
            )

    fold_df = pl.DataFrame(fold_rows) if fold_rows else empty_folds
    # إطار بأعمدة كل البرامج المكتشفة (قد تتكرر الأسماء عبر الطيات — نُبقي الأخيرة)
    by_name: dict[str, SymbolicProgram] = {p.name: p for p in all_programs}
    unique_programs = tuple(by_name.values())
    out_frame = materialize_programs_on_frame(work, unique_programs)

    mask = np.isfinite(oos_values) & np.isfinite(oos_fwd)
    oos_n = int(mask.sum())
    if oos_n >= _MIN_IC_SAMPLES and float(np.std(oos_values[mask])) > 0:
        oos_ic = float(information_coefficient(oos_values[mask], oos_fwd[mask], method="spearman"))
        if selection_aware_null and n_permutations > 0 and any(fold_programs):
            if progress is not None:
                progress.op(
                    f"symbolic selection-under-null: {n_permutations} تبديل · "
                    f"fold_programs={sum(len(p) for p in fold_programs)}"
                )
            null_ics = np.empty(n_permutations, dtype=np.float64)
            for p_i in range(n_permutations):
                perm_fwd = rng.permutation(forward)
                null_oos = np.full(work.height, np.nan, dtype=np.float64)
                null_y = np.full(work.height, np.nan, dtype=np.float64)
                for fold_i, fold in enumerate(folds):
                    progs = fold_programs[fold_i]
                    if not progs:
                        continue
                    best_null = max(
                        progs,
                        key=lambda prog: abs(
                            _spearman_ic(prog.values[fold.train_idx], perm_fwd[fold.train_idx])
                        ),
                    )
                    null_train_ic = _spearman_ic(
                        best_null.values[fold.train_idx], perm_fwd[fold.train_idx]
                    )
                    employed_sign = 1.0 if null_train_ic >= 0.0 else -1.0
                    null_oos[fold.test_idx] = employed_sign * best_null.values[fold.test_idx]
                    null_y[fold.test_idx] = perm_fwd[fold.test_idx]
                nmask = np.isfinite(null_oos) & np.isfinite(null_y)
                if int(nmask.sum()) >= _MIN_IC_SAMPLES and float(np.std(null_oos[nmask])) > 0:
                    null_ics[p_i] = float(
                        information_coefficient(null_oos[nmask], null_y[nmask], method="spearman")
                    )
                else:
                    null_ics[p_i] = 0.0
                if progress is not None and ((p_i + 1) % max(1, n_permutations // 10) == 0):
                    progress.heartbeat(p_i + 1, n_permutations, label="sym-selection-null")
            oos_p = float((int(np.sum(np.abs(null_ics) >= abs(oos_ic))) + 1) / (n_permutations + 1))
        else:
            oos_ev = evaluate_signal(
                "symbolic_wf",
                oos_values[mask],
                oos_fwd[mask],
                n_permutations=n_permutations,
                rng=rng,
                progress=progress,
                progress_label="sym-oos-perm",
            )
            oos_ic, oos_p = float(oos_ev.ic), float(oos_ev.ic_pvalue)
    else:
        oos_ic, oos_p = 0.0, 1.0

    best_name: str | None = None
    if fold_df.height > 0 and "selected" in fold_df.columns:
        nonempty = fold_df.filter(pl.col("selected") != "")
        if nonempty.height > 0:
            counts = (
                nonempty.group_by("selected")
                .len()
                .sort(["len", "selected"], descending=[True, False])
            )
            best_name = str(counts["selected"][0])

    return SymbolicSearchResult(
        programs=unique_programs,
        frame=out_frame,
        fold_selections=fold_df,
        oos_ic=oos_ic,
        oos_pvalue=oos_p,
        oos_n=oos_n,
        best_name=best_name,
    )


def default_symbolic_feature_columns() -> tuple[str, ...]:
    """ميزات طرفية افتراضية — أنطولوجيا واحدة: deltas/trap/fail + ``vp_*`` (لا خلط streaming VA).

    تُستبعد ``near_vah`` / ``in_value_area`` / ``poc_dist_norm`` (مسار التيك) حتى لا
    تُخلط مع أعمدة الملف الحجمي المجمّع ``vp_*`` في نفس فضاء البحث.
    """
    return (
        "nq_delta",
        "mnq_delta",
        "trap_setup",
        "phase_balance",
        "phase_expansion",
        "fail_fvg",
        "fail_breakout",
        "vp_balance",
        "vp_imbalance",
        "vp_expansion",
        "vp_close_in_value",
        "vp_flip_to_imbalance",
    )


__all__ = [
    "Backend",
    "SymbolicProgram",
    "SymbolicSearchResult",
    "default_symbolic_feature_columns",
    "discover_symbolic_on_train",
    "evolve_deap",
    "evolve_gplearn",
    "feature_matrix",
    "materialize_programs_on_frame",
    "protected_div",
    "require_gp_deps",
    "search_symbolic_hypotheses",
]
