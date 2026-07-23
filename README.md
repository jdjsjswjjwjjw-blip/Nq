# Nq — نظام بحثي كمّي لبنية السوق الدقيقة (Market Microstructure Research Engine)

نظام بحثي كمّي متكامل مبني بالكامل على بيانات **MBO (Market By Order)** لعقود **NQ / MNQ**،
يمتد من إعادة بناء دفتر الأوامر إلى نموذج تأسيسي ذاتي الإشراف (SSL) ومساعد بحثي
مؤسَّس على الأدلّة لاكتشاف هياكل السوق وإشارات الألفا.

> الهدف: تحويل تدفّق الأوامر الخام إلى **تمثيلات كامنة لحالات السوق** قابلة
> للاختبار الإحصائي وللتفسير البحثي — بلا تسريب زمني، وبصرامة كمية.

**متطلبات التشغيل:** Python **≥ 3.11**

---

## المبادئ الحاكمة (Non‑Negotiable Principles)

هذه المبادئ الأربعة **ملزِمة في كل سطر كود وكل محطة**:

### 1) منع التسريب الزمني نهائيًا (Zero Temporal Leakage)
- كل حساب يعتمد فقط على المعلومات المتاحة حتى اللحظة `t` (Point‑in‑Time / Causal Only).
- ممنوع look‑ahead بأي شكل مباشر أو غير مباشر.
- كل ميزة تحمل طابعًا زمنيًا صريحًا (`event_ts`, `ingest_ts`, `availability_ts`).
- التقسيم Walk‑Forward / Purged + Embargo — لا عشوائي.
- التطبيع fit‑on‑past ثم تطبيق للأمام فقط.
- **OOS R² (Campbell–Thompson):** خط الأساس = متوسط أهداف **التدريب**، لا متوسط عيّنة الاختبار.
- **قمم/قيعان intraday:** `cum_max` / `cum_min` داخل `session_date` (ET) فقط — لا تراكمي عالمي عبر الأيام.
- **قاعدة PR:** إثبات خلوّ من التسريب (Leakage Test) قبل الدمج.

### 2) صرامة كمية وعلمية بلا أخطاء
- تعريف رياضي موثّق قبل الكود.
- اختبارات وحدة + خصائص لكل مخرج.
- دلالة إحصائية + متانة + تحقّق خارج العينة.
- حتمية كاملة (seeds، بيانات مُصدَّرة، deps مثبتة).

### 3) أداء عالٍ لبيانات ضخمة
- متجهي/عمودي؛ بلا حلقات بايثون على المسارات الساخنة.
- Parquet/Arrow + تدفّق؛ استخدم `--max-rows` أو شرائح يومية للملفات الضخمة.
- Benchmark قبل اعتماد المكوّنات الحرجة.

### 4) MBO فقط
- المصدر الوحيد للحقيقة هو تدفّق **MBO** الخام.
- كل الطبقات الأعلى تُشتق حصريًا من إعادة بناء الدفتر.

---

## المخطط المعماري (Architecture)

```
MBO Raw (Parquet / Arrow / CSV / .zst / Databento)
   → Ingestion + Order Book Reconstruction
   → Streaming State Machine (افتراضي) — دفتر حي · VP · Regimes · trap
        أو Simulation batch (اختياري: features.mode = "batch")
   → Unified Feature Frame (availability_ts = event_ts للبث)
   → ┌─ SSL (tick/event أو bucket)
      ├─ M9 Coverage Monitor
      └─ Alpha Screen (trap_setup, phase_*, fail_fvg, vp_*, …)
   → ResearchAssistant (فرضيات بأدلّة قابلة للتتبع)
   → Unified Report (Markdown + Parquet metrics)
```

التفاصيل: `docs/architecture.md` · عقود البيانات: `docs/data_contracts.md`

---

## خطوط التشغيل (Runbooks)

### 0) التثبيت

```bash
git clone <repo-url> Nq && cd Nq

# يفضَّل بيئة معزولة
python3.12 -m venv .venv && source .venv/bin/activate   # أو 3.11+

pip install -e ".[dev]"          # تطوير + اختبارات
pip install -e ".[dev,data]"     # + zstandard لقراءة .zst
```

ضع ملفات MBO تحت `data/raw/` أو مرّر المسار من CLI.

| الصيغة المدعومة | ملاحظات |
|-----------------|----------|
| `.parquet` / `.arrow` / `.ipc` | افتراضي |
| `.csv` | مدعوم |
| `.zst` | يحتاج `pip install -e ".[data]"` |
| Databento columns | تُطبَّع تلقائيًا عبر `normalize_databento_frame` |

---

### اختيار المسار — مهم

**ما فيش حذف لأي طبقة.** أوامر التشغيل المنفصلة (`run_fail_fvg` / `run_vp_auction`) **ليست خروجًا من المنظومة**: كلها تستدعي نفس المحرك (`run_research_pipeline`) وتمرّ بنفس المرّات (تحميل → ميزات → SSL ‖ M9 ‖ ألفا) وتكتب **نفس شكل المخرجات** (`report.md`, `features.parquet`, مقاييس SSL/M9/ألفا).

الفرق فقط: **أي إشارات تُفرَز** في قناة الألفا لهذه الجولة.

| الأمر | التركيز | داخل المنظومة؟ | المخرجات |
|--------|---------|----------------|----------|
| `run_week` + `configs/research.toml` | **الكل مع بعض** | نعم | كاملة |
| `run_fail_fvg` | Failed FVG (فرضية افتراضية) | نعم — أمر تشغيل منفصل فقط | كاملة (SSL‖M9‖ألفا) |
| `run_fail_fvg --search` | شبكة تايم فريم/إعدادات FVG + بوابة SSL | نعم — walk-forward بلا تسريب | تقرير بحث + folds + screen |
| `run_fail_breakout` | Failed Breakout (دخول=إغلاق شمعة) | نعم — أمر تشغيل منفصل فقط | كاملة (SSL‖M9‖ألفا) |
| `run_fail_breakout --search` | شبكة إعدادات FB + بوابة SSL كتأكيد | نعم — walk-forward بلا تسريب | تقرير بحث + folds + screen |
| `run_vp_auction` + `configs/vp_auction.toml` | VP + توازن/اختلال | نعم — أمر تشغيل منفصل فقط | كاملة (SSL‖M9‖ألفا) |

> لو عايز الكل شغّال → `run_week`.  
> لو عايز فرضية واحدة للفرز → الأمر المنفصل المناسب (نفس المعالجة والمخرجات).  
> لو عايز **أفضل تايم فريم/إعدادات** لـ FVG → `run_fail_fvg --search`.  
> لو عايز Failed Breakout → `run_fail_breakout` (أو `--search` لشبكة الفرضيات + SSL gate).

---

### 1) الخط الموحّد — من MBO إلى التقرير (`run_week`)

نقطة الدخول الأساسية: **SSL ‖ M9 ‖ ألفا** في تقرير واحد.  
الإشارات الافتراضية معًا: `trap_setup` / `lead_lag` / `fail_fvg` / `vp_balance` / `vp_imbalance` / …

```bash
# NQ فقط (بدون ملف MNQ منفصل) + حد ذاكرة
python scripts/run_week.py \
  --nq /path/to/nq.parquet \
  --nq-only \
  --max-rows 500000 \
  --output data/runs/latest

# NQ + MNQ
python scripts/run_week.py \
  --nq data/raw/nq.parquet \
  --mnq data/raw/mnq.parquet \
  --config configs/research.toml \
  --output data/runs/w29
```

**المخرجات** في `--output` (نفس الشكل لكل الأوامر):

| ملف | المحتوى |
|-----|---------|
| `report.md` | التقرير الموحّد (SSL + M9 + ألفا) |
| `features.parquet` | إطار الميزات |
| `ssl_metrics.parquet` | مقاييس SSL |
| `coverage_metrics.parquet` | مقاييس M9 |
| `alpha_evaluations.parquet` | فرز الإشارات |

**تقدّم التشغيل (stderr + `progress.log`):** كل خطوة وكل عملية داخلها تُطبع سطرًا بسطر
(`→` للخطوات، `-` للعمليات، `…` لنسبة التقدّم داخل الحلقات الطويلة مثل tick_stream / walk-forward / M9).
يغطي: تحميل MBO، الميزات، SSL (tick أو bucket) طيّةً بطِيّة، اكتشاف الألفا إشارةً بإشارة،
مقاييس M9، وبحث فرضيات FVG/Breakout. عند تحديد `--output` يُكتب نفس اللوج إلى `progress.log`.
عطّل بـ `--quiet` أو `[run] quiet = true`.

**الإعدادات:** `configs/research.toml`

| قسم | أهم المفاتيح |
|-----|----------------|
| `[data]` | `nq_path`, `mnq_path`, `cross_market_mode` (`nq_only` / `dual`), `max_rows` |
| `[ssl]` | `mode` = `tick` \| `bucket`, `window`, `n_components` |
| `[features]` | `mode` = `streaming` (افتراضي) \| `batch` |
| `[signals]` | `include_failed_fvg`, `include_auction_vp`, قائمة `columns` للفرز |
| `[run]` | `quiet` = تعطيل طباعة التقدّم |
| `[execution]` | `mode` = `intraday` \| `mid`, slippage |
| `[temporal]` | `interval_ns`, `horizon` |

---

### 2) أمر منفصل: Failed FVG (`run_fail_fvg`)

أمر تشغيل **منفصل** لفرز Failed FVG — **بدون** الخروج من المنظومة:  
نفس خط المعالجة الكامل ونفس المخرجات (`report.md` + parquet). يضيّق فقط أعمدة الفرز على `fail_fvg` (+ سياق cross-market).

```bash
# أمر منفصل — مخرجات كاملة في data/runs/fail_fvg
python scripts/run_fail_fvg.py \
  --nq /path/to/nq.parquet \
  --max-rows 500000 \
  --output data/runs/fail_fvg

# بحث تايم فريم + إعدادات + بوابة SSL سببية (walk-forward / بلا تسريب)
python scripts/run_fail_fvg.py \
  --nq /path/to/nq.parquet \
  --search \
  --max-rows 500000 \
  --output data/runs/fail_fvg_search

# أو عبر run_week + إعداد مركّز (الفرضية الافتراضية فقط)
python scripts/run_week.py \
  --config configs/fail_fvg.toml \
  --nq /path/to/nq.parquet \
  --nq-only \
  --max-rows 500000
```

**`--search` ماذا يفعل (داخل المبادئ الأربعة):**

| مبدأ | التطبيق |
|------|---------|
| منع التسريب | إشارات asof خلفي؛ اختيار الإعداد على **train فقط**؛ قياس OOS على **test** (purged walk-forward) |
| صرامة كمية | IC + permutation؛ BH استكشافي على الشبكة؛ الحكم = IC خارج العينة |
| أداء | كاش شموع OHLCV حسب `interval_ns` |
| MBO فقط | الفرضيات من شريط صفقات MBO → OHLCV → FVG |

SSL هنا **بوابة ظرف** (`z0` + كمّية ماضية)، مش مولّد قواعد FVG جديدة.

**مخرجات `--search`** في `--output`:

| ملف | المحتوى |
|-----|---------|
| `report.md` | تقرير البحث (IC خارج العينة + أدلّة) |
| `features.parquet` | ساعة التقييم + أعمدة الفرضيات (و`__ssl` إن وُجدت) |
| `fold_selections.parquet` | الفرضية المختارة لكل طيّة train→test |
| `exploratory_screen.parquet` | فرز BH استكشافي (ليس أساس الاختيار) |
| `ssl_metrics.parquet` | مقاييس SSL عند تفعيل البوابة |

> في الخط العام: `include_failed_fvg = true` يُلحق `fail_fvg` **مع** باقي الإشارات.  
> `run_fail_fvg` = جولة فرز مركّزة؛ `--search` = بحث إعدادات/تايم فريم فوق نفس المحرك.

---

### 3) أمر منفصل: Failed Breakout (`run_fail_breakout`)

كسر فاشل (Failed Breakout) من MBO → شموع سببية:

* جهد عالٍ مقابل متوسطات **ماضية** فقط.
* كسر مدى آخر N شموع ثم إغلاق عكسي + تأكيد SMA اختياري.
* **إصلاح تسريب الدخول:** الإشارة عند إغلاق الشمعة؛ `fb_entry_ref = close`؛  
  `fb_break_level` تحليلي فقط — التقييم عبر مسار الألفا (bid/ask + slippage)، مش ملء عند مستوى الكسر.
* **عمق لا يُطمس:** عند الدخول تُلتقط لقطة سلم دفتر (L1–L5) عند `bucket_end`؛
  أثناء المراقبة تبقى أعمدة `depth_*` + سيولة VAH/VAL/trail؛ التنفيذ والخروج
  بمسح السيولة الظاهرة (`execution_forward_returns_depth`) بلا اختلاق عمق.

```bash
python scripts/run_fail_breakout.py \
  --nq /path/to/nq.parquet \
  --max-rows 500000 \
  --output data/runs/fail_breakout

# شبكة فرضيات + بوابة SSL كتأكيد (walk-forward)
python scripts/run_fail_breakout.py \
  --nq /path/to/nq.parquet \
  --search \
  --max-rows 500000 \
  --output data/runs/fail_breakout_search
```

مع `--search` (افتراضي): SSL يولّد **مرشّحي تعزيز** (`ssl_abs_q*`, `ssl_sign_*`, `ctx_*`) فوق نواة Failed Breakout،
ثم walk-forward يختار الأفضل خارج العينة. عطّل بـ `--no-enhance` أو `--no-ssl-gate`.

| عمود | المعنى |
|------|--------|
| `fail_breakout` | `+1` LONG / `−1` SHORT / `0` |
| `fb_entry_ref` | مرجع دخول قابل للتنفيذ (إغلاق شمعة الإشارة) |
| `fb_break_level` | مستوى الكسر الفاشل (ليس سعر ملء) |
| `fb_depth_at_break` | سيولة ظاهرة عند مستوى الكسر (من السلم) |
| `depth_cum_*` / `depth_*_sz_k` | سلم عمق L1–L5 للمراقبة والتنفيذ/الخروج |
| `*__enh__*` | تعزيزات SSL/سياق مرشّحة (عند `--search`) |

---

### 4) أمر منفصل: Volume Profile + التوازن/الاختلال (`run_vp_auction`)

أمر تشغيل **منفصل** لفرضيات الملف الحجمي والسوق المتوازن/غير المتوازن (NQ فقط).  
نفس المنطق: داخل المنظومة، معالجة كاملة، مخرجات كاملة — مع تضييق الفرز على إشارات VP.

| إشارة | المعنى |
|--------|--------|
| `vp_balance` | `+1` متوازن / `−1` مختلّ |
| `vp_imbalance` | `1` عند الاختلال |
| `vp_expansion` | تمدّد المدى |
| `vp_close_in_value` | إغلاق داخل [VAL, VAH] |
| `vp_flip_to_imbalance` | انتقال توازن → اختلال |

```bash
# أمر منفصل — مخرجات كاملة في data/runs/vp_auction
python scripts/run_vp_auction.py \
  --nq /path/to/nq.parquet \
  --max-rows 500000 \
  --output data/runs/vp_auction

# أو عبر run_week + إعداد مركّز
python scripts/run_week.py \
  --config configs/vp_auction.toml \
  --nq /path/to/nq.parquet \
  --nq-only \
  --max-rows 500000
```

> في الخط العام (`configs/research.toml`): `include_auction_vp = true` يُلحق إشارات VP **مع** باقي الإشارات، بدون استبدالها.

---

### 4) من بايثون (API)

```python
from pathlib import Path
from nq.research.orchestrator import PipelineConfig, run_research_pipeline
from nq.strategies.fail_fvg import run_fail_fvg_research
from nq.strategies.fvg_hypothesis import search_fail_fvg_hypotheses
from nq.strategies.vp_auction import run_vp_auction_research

# الخط الكامل
cfg = PipelineConfig.from_toml("configs/research.toml")
result = run_research_pipeline(
    "data/raw/nq.parquet",
    "data/raw/nq.parquet",          # أو mnq؛ مع nq_only يُكرَّر NQ
    config=cfg,
    output_dir=Path("data/runs/api"),
)
print(result.report.to_markdown())
assert "fail_fvg" in result.features.columns
assert "vp_balance" in result.features.columns

# تركيز Failed FVG (فرضية افتراضية)
fvg = run_fail_fvg_research(
    "data/raw/nq.parquet",
    max_rows=500_000,
    output_dir="data/runs/fail_fvg",
)
print(fvg.unified.to_markdown())

# بحث تايم فريم/إعدادات FVG + بوابة SSL سببية
search = search_fail_fvg_hypotheses(
    "data/raw/nq.parquet",
    use_ssl_gate=True,
    max_rows=500_000,
    output_dir="data/runs/fail_fvg_search",
)
print(search.report.to_markdown())
print(search.best_oos_spec, search.oos_selected_ic)

# تركيز VP / توازن·اختلال (NQ فقط)
vp = run_vp_auction_research(
    "data/raw/nq.parquet",
    max_rows=500_000,
    output_dir="data/runs/vp_auction",
)
print(vp.unified.to_markdown())
```

---

### 5) تدفق البيانات داخل الخط الموحّد

```
load_mbo_frame (Databento normalize + null-price sanitize + max_rows)
  → [features.mode=streaming] build_streaming_research_features
       # آلة حالة: OrderBook + DevelopingVolumeProfile + CausalRegimeTracker
       # availability_ts = event_ts؛ عيّنة = آخر حالة في كل interval
  → [features.mode=batch] cross_market_features   # نوافذ مجمّعة (اختياري)
  → asof-join failed_fvg_features  # fail_fvg, effort_*  (خلفي فقط)
  → asof-join auction_signal_frame # vp_balance, vp_imbalance, … (خلفي فقط)
  → ┌ run_ssl_tick_pipeline  أو  run_ssl_pipeline
    ├ run_coverage_on_features     # + كتلة streaming_microstructure
    └ discover_alpha_from_features # IC + BH على columns من [signals]
  → build_unified_report
```

**تدفق `--search` (FVG hypothesis search):**

```
MBO
  → شبكة فرضيات (تايم فريم + عتبات)  # failed_fvg_from_bars + كاش OHLCV
  → asof على ساعة التقييم            # خلفي فقط
  → بوابة SSL اختيارية               # z0 asof + كمّية ماضية
  → walk-forward purged              # اختيار على train → IC على test
  → تقرير + fold_selections + screen
```

**SSL**

| `ssl.mode` | المدخل | الإخفاء |
|------------|--------|---------|
| `tick` (افتراضي) | MBO event + دفتر حي + VP + عمق VAH/VAL | هيكلي (`masking_structural`) |
| `bucket` | أعمدة الإشارة المجمّعة | عشوائي (`mask_matrix`) |

---

### 6) بوابات الجودة (محلي + CI)

```bash
ruff check src tests
ruff format --check src tests
mypy                          # strict على src + tests
pytest --cov                  # وحدة + تسريب + خصائص
```

CI: `.github/workflows/ci.yml` على كل push/PR إلى `main`.

---

### 7) نصائح تشغيل على بيانات حقيقية

1. **Python ≥ 3.11** — المشروع يرفض أقل من ذلك في السكربتات.
2. **الذاكرة:** ملف شهر كامل (~300M صف) — استخدم `--max-rows` أو شريحة يومية؛ لا تحمّل الشهر دفعة واحدة.
3. **NQ فقط:** `--nq-only` أو `cross_market_mode = "nq_only"` في TOML.
4. **أسعار Databento float:** تُحوَّل تلقائيًا إلى fixed-point عبر `PRICE_SCALE`.
5. **أسعار null (Clear):** تُعالَج في `sanitize_mbo_frame` قبل إعادة بناء الدفتر.

---

## هيكل المستودع

```
Nq/
├── README.md
├── configs/
│   ├── default.toml
│   ├── research.toml          # الخط العام — كل الإشارات معًا
│   ├── fail_fvg.toml          # أمر FVG منفصل (فرز مركّز، مخرجات كاملة)
│   └── vp_auction.toml        # أمر VP منفصل (فرز مركّز، مخرجات كاملة)
├── scripts/
│   ├── run_week.py            # الخط الموحّد MBO → تقرير
│   ├── run_fail_fvg.py        # FVG منفصل (+ --search للشبكة/SSL gate)
│   └── run_vp_auction.py      # أمر منفصل VP / توازن·اختلال (داخل المنظومة)
├── docs/
│   ├── architecture.md
│   └── data_contracts.md
├── data/                      # raw / runs (محلي)
├── src/nq/
│   ├── contracts/             # MBO schema + زمني
│   ├── core/                  # حتمية، جلسة، سياسة زمنية
│   ├── ingestion/             # قارئ + Databento
│   ├── orderbook/             # إعادة بناء الدفتر
│   ├── simulation/            # محاكيات + fvg
│   ├── features/              # Feature Store
│   ├── models/                # SSL tick/bucket + masking
│   ├── states/                # Regimes / CausalRegimeTracker
│   ├── statistics/            # اختبارات + تصحيح تعدّد
│   ├── research/              # orchestrator + assistant
│   ├── alpha/                 # اكتشاف/فرز الإشارات
│   ├── strategies/            # fail_fvg + fvg_hypothesis search + vp_auction
│   ├── coverage/              # مراقب M9
│   └── validation/            # leakage tests
├── tests/
└── benchmarks/
```

---

## حالة التقدّم

| المحطة | الوصف | الحالة |
|--------|-------|--------|
| 0 | الأساسات والحوكمة | ✅ |
| 1 | استيعاب MBO + دفتر الأوامر | ✅ |
| 2 | طبقة المحاكاة (+ Failed FVG) | ✅ |
| 3 | Feature Store | ✅ |
| 4 | SSL تأسيسي (bucket + tick/event) | ✅ |
| 5 | الحالات الكامنة / Regimes | ✅ |
| 6 | الاختبار الإحصائي | ✅ |
| 7 | مساعد البحث LLM | ✅ |
| 8 | ألفا + الخط الموحّد + بحث فرضيات FVG (WF) | ✅ |
| 9 | مراقب التغطية M9 | ✅ |

---

## المكوّنات (API مختصر)

* `nq.contracts` — `MBO_SCHEMA`, `PRICE_SCALE`, `validate_mbo_frame`
* `nq.ingestion` — `load_mbo_frame`, `iter_mbo_batches`, `normalize_databento_frame`
* `nq.orderbook` — `OrderBook`, `reconstruct`, `check_integrity`
* `nq.features` — Feature Store PIT + **`build_streaming_research_features`** (آلة حالة MBO)
* `nq.models` — `run_ssl_pipeline`, `run_ssl_tick_pipeline`, `build_tick_stream`, `structural_mask_*`, PCA / world model / contrastive
* `nq.research` — **`run_research_pipeline`** (نقطة الدخول)، `ResearchAssistant`, `Evidence`
* `nq.alpha` — `evaluate_signal` / `evaluate_signal_intraday`, `screen_signals`, `discover_alpha_from_features`
* `nq.strategies` — `run_fail_fvg_research` / `search_fail_fvg_hypotheses` / `run_vp_auction_research`
* `nq.coverage` — مقاييس MFIG/CER/PSG/CRS/LORI/QDUF؛ كتل تشمل `failed_fvg` و `volume_profile_auction`
* `nq.states` — `KMeansRegimes`, `CausalRegimeTracker`
* `nq.validation` — `detect_leakage_by_perturbation`, `assert_availability_not_before_event`

---

## قواعد المساهمة

1. لا يُدمج أي PR يخالف المبادئ الحاكمة الأربعة.
2. كل PR: تعريف رياضي عند الحاجة، اختبارات، إثبات منع تسريب، وقياس أداء عند اللزوم.
3. توسيع متداخل في الطبقات الحالية — **لا fork معماري موازٍ**.
4. الإشارات الجديدة تُدمَج في إطار البحث الموحّد (`availability_ts`) وتُفرَز عبر `discover_alpha_from_features`.
