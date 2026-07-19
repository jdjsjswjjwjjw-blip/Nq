# Nq — نظام بحثي كمّي لبنية السوق الدقيقة (Market Microstructure Research Engine)

نظام بحثي كمّي متكامل مبني بالكامل على بيانات **MBO (Market By Order)** لعقود **NQ / MNQ**،
يمتد من إعادة بناء دفتر الأوامر (Order Book Reconstruction) وصولًا إلى نموذج تأسيسي
ذاتي الإشراف (Self-Supervised Foundation Model) ومساعد بحثي قائم على LLM لاكتشاف
هياكل السوق وإشارات الألفا الجديدة.

> الهدف النهائي: تحويل التدفق الخام للأوامر إلى **تمثيلات كامنة لحالات السوق** قابلة
> للاختبار الإحصائي وللتفسير البحثي، بدقّة علمية صارمة وأداء عالٍ على بيانات ضخمة.

---

## المبادئ الحاكمة (Non‑Negotiable Principles)

هذه المبادئ الأربعة **ملزِمة في كل سطر كود وكل محطة**، ولا يُقبل أي عمل يخالفها:

### 1) منع التسريب الزمني نهائيًا (Zero Temporal Leakage)
- كل حساب يعتمد **فقط** على المعلومات المتاحة حتى اللحظة `t` (Point‑in‑Time / Causal Only).
- ممنوع أي استخدام لبيانات مستقبلية بشكل مباشر أو غير مباشر (No look‑ahead, no future peeking).
- كل ميزة (Feature) تحمل **طابعًا زمنيًا صريحًا** (`event_ts`, `ingest_ts`, `valid_from`).
- التقسيم للتدريب/التحقق يكون زمنيًا صارمًا (Walk‑Forward / Purged + Embargo) لا عشوائيًا.
- أي تطبيع/تسوية (Normalization, Scaling) يُحسب من الماضي فقط ويُطبّق للأمام (Fit‑on‑past).
- **قاعدة**: كل PR يجب أن يثبت خلوّه من التسريب عبر اختبار زمني (Leakage Test) قبل الدمج.

### 2) صرامة كمية وعلمية بلا أخطاء (Quantitative & Scientific Rigor)
- كل مؤشر/محاكاة معرَّف **رياضيًا** بصيغة واضحة قبل كتابة الكود.
- كل مخرج مصحوب بـ **اختبارات وحدة (Unit Tests)** واختبارات خصائص (Property Tests).
- كل نتيجة بحثية تمرّ عبر **اختبار دلالة إحصائية + متانة + تحقق خارج العينة**.
- قابلية إعادة الإنتاج الكاملة (Deterministic seeds, versioned data, pinned deps).

### 3) أداء عالٍ لمعالجة بيانات ضخمة (High‑Performance / Big Data)
- أسلوب برمجي متجهي (Vectorized / Columnar) لا حلقات بايثون على المسارات الساخنة.
- الاعتماد على أعمدة بصيغة Arrow/Parquet ومعالجة تدفّقية (Streaming) لا تحميل كامل بالذاكرة.
- استخدام Polars / DuckDB / Numba / Rust‑bindings عند الحاجة للأداء الحرج.
- كل مكوّن يُقاس بـ **Benchmark** (Throughput, Latency, Memory) قبل الاعتماد.

### 4) MBO فقط (MBO‑Only Data Source)
- المصدر الوحيد للحقيقة هو تدفّق **MBO** الخام؛ لا يُسمح بمصادر مجمّعة (OHLC/Aggregated) كمدخل أساسي.
- كل الطبقات الأعلى (Footprint, Volume Profile, Order Flow...) تُشتق **حصريًا** من إعادة بناء دفتر الأوامر من MBO.

---

## المخطط المعماري العام (Architecture Overview)

```
MBO Raw Data
   → Order Book Reconstruction
   → Simulation Layer (Footprint | Volume Profile | Order Flow | Liquidity | Auction | Cross-Market)
   → Feature Store
   → Self-Supervised Foundation Model
   → Latent Market Representations / Market States
   → Statistical Testing
   → LLM Research Assistant
   → Research Reports | Trading Hypotheses | Novel Alpha Signals
```

راجع المخطط التفصيلي في `docs/architecture.md`.

---

## تقسيم العمل إلى محطات (Roadmap / Milestones)

كل محطة لها: **هدف**، **مخرجات (Deliverables)**، و**معايير قبول (Definition of Done)**.
لا يُبدأ بأي محطة قبل استيفاء معايير قبول المحطة التي تسبقها.

### المحطة 0 — الأساسات والحوكمة (Foundations & Governance)
- **الهدف**: بنية مستودع نظيفة، عقود بيانات، معايير كود، وبوابات جودة.
- **المخرجات**:
  - هيكل المجلدات (`data/`, `src/`, `tests/`, `docs/`, `benchmarks/`, `configs/`).
  - عقود البيانات (Data Contracts / Schemas) لتدفّق MBO والحقول الزمنية.
  - إعداد الأدوات: إدارة الحزم، linting، type‑checking، CI، اختبارات، تثبيت البذور.
  - أداة **Leakage Test** عامة تُستخدم في كل المحطات.
- **DoD**: CI أخضر، فحص أنواع صارم، قالب PR يفرض إثبات منع التسريب.

### المحطة 1 — استيعاب MBO وإعادة بناء دفتر الأوامر (Ingestion & Order Book Reconstruction)
- **الهدف**: قارئ MBO عالي الأداء يعيد بناء دفتر أوامر دقيق حدثًا بحدث (event‑by‑event).
- **المخرجات**:
  - مُحلِّل (Parser) تدفّقي لأحداث MBO (Add / Modify / Cancel / Trade / Fill).
  - محرك إعادة بناء دفتر الأوامر بترتيب سببي صارم مع طوابير الأسعار (Price‑level Queues).
  - تحقق من السلامة (Sequence gaps, out‑of‑order, integrity checks).
- **DoD**: تطابق حالة الدفتر مع لقطات مرجعية، Benchmark للـ throughput، صفر تسريب زمني.

### المحطة 2 — طبقة المحاكاة (Simulation Layer)
تُشتق كلها من دفتر الأوامر المُعاد بناؤه، بترتيب زمني سببي:
- **Footprint Simulator**: Bid/Ask Volume، Delta، Imbalance، Absorption.
- **Volume Profile Simulator**: POC، VAH/VAL، HVN/LVN، Value Migration.
- **Order Flow Simulator**: Aggressive Buying/Selling، Trade Initiation، Liquidity Consumption.
- **Liquidity Simulator**: Resting Orders، Pulling/Adding Liquidity، Iceberg Detection.
- **Auction Market Simulator**: Balance، Imbalance، Expansion، Pullback Defense.
- **Cross-Market Simulator**: NQ↔MNQ Lead/Lag، Confirmation Failure، Divergence، Trader Trap.
- **DoD**: تعريف رياضي موثّق لكل مؤشر + اختبارات وحدة + تحقق من عدم التسريب لكل مُحاكٍ.

### المحطة 3 — مخزن الميزات (Feature Store)
- **الهدف**: تخزين ميزات point‑in‑time قابلة لإعادة الإنتاج مع أطر زمنية صريحة.
- **المخرجات**: مخطط ميزات موحّد، إصدارات (Versioning)، واسترجاع زمني دقيق (Time‑travel).
- **DoD**: استرجاع أي ميزة كما كانت في زمن `t` تمامًا، بدون أي قيمة مستقبلية.

### المحطة 4 — النموذج التأسيسي ذاتي الإشراف (Self‑Supervised Foundation Model)
مكوّنات التعلّم التمثيلي:
- Temporal / Hierarchical / Multi‑Scale Representation Learning.
- Contrastive Learning، Masked Modeling، World Model (Predictive).
- Memory Mechanism، Latent State Learning، Cross‑Market، Causal Representation Learning.
- **DoD**: مسارات تدريب/تقييم زمنية (Walk‑Forward)، منع تسريب في بناء الأزواج/الأقنعة، مقاييس تمثيل موثّقة.

### المحطة 5 — التمثيلات الكامنة / حالات السوق (Latent Representations / Market States)
- **الهدف**: استخراج تمثيلات كامنة وحالات/أنظمة سوقية (Regimes) قابلة للاستخدام البحثي.
- **DoD**: تمثيلات مستقرة، قابلة للتفسير، ومُرفقة بطوابع زمنية سليمة.

### المحطة 6 — الاختبار الإحصائي (Statistical Testing)
- Significance Testing، Robustness Testing، Out‑of‑Sample Validation، Regime Validation، Hypothesis Verification.
- **DoD**: كل فرضية تمرّ ببروتوكول إحصائي موثّق مع تصحيح التعدد (Multiple‑testing correction).

### المحطة 7 — مساعد البحث LLM (LLM Research Assistant)
- Pattern Discovery، Representation Interpretation، Microstructure Reasoning، Hypothesis Generation،
  Explain Hidden Behaviors، Compare Regimes، Research Planning، Automatic Report Writing.
- **DoD**: كل ادعاء من المساعد مرتبط بأدلة كمية قابلة للتتبع (No hallucinated evidence).

### المحطة 8 — المخرجات النهائية (Outputs)
- Research Reports، Trading Hypotheses، Discovered Market Structures، Novel Alpha Signals.
- **DoD**: كل مخرج مدعوم بنتائج إحصائية وقابل لإعادة الإنتاج من البيانات الخام.

---

## هيكل المستودع المقترح (Proposed Repository Layout)

```
Nq/
├── README.md
├── docs/
│   └── architecture.md          # المخطط المعماري الكامل
├── configs/                     # إعدادات التشغيل والتجارب (versioned)
├── data/                        # عقود البيانات + بيانات معالجة (Parquet/Arrow)
├── src/
│   ├── ingestion/               # المحطة 1: قراءة MBO
│   ├── orderbook/               # المحطة 1: إعادة بناء الدفتر
│   ├── simulation/              # المحطة 2: المحاكيات
│   ├── features/                # المحطة 3: Feature Store
│   ├── models/                  # المحطة 4: النموذج التأسيسي
│   ├── representations/         # المحطة 5: الحالات الكامنة
│   ├── statistics/              # المحطة 6: الاختبار الإحصائي
│   └── research_assistant/      # المحطة 7: مساعد LLM
├── tests/                       # اختبارات وحدة + خصائص + منع تسريب
└── benchmarks/                  # قياسات الأداء
```

---

## حالة التقدّم (Progress)

| المحطة | الوصف | الحالة |
|--------|-------|--------|
| 0 | الأساسات والحوكمة | ✅ مكتملة |
| 1 | استيعاب MBO + إعادة بناء الدفتر | ✅ مكتملة |
| 2 | طبقة المحاكاة | ✅ مكتملة |
| 3 | Feature Store | ✅ مكتملة |
| 4 | النموذج التأسيسي | ✅ مكتملة (أساس + بنية تحتية) |
| 5 | التمثيلات الكامنة | ⏳ |
| 6 | الاختبار الإحصائي | ⏳ |
| 7 | مساعد البحث LLM | ⏳ |
| 8 | المخرجات النهائية | ⏳ |

---

## التطوير المحلي وبوابات الجودة (Local Dev & Quality Gates)

```bash
pip install -e ".[dev]"     # التثبيت مع أدوات التطوير

ruff check src tests         # فحص الأسلوب
ruff format --check src tests
mypy                         # فحص أنواع صارم (strict)
pytest --cov                 # اختبارات الوحدة + التسريب
```

بوابات الجودة نفسها تُنفَّذ آليًا في CI (`.github/workflows/ci.yml`) على كل PR.

المكوّنات المتاحة بعد المحطة 0:

* `nq.contracts` — عقد MBO والحقول الزمنية (`MBO_SCHEMA`, `validate_mbo_frame`).
* `nq.core` — الحتمية (`seed_everything`) والترتيب السببي (`sort_causal`).
* `nq.validation` — **أداة اختبار التسريب الزمني** (`detect_leakage_by_perturbation`, ...).
* `nq.ingestion` — قارئ MBO تدفّقي (`load_mbo_frame`, `iter_mbo_batches`).
* `nq.orderbook` — إعادة بناء الدفتر (`OrderBook`, `reconstruct`) وفحوص السلامة (`check_integrity`).
* `nq.simulation` — طبقة المحاكاة الكاملة:
  * `footprint_cells` / `footprint_summary` — البصمة السعرية (Delta، Imbalance، Absorption).
  * `build_volume_profile` / `value_area` / `classify_nodes` / `developing_value_area` — ملف الحجم (POC، VAH/VAL، HVN/LVN، هجرة القيمة).
  * `order_flow_summary` / `order_flow_imbalance` / `ofi_by_bucket` — تدفّق الأوامر و OFI.
  * `liquidity_summary` / `detect_icebergs` — السيولة وكشف الآيسبرغ.
  * `auction_states` — حالات المزاد (توازن/تمدّد/دفاع ارتداد).
  * `cross_market_features` — **NQ↔MNQ** (Lead/Lag، تباعد، فشل تأكيد، مصيدة المتداولين).
* `nq.features` — **مخزن الميزات point-in-time** (`FeatureStore`): توحيد مخرجات المحاكيات، استرجاع `as_of`، دمج `point_in_time_join`، إصدارات، وحفظ/قراءة Parquet.
* `nq.models` — **النموذج التأسيسي ذاتي الإشراف** (أساس numpy، خالٍ من التسريب):
  * `purged_walk_forward_split` — تقسيم زمني مع purge/embargo.
  * `build_sequences` / `CausalStandardScaler` — تقطيع سببي وتطبيع fit-on-train.
  * `PCAEncoder` (خلف `Encoder` Protocol) — تعلّم تمثيلي كامن.
  * `mask_matrix` / `masked_reconstruction_error` — النمذجة المُقنّعة.
  * `NextStatePredictor` — نموذج العالم التنبّئي (next-state).
  * `augment_windows` / `info_nce_loss` — التعلّم التبايني (contrastive).

## قواعد المساهمة (Contribution Rules)

1. لا يُدمج أي PR يخالف أحد المبادئ الحاكمة الأربعة.
2. كل PR يتضمّن: تعريفًا رياضيًا، اختبارات، إثبات منع تسريب، وقياس أداء عند الحاجة.
3. الالتزام بالتقسيم إلى محطات؛ لا قفز لمحطة قبل استيفاء معايير قبول سابقتها.
