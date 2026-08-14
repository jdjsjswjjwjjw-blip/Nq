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
   → (اختياري) Auction Behavior Phase‑1 — فهم سلوك المزاد بلا قرارات تداول
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
| `.zst` | يحتاج `pip install -e ".[data]"`؛ مع `--max-rows` فكّه مرة إلى صيغة عمودية لضمان الذاكرة المحدودة |
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
| `run_fail_fvg --search --understand` | نفس البحث + طبقات فهم كمية (OOS) | نعم — تشخيص بعد الاختيار فقط | + `understanding/` |
| `run_fail_breakout` | Failed Breakout (فوليوم + عمق دفتر) | نعم — أمر تشغيل منفصل فقط | كاملة (SSL‖M9‖ألفا) |
| `run_fail_breakout --search` | شبكة فوليوم (~144) / نواة+SSL | نعم — walk-forward بلا تسريب | تقرير بحث + folds + screen |
| `run_fail_breakout --search --compose-hold` | تركيب volume-first × hold داخل الكسر | نعم — فوليوم يولّد · بنية تؤكّد | نفس مخرجات البحث |
| `run_fail_breakout --search --understand` | نفس البحث + طبقات فهم كمية (OOS) | نعم — تشخيص بعد الاختيار فقط | + `understanding/` |
| `run_fail_breakout_days` | نفس FB على شرائح يومية متوازية | نعم — كل يوم كون سببي مغلق؛ لا اختيار عبر الأيام | `manifest.json` + مجلد/يوم |
| `run_symbolic_search` | DEAP + gplearn (معادلات بلا `if`) | نعم — WF فوق ميزات الخط · يحتاج `nq[gp]` | programs.json + folds + signals |
| `run_vp_auction` + `configs/vp_auction.toml` | VP + توازن/اختلال + تضليل + هولد + R:R | نعم — مسار واحد متصل داخل الاستراتيجية | كاملة + edge_* |
| `run_vp_auction_days` | نفس VP على أيام متوازية (شهر) | نعم — كل يوم كون مغلق؛ stream=snapshots | `manifest.json` + مجلد/يوم |
| `run_liquidity_edge` | غلاف توافق → نفس `run_vp_auction` | نعم — ليس تشعّبًا منفصلًا | نفس مخرجات VP |
| API `nq.auction_behavior` | فهم سلوك المزاد Phase‑1 (احتمالات بلا تداول) | نعم — طبقة فوق VP السببي؛ لا تحل محل التنفيذ | probabilities + events + validation |

> لو عايز الكل شغّال → `run_week`.  
> لو عايز فرضية واحدة للفرز → الأمر المنفصل المناسب (نفس المعالجة والمخرجات).  
> لو عايز **أفضل تايم فريم/إعدادات** لـ FVG → `run_fail_fvg --search`.  
> لو عايز Failed Breakout (فوليوم + عمق) → `run_fail_breakout` أو `--search`.  
> لو عايز يولّف استراتيجيات volume-first + hold داخل الكسر → `--search --compose-hold`.  
> لو عايز **تفسير كمي بعد الاختيار** (لماذا فازت الإشارة؟) → أضف `--understand` مع `--search`.  
> لو عايز **معادلات رمزية بلا if** → `pip install 'nq[gp]'` ثم `run_symbolic_search`.  
> لو عايز **VP كامل متصل** (إشارة + تضليل + تنفيذ R:R) → `run_vp_auction` (الافتراضي).

---

### 0) معالجة صحيحة ضمن القدرة (قبل الحجم)

**الكثرة لا تعوّض البروتوكول.** القدرة المحدودة → عيّنة مضبوطة + اختيار lean + دلالة على OOS فقط.

| قاعدة | التطبيق في المشروع |
|-------|---------------------|
| عيّنة محدودة | `--max-rows 500000` أو `configs/lean.toml`؛ القارئ يحتفظ بأقدم N سببيًا بذاكرة محدودة ولا يجسّد الملف كاملًا |
| شبكة مضغوطة | FB: نواة+تعزيزات (افتراضي) · FVG: `core_fvg_grid` (~16) |
| فلاتر lean | كمّية عمق/تعزيز `0.7` فقط (عطّل بـ `--no-lean-filters`) |
| ترتيب رخيص | walk-forward يرتّب بـ Spearman IC فقط (بلا تبديل لكل مرشّح) |
| دلالة مرة واحدة | temporal block permutation على **OOS المجمّع** (`--n-permutations` افتراضي 100) |
| استكشاف اختياري | `--exploratory` مغلق افتراضيًا (ليس أساس `best`) |
| فهم بعد الاختيار | `--understand` بـ 50 تبديلًا فقط — لا يغيّر الاختيار |

> **حدّ الصفوف ليس جلسة:** إذا قطع `--max-rows` جلسة CME من المنتصف في مسار VP،
> تُوسَم الجولة تلقائيًا `exploratory_only=true` ولا تُعد إثباتًا للإيدج. للتحقق
> الرسمي استخدم ملف يوم/جلسة مكتملة أو اجعل الحد يقع على انتقال `session_date`.

```bash
# ملف lean للخط الموحّد
python scripts/run_week.py --config configs/lean.toml --nq /path/to/nq.parquet

# بحث FB/FVG ضمن القدرة
python scripts/run_fail_breakout.py --nq ... --search --max-rows 500000
python scripts/run_fail_fvg.py --nq ... --search --max-rows 500000
```

الثوابت: `nq.research.capacity`.

---


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

**تقدّم التشغيل (stderr + `progress.log`):** مسار خطي — كل خطوة وكل عملية داخلها تُطبع فورًا
(`→` للخطوات، `-` للعمليات، `…` لنسبة التقدّم + سرعة + ETA كل ~1 ث داخل الحلقات الطويلة).
يغطي بالتفصيل: تحميل MBO، الميزات، إعادة بناء الدفتر (`reconstruct`)، مسح العمق (ساعة البحث + FB 30m)،
مسار أحداث العمق + asof + توليد `__depth__*`، FVG/Auction/VP شموعًا بشموع، تجسيد فرضيات FB/FVG،
بناء نوافذ SSL (مع نبض كل طيّة)، tick_stream، ألفا (عمق + تبديلات)، مقاييس M9 الستة مع نبض التبديل،
والشاشة الاستكشافية. الكتابة thread-safe؛ نبض كل قناة مستقل حتى لا يكتم SSL نبض M9.
عند `parallel_coverage=true` تظهر بادئة `[SSL]` / `[M9]`.
الافتراضي تسلسلي (`parallel_coverage=false`) حتى لا يبدو اللوج «دائرة» متداخلة.
عطّل بـ `--quiet` أو `[run] quiet = true`.

**الإعدادات:** `configs/research.toml`

| قسم | أهم المفاتيح |
|-----|----------------|
| `[data]` | `nq_path`, `mnq_path`, `cross_market_mode` (`nq_only` / `dual`), `max_rows` |
| `[ssl]` | `mode` = `tick` \| `bucket`, `window`, `n_components` |
| `[features]` | `mode` = `streaming` (افتراضي) \| `batch` |
| `[signals]` | `include_failed_fvg`, `include_auction_vp`, قائمة `columns` للفرز |
| `[run]` | `quiet` = تعطيل التقدّم · `parallel_coverage` = SSL‖M9 (افتراضي `false`) |
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

# نفس البحث + طبقات فهم كمية بعد الاختيار (OOS فقط — لا تغيّر best)
python scripts/run_fail_fvg.py \
  --nq /path/to/nq.parquet \
  --search --understand \
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
| منع التسريب | نبضة تطابقية لـ fail_*؛ asof خلفي للحالة المستمرة (VP/عمق)؛ اختيار الإعداد على **train فقط**؛ قياس OOS على **test** (purged walk-forward) |
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
| `understanding/` | مع `--understand`: ablation / regime / attribution / stability / depth CF / SSL link (OOS فقط) |

> **`--understand`**: طبقات فهم كمية **بعد** اختيار walk-forward. لا تغيّر `best_oos_spec`
> ولا تضيف مرشّحين — كل المقاييس على طيّات الاختبار (purged) فقط. التفاصيل في القسم 2b.

> في الخط العام: `include_failed_fvg = true` يُلحق `fail_fvg` **مع** باقي الإشارات.  
> `run_fail_fvg` = جولة فرز مركّزة؛ `--search` = بحث إعدادات/تايم فريم فوق نفس المحرك.

---

### 2b) طبقات الفهم الكمي (`--understand`)

تشخيص **بعد** `--search` فقط. الهدف: تفسير كمي/رياضي للإشارة المختارة — **ليس** بحث ألفا جديد.

| طبقة | المقياس | قيد التسريب |
|------|---------|-------------|
| Ablation | Δ IC بعد نزع `__ssl` / `__depth__*` / `__enh__*` + BH داخل العائلة | OOS test folds فقط |
| Regime map | Spearman IC حسب `session_phase` | OOS فقط |
| Gate attribution | pass-rate + \|selected\|↔\|base\| معاصر | OOS فقط (ليس عائد أمامي) |
| Temporal stability | mean/std/`positive_rate` لـ `test_ic` من الطيّات | الطيّات نفسها purged |
| Depth counterfactual | IC مع/بدون عمق + permutation على تسميات OOS | خلط labels داخل OOS فقط |
| SSL state link | \|z\| ↔ \|signal\| معاصر | ارتباط حالة — **ليس** forward alpha |

**مخرجات** تحت `--output/understanding/`:

| ملف | المحتوى |
|-----|---------|
| `report.md` | ملخص النتائج + ملاحظات القيود |
| `ablation.parquet` / `regimes.parquet` / `attribution.parquet` | جداول الطبقات |
| `stability.parquet` | `test_ic` لكل طيّة |
| `depth_counterfactual.parquet` / `ssl_state_link.parquet` | عند انطباق الطبقة |
| `summary.parquet` | ملخص عددي سريع |

الوحدة: `nq.research.understanding` (`run_understanding_layers`, `write_understanding_outputs`).

---

### 3) أمر منفصل: Failed Breakout (`run_fail_breakout`)

كسر فاشل (Failed Breakout) من MBO → شموع سببية **بتركيز فوليوم + عمق دفتر**:

* جهد سعري + فرضيات فوليوم كثيرة (فردي / تراكمي / دلتا / جهد×نتيجة).
* كل متوسطات الفوليوم **ماضية فقط** (`shift(1)`); `availability_ts = bucket_end`.
* **إصلاح تسريب الدخول:** الإشارة عند إغلاق الشمعة؛ `fb_entry_ref = close`؛  
  `fb_break_level` تحليلي فقط — التقييم عبر مسار الألفا.
* **التنخيل:** walk-forward purged + تعزيزات SSL/سياق/فوليوم (ليس إعادة كتابة القاعدة).
* **عمق لا يُطمس:** لقطة سلم L1–L5 عند `bucket_end` للدخول؛ أعمدة `depth_*` +
  سيولة VAH/VAL/trail للمراقبة؛ التنفيذ والخروج بمسح السيولة الظاهرة
  (`execution_forward_returns_depth`) بلا اختلاق عمق.
* **تسريع مسار العمق (آمن):** تحديث الدفتر لكل الأحداث + قياس `depth_path_*`
  عند open/close الشمعة فقط؛ إن كانت إشارات الأساس كلها صفر يُتخطّى المسار.

| وضع فوليوم (`vol_mode`) | المعنى |
|-------------------------|--------|
| `bar` | حجم الشمعة / متوسط حجم ماضٍ |
| `cum` | حجم تراكمي لآخر N شموع / متوسط تراكمي ماضٍ |
| `delta` | \|Δ\| عالٍ + اتفاق عدوان الشراء/البيع مع فشل الكسر |
| `effort_result` | جهد حجم عالٍ + امتصاص عالٍ (حجم كبير / مدى صغير) = جهد بلا نتيجة |

| أولوية (`priority`) | المعنى |
|---------------------|--------|
| `structure_first` | الكسر الفاشل أولًا ثم بوابة فوليوم (الافتراضي التاريخي) |
| `volume_first` | **حدث الفوليوم يولّد** المرشّح ثم بنية الكسر تؤكّد |

| hold عند الدخول (`hold_mode`) | المعنى (سببي — بلا look-ahead للخروج) |
|-------------------------------|----------------------------------------|
| `none` | بلا شرط hold إضافي |
| `persist` | جهد حجم الشمعة السابقة مرتفع أيضًا (بناء فوليوم) |
| `absorption` | امتصاص عالٍ = حجم يُمسَك بلا نتيجة سعرية |
| `imbalance` | اختلال تدفّق يتفق مع فشل الكسر |

أفق الـ hold التنفيذي عند التقييم = `--horizon` (شموع ساعة البحث).

| مرحلة | العمق |
|--------|--------|
| دخول | `depth_*` عند إغلاق الشمعة + `fb_depth_at_break` عند مستوى الكسر |
| مراقبة | `depth_cum_*` / trail / VAH–VAL liq على إطار البحث |
| تنفيذ | مسح سلم ظاهر VWAP (كمية ≤ السيولة المعروضة) |
| خروج | نفس المسح على لقطة `t+horizon` (تسمية فقط؛ الميزات past-only) |

```bash
python scripts/run_fail_breakout.py \
  --nq /path/to/nq.parquet \
  --max-rows 500000 \
  --output data/runs/fail_breakout

# نواة فوليوم + تعزيزات SSL (تنخيل walk-forward)
python scripts/run_fail_breakout.py \
  --nq /path/to/nq.parquet \
  --search \
  --max-rows 500000 \
  --output data/runs/fail_breakout_search

# تركيب volume-first × hold داخل الكسر (الفوليوم يولّد · البنية تؤكّد)
python scripts/run_fail_breakout.py \
  --nq /path/to/nq.parquet \
  --search --compose-hold \
  --horizon 2 \
  --max-rows 500000 \
  --output data/runs/fail_breakout_hold

# شبكة تركيب كاملة بلا تعزيز SSL
python scripts/run_fail_breakout.py \
  --nq /path/to/nq.parquet \
  --search --compose-hold --no-enhance \
  --max-rows 500000

# نفس البحث + فهم كمي بعد الاختيار
python scripts/run_fail_breakout.py \
  --nq /path/to/nq.parquet \
  --search --understand \
  --max-rows 500000 \
  --output data/runs/fail_breakout_search

# شبكة فوليوم كاملة (~144 فرضية) بلا مولّد تعزيزات
python scripts/run_fail_breakout.py \
  --nq /path/to/nq.parquet \
  --search --no-enhance \
  --max-rows 500000

# أيام متوازية (بيانات يوم-بيوم) — كل يوم كون سببي مغلق؛ لا اختيار عبر الأيام
python scripts/run_fail_breakout_days.py \
  --nq-glob '/data/nq/*.parquet' \
  --mnq-dir /data/mnq \
  --jobs 30 \
  --threads-per-worker 2 \
  --search --compose-hold \
  --n-splits 3 \
  --n-permutations 100 \
  --output data/runs/fail_breakout_month
```

> **Day-parallel والمبادئ الأربعة:** التوازي على **مستوى الملف اليومي** فقط
> (`ProcessPool`). داخل كل يوم يبقى نفس المحرّك السببي (نبضة fail_* + asof للحالة
> المستمرة + purged WF). كاش OHLCV/مسح/tick_stream = **داخل اليوم فقط** — ممنوع
> مشاركة عبر الأيام. `manifest.json` / `summary.md` وصفيان — **لا** يختاران فرضية
> موحّدة عبر الشهر.
مع `--search` (افتراضي): SSL يولّد **مرشّحي تعزيز** (`ssl_abs_q*`, `ssl_sign_*`, `ctx_*` بما فيها فلاتر فوليوم)
فوق نواة Failed Breakout، ثم walk-forward يختار الأفضل خارج العينة.

مع `--search --compose-hold`: المحرّك **يولّف** فرضيات `volume_first × hold_mode × vol_mode`
(نواة مع تعزيز / شبكة كاملة مع `--no-enhance`) ثم نفس التنخيل OOS.

`--understand` (اختياري مع `--search`): طبقات فهم كمية بعد الاختيار — انظر **§2b**. لا تغيّر `best_oos_spec`.

| عمود | المعنى |
|------|--------|
| `fail_breakout` | `+1` LONG / `−1` SHORT / `0` |
| `fb_entry_ref` | مرجع دخول قابل للتنفيذ (إغلاق شمعة الإشارة) |
| `fb_break_level` | مستوى الكسر الفاشل (ليس سعر ملء) |
| `fb_effort_volume_ratio` | جهد حجم فردي |
| `fb_effort_result_ratio` | جهد مقابل نتيجة (امتصاص نسبي) |
| `fb_bar_volume` / `fb_cum_volume` | حجم فردي / تراكمي |
| `fb_delta` / `fb_cum_delta` | دلتا / دلتا تراكمية |
| `fb_absorption` / `fb_vol_imbalance` | امتصاص / اختلال حجم |
| `fb_depth_at_break` | سيولة ظاهرة عند مستوى الكسر؛ `NaN` = لا تطابق / لا دفتر (ليس صفرًا) |
| `depth_cum_*` / `depth_*_sz_k` | سلم عمق L1–L5 للمراقبة والتنفيذ/الخروج |
| `*__enh__*` | تعزيزات SSL/سياق/فوليوم مرشّحة (عند `--search`) |

---

### 4) Volume Profile المتصل (`run_vp_auction`)

أمر تشغيل لفرضيات الملف الحجمي — **مسار واحد**: إشارة مزاد → فلتر تضليل → هولد → دخول/خروج R:R.  
`run_liquidity_edge` غلاف توافق فقط (يفوّض لنفس الدالة).

| طبقة | المعنى |
|------|--------|
| `vp_balance` / `vp_imbalance` / … | إشارات المزاد على VA التراكمي |
| فلتر تضليل | إسقاط أوامر وهمية قبل بناء الميزات (TRADE لا تُمس) |
| `entry_gate` / `market_true` | هولد سيولة حقيقية + حكم صدق السوق |
| `vp_*_gated` | نفس إشارة VP × بوابة الهولد |
| `edge_*` | وقف/هدف من `decision_VAL/VAH/POC` المكتملة فقط؛ اختيار داخلي وملخص نهائي من outer holdout المختوم |

`absorb` و`look_fail` والـFSM ومسافات VP التنفيذية كلها تقرأ حدود
`decision_vah/decision_poc/decision_val` السابقة المكتملة. تبقى `vah/poc/val`
الحالية وصفية فقط ولا تدخل قرار البرميل أو مستويات الوقف/الهدف.

```bash
# المسار الكامل المتصل (افتراضي سريع: batch — بدون tick_stream الثقيل)
python scripts/run_vp_auction.py \
  --nq /path/to/nq.parquet \
  --max-rows 500000 \
  --min-oos-rr 2.5 \
  --output data/runs/vp_auction

# دخان يوم واحد: قلّل تبديلات M9/ألفا (2000 الافتراضي ثقيل على mfig)
python scripts/run_vp_auction.py \
  --nq /path/to/one_day.parquet \
  --max-rows 500000 \
  --n-permutations 200 \
  --min-oos-rr 2.5 \
  --output data/runs/vp_one_day_smoke

# IC/WF فقط بدون طبقة التنفيذ
python scripts/run_vp_auction.py --nq ... --no-execution

# مسار streaming كامل (snapshots كل interval — ليس صفًا لكل حدث)
python scripts/run_vp_auction.py --nq ... --streaming

# شهر: يوم-بيوم متوازٍ (20 يوم × 4 خيوط ≈ يستغل ~80 كور)
python scripts/run_vp_auction_days.py \
  --nq-dir /data/mnq_days \
  --jobs 20 \
  --threads-per-worker 4 \
  --output data/runs/vp_month

# أو عبر run_week + إعداد مركّز
python scripts/run_week.py \
  --config configs/vp_auction.toml \
  --nq /path/to/nq.parquet \
  --nq-only \
  --max-rows 500000
```

> في الخط العام (`configs/research.toml`): `include_auction_vp = true` يُلحق إشارات VP **مع** باقي الإشارات، بدون استبدالها.

---

### 4b) فهم سلوك المزاد — المرحلة 1 (`nq.auction_behavior`)

طبقة **منفصلة** فوق نفس البنية السببية لـ VP/المزاد: هدفها فهم احتمالي لسلوك المزاد
(توازن / كسر حقيقي·كاذب / ريتست / توسّع / عودة للقيمة) **بدون** توصيات دخول/خروج
وبدون RL وبدون إعادة تصميم دفتر الأوامر.

| طبقة | الوظيفة | قيد التسريب |
|------|---------|-------------|
| 1–2 | حالات مزاد + ملخص جلسات السيولة (آسيا/لندن/نيويورك) | `decision_*` متأخرة فقط |
| 3 | سيناريو لندن مقابل قيمة آسيا المكتملة (وصفي) | حدود آسيا = `decision_*` سابقة |
| 4 | نية أوردرفلو (درجات تضليل) | **درجات فقط** — لا `filter_deceptive_liquidity` |
| 5–6 | دمج إشارات VP/FSM + أحداث سلوكية | نبضات من أعمدة سببية جاهزة |
| 7 | ذاكرة سوقية | `shift(k)` خلفي فقط (`k≥1`) |
| 8–9 | جودة إشارة + متجه حالة | بلا تحجيم صفقة |
| 10–11 | احتمالات تجريبية + تحقق | purged walk‑forward · بلا مخرجات `edge_*` |

```python
from nq.auction_behavior import BehaviorConfig, run_auction_behavior_analysis

result = run_auction_behavior_analysis(
    mbo_frame,
    config=BehaviorConfig(include_deceptive_scores=True),  # درجات فقط، بلا حذف
)
print(result.probabilities)          # p_true_break / p_false_break / …
assert result.validation.ok
assert result.diagnostics["deceptive_filtered"] is False
assert "entry_gate" not in result.blended.columns
```

> هذه الطبقة **لا تستبدل** `run_vp_auction` (مسار التنفيذ/R:R). هي مسار فهم سابق
> لقرارات التداول، فوق نفس `decision_*` و`join_asof(..., backward)`.

---

### 4c) من بايثون (API)

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
       # عمق: VAH/VAL/trail + stream_*_liq (لا طمس الدفتر)
  → [features.mode=batch] cross_market_features   # نوافذ مجمّعة (اختياري)
  → filter_depth_noise (سببي) ثم depth_at_bar_close (L1–L5) asof خلفي
  → bottom_book L2–L5 / iceberg asof خلفي         # دخول/مراقبة/تنفيذ/خروج
  → pulse-join failed_fvg_features  # fail_fvg, effort_*  (تطابق availability_ts فقط)
  → asof-join auction_signal_frame # vp_balance, vp_imbalance, … (حالة مستمرة)
  → pulse-join failed_breakout_features
       # fail_breakout + فوليوم (bar/cum/delta/effort_result)
       # + fb_depth_at_break (NaN = لا تطابق مستوى؛ ليس sticky asof)
  → ┌ run_ssl_tick_pipeline  أو  run_ssl_pipeline
    ├ run_coverage_on_features     # كتل: streaming + order_book_depth + FB
    └ discover_alpha_from_features # IC؛ intraday = depth-walk إن وُجد سلم
  → build_unified_report
```

**تايم فريمات Failed Breakout**

| طبقة | الإطار |
|------|--------|
| إشارة FB (افتراضي) | 30 دقيقة |
| شبكة البحث | 15م و 30م |
| فلتر SMA | 60 دقيقة |
| ساعة التقييم / ألفا / عمق البحث | `interval_ns` (غالبًا 1 ثانية) |

**تدفق `--search` (FVG / Breakout hypothesis search):**

```
MBO
  → شبكة فرضيات (تايم فريم + عتبات / أوضاع فوليوم)
  → asof على ساعة التقييم            # خلفي فقط
  → مسار أحداث العمق داخل الشمعة     # depth_path_* عند bucket_end فقط
  → مرشّحو __depth__*                 # إشارة × بوابة ماضية (كمّية/اتفاق إشارة)
  → بوابة/تعزيزات SSL اختيارية       # z* asof + كمّية ماضية + سياق/فوليوم
  → walk-forward purged              # اختيار على train → IC على test
  → تقرير + fold_selections + screen
```

فلتر العمق **لا يغيّر** قاعدة FB/FVG؛ يضيف مرشّحين فقط. عطّله بـ `--no-depth-filter`.

**SSL**

| `ssl.mode` | المدخل | الإخفاء |
|------------|--------|---------|
| `tick` (افتراضي) | MBO event + دفتر حي + VP + عمق VAH/VAL/trail | هيكلي (`masking_structural`) |
| `bucket` | أعمدة الإشارة المجمّعة | عشوائي (`mask_matrix`) |

**عمق الدفتر (مبدأ: لا طمس)**

| مرحلة | المصدر | `availability_ts` |
|--------|--------|-------------------|
| دخول | `depth_at_bar_close` + `fb_depth_at_break` | `bucket_end` |
| مسار أحداث (فلتر فرضيات) | `depth_event_path_at_bar_close` → `__depth__*` | `bucket_end` |
| مراقبة | tick_stream / streaming `depth_*` + trail | `event_ts` ثم عيّنة `bucket_end` |
| تنفيذ | `execution_forward_returns_depth` (مسح L1–L5) | لقطة عند `t` |
| خروج | نفس المسح على لقطة `t+horizon` | تسمية فقط (ليس ميزة) |

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
2. **الذاكرة / القدرة:** لا تحمّل شهرًا كاملاً (~300M صف). للبحث التفاعلي استخدم
   `--max-rows 500000` أو `configs/lean.toml` أو شريحة يومية.
3. **NQ فقط:** `--nq-only` أو `cross_market_mode = "nq_only"` في TOML.
   `build_tick_stream` يبني مسارًا أحاديًا عند `nq is mnq` (بدون مضاعفة الأحداث).
   بحث FB/FVG يتخطّى SSL تلقائيًا إذا كانت إشارات الأساس أقل من عتبة WF.
4. **أسعار Databento float:** تُحوَّل تلقائيًا إلى fixed-point عبر `PRICE_SCALE`.
5. **أسعار null (Clear):** تُعالَج في `sanitize_mbo_frame` قبل إعادة بناء الدفتر.
6. **بحث ثقيل عمدًا:** `--no-enhance` / `--full-grid` / `--no-lean-filters` / `--exploratory`
   للحالات الخبيرة فقط — الافتراضي capacity-correct.

---

## هيكل المستودع

```
Nq/
├── README.md
├── configs/
│   ├── default.toml
│   ├── research.toml          # الخط العام — كل الإشارات معًا
│   ├── lean.toml              # قدرة محدودة: max_rows=500k · perms=200
│   ├── fail_fvg.toml          # أمر FVG منفصل (فرز مركّز، مخرجات كاملة)
│   ├── fail_breakout.toml     # أمر FB منفصل (فوليوم + عمق)
│   └── vp_auction.toml        # أمر VP منفصل (فرز مركّز، مخرجات كاملة)
├── scripts/
│   ├── run_week.py            # الخط الموحّد MBO → تقرير
│   ├── run_fail_fvg.py        # FVG منفصل (+ --search / --understand)
│   ├── run_fail_breakout.py   # FB منفصل (+ --search فوليوم/SSL / --understand)
│   ├── run_fail_breakout_days.py  # FB يوم-بيوم متوازٍ (ProcessPool · عزل سببي)
│   ├── run_symbolic_search.py # DEAP + gplearn (معادلات بلا if · nq[gp])
│   └── run_vp_auction.py      # VP متصل: إشارة + تضليل + هولد + R:R
│   └── run_vp_auction_days.py # VP يوم-بيوم متوازٍ (شهر)
│   └── run_liquidity_edge.py  # غلاف توافق → نفس vp_auction
├── docs/
│   ├── architecture.md
│   └── data_contracts.md
├── data/                      # raw / runs (محلي)
├── src/nq/
│   ├── contracts/             # MBO schema + زمني
│   ├── core/                  # حتمية، جلسة، سياسة زمنية
│   ├── ingestion/             # قارئ + Databento
│   ├── orderbook/             # إعادة بناء الدفتر + DepthSnapshot / walk VWAP
│   ├── simulation/            # محاكيات + fvg + breakout + depth_lifecycle
│   ├── features/              # Feature Store + streaming (عمق كامل)
│   ├── models/                # SSL tick/bucket + masking
│   ├── states/                # Regimes / CausalRegimeTracker
│   ├── statistics/            # اختبارات + تصحيح تعدّد
│   ├── research/              # orchestrator + assistant + progress + understanding
│   ├── alpha/                 # اكتشاف/فرز (intraday أو depth-walk)
│   ├── strategies/            # fail_fvg + fail_breakout + vp_auction + search + depth filter
│   ├── auction_behavior/      # مرحلة‑1: فهم سلوك المزاد (احتمالات بلا تداول)
│   ├── coverage/              # مراقب M9 (+ كتلة order_book_depth)
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
| 8 | ألفا + الخط الموحّد + بحث فرضيات FVG/FB (فوليوم+عمق) | ✅ |
| 9 | مراقب التغطية M9 | ✅ |
| — | عمق سببي دخول/مراقبة/تنفيذ/خروج (L1–L5) | ✅ |
| — | فرضيات فوليوم FB (bar/cum/delta/effort_result) | ✅ |
| — | تركيب volume-first + hold داخل الكسر (`--compose-hold`) | ✅ |
| — | فلتر دخول مسار أحداث العمق (`__depth__*`) | ✅ |
| — | طبقات فهم كمية OOS (`--understand`) | ✅ |
| — | فهم سلوك المزاد Phase‑1 (`nq.auction_behavior`) | ✅ |

---

## المكوّنات (API مختصر)

* `nq.contracts` — `MBO_SCHEMA`, `PRICE_SCALE`, `validate_mbo_frame`
* `nq.ingestion` — `load_mbo_frame`, `iter_mbo_batches`, `normalize_databento_frame`
* `nq.orderbook` — `OrderBook` (+ `snapshot`/`top_n`/`cum_depth`), `DepthSnapshot`, `walk_buy_vwap`/`walk_sell_vwap`, `reconstruct`
* `nq.features` — Feature Store PIT + **`build_streaming_research_features`** (آلة حالة + عمق كامل)
* `nq.models` — `run_ssl_pipeline`, `run_ssl_tick_pipeline`, `build_tick_stream`, `structural_mask_*`
* `nq.research` — **`run_research_pipeline`**, `PipelineProgress`, `ResearchAssistant`؛
  فهم كمي: `nq.research.understanding` (`run_understanding_layers`)
* `nq.alpha` — `evaluate_signal` / `evaluate_signal_intraday`؛ depth-walk تلقائي إن وُجد سلم
* `nq.simulation` — `failed_breakout_*`, `depth_at_bar_close`, `depth_event_path_at_bar_close`,
  `execution_forward_returns_depth`
* `nq.strategies` — `run_fail_fvg_research` / `search_fail_fvg_hypotheses` /
  `run_fail_breakout_research` / `search_fail_breakout_hypotheses` /
  `generate_depth_entry_candidates` / `run_vp_auction_research`
* `nq.auction_behavior` — `run_auction_behavior_analysis` (Phase‑1: احتمالات سلوك بلا تداول)
* `nq.coverage` — MFIG/CER/PSG/CRS/LORI/QDUF؛ كتل `failed_breakout` + `order_book_depth` + VP
* `nq.validation` — `detect_leakage_by_perturbation`, `assert_availability_not_before_event`

---

## قواعد المساهمة

1. لا يُدمج أي PR يخالف المبادئ الحاكمة الأربعة.
2. كل PR: تعريف رياضي عند الحاجة، اختبارات، إثبات منع تسريب، وقياس أداء عند اللزوم.
3. توسيع متداخل في الطبقات الحالية — **لا fork معماري موازٍ**.
4. الإشارات الجديدة تُدمَج في إطار البحث الموحّد (`availability_ts`) وتُفرَز عبر `discover_alpha_from_features`.
