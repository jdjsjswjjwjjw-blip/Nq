# Graph Report - .  (2026-07-23)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1584 nodes · 3885 edges · 81 communities (66 shown, 15 thin omitted)
- Extraction: 63% EXTRACTED · 37% INFERRED · 0% AMBIGUOUS · INFERRED: 1423 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Order Book Depth|Order Book Depth]]
- [[_COMMUNITY_Statistics Validation|Statistics Validation]]
- [[_COMMUNITY_Failed Breakout|Failed Breakout]]
- [[_COMMUNITY_Tests|Tests]]
- [[_COMMUNITY_Tests 2|Tests 2]]
- [[_COMMUNITY_Failed Breakout 2|Failed Breakout 2]]
- [[_COMMUNITY_Statistics Validation 2|Statistics Validation 2]]
- [[_COMMUNITY_Statistics Validation 3|Statistics Validation 3]]
- [[_COMMUNITY_Research Pipeline|Research Pipeline]]
- [[_COMMUNITY_Order Book Depth 2|Order Book Depth 2]]
- [[_COMMUNITY_Failed FVG|Failed FVG]]
- [[_COMMUNITY_Failed Breakout 3|Failed Breakout 3]]
- [[_COMMUNITY_SSL Models|SSL Models]]
- [[_COMMUNITY_Temporal Contracts|Temporal Contracts]]
- [[_COMMUNITY_SSL Models 2|SSL Models 2]]
- [[_COMMUNITY_Failed FVG 2|Failed FVG 2]]
- [[_COMMUNITY_Research Pipeline 2|Research Pipeline 2]]
- [[_COMMUNITY_SSL Models 3|SSL Models 3]]
- [[_COMMUNITY_Order Book Depth 3|Order Book Depth 3]]
- [[_COMMUNITY_SSL Models 4|SSL Models 4]]
- [[_COMMUNITY_Order Book Depth 4|Order Book Depth 4]]
- [[_COMMUNITY_Order Book Depth 5|Order Book Depth 5]]
- [[_COMMUNITY_Research Pipeline 3|Research Pipeline 3]]
- [[_COMMUNITY_Research Pipeline 4|Research Pipeline 4]]
- [[_COMMUNITY_Order Book Depth 6|Order Book Depth 6]]
- [[_COMMUNITY_Order Book Depth 7|Order Book Depth 7]]
- [[_COMMUNITY_Tests 3|Tests 3]]
- [[_COMMUNITY_Temporal Contracts 2|Temporal Contracts 2]]
- [[_COMMUNITY_Research Pipeline 5|Research Pipeline 5]]
- [[_COMMUNITY_Volume Profile Auction|Volume Profile Auction]]
- [[_COMMUNITY_Order Book Depth 8|Order Book Depth 8]]
- [[_COMMUNITY_Tests 4|Tests 4]]
- [[_COMMUNITY_SSL Models 5|SSL Models 5]]
- [[_COMMUNITY_Order Book Depth 9|Order Book Depth 9]]
- [[_COMMUNITY_Order Book Depth 10|Order Book Depth 10]]
- [[_COMMUNITY_Volume Profile Auction 2|Volume Profile Auction 2]]
- [[_COMMUNITY_SSL Models 6|SSL Models 6]]
- [[_COMMUNITY_SSL Models 7|SSL Models 7]]
- [[_COMMUNITY_Tests 5|Tests 5]]
- [[_COMMUNITY_Tests 6|Tests 6]]
- [[_COMMUNITY_Order Book Depth 11|Order Book Depth 11]]
- [[_COMMUNITY_Temporal Contracts 3|Temporal Contracts 3]]
- [[_COMMUNITY_Order Book Depth 12|Order Book Depth 12]]
- [[_COMMUNITY_SSL Models 8|SSL Models 8]]
- [[_COMMUNITY_Order Book Depth 13|Order Book Depth 13]]
- [[_COMMUNITY_Order Book Depth 14|Order Book Depth 14]]
- [[_COMMUNITY_Tests 7|Tests 7]]
- [[_COMMUNITY_Tests 8|Tests 8]]
- [[_COMMUNITY_Tests 9|Tests 9]]
- [[_COMMUNITY_Order Book Depth 15|Order Book Depth 15]]
- [[_COMMUNITY_Research Pipeline 6|Research Pipeline 6]]
- [[_COMMUNITY_SSL Models 9|SSL Models 9]]
- [[_COMMUNITY_SSL Models 10|SSL Models 10]]
- [[_COMMUNITY_Failed FVG 3|Failed FVG 3]]
- [[_COMMUNITY_Core Utilities|Core Utilities]]
- [[_COMMUNITY_SSL Models 11|SSL Models 11]]
- [[_COMMUNITY_Core Utilities 2|Core Utilities 2]]
- [[_COMMUNITY_Order Book Depth 16|Order Book Depth 16]]
- [[_COMMUNITY_Order Book Depth 17|Order Book Depth 17]]
- [[_COMMUNITY_Research Pipeline 7|Research Pipeline 7]]
- [[_COMMUNITY_Tests 10|Tests 10]]
- [[_COMMUNITY_Core Utilities 3|Core Utilities 3]]
- [[_COMMUNITY_Volume Profile Auction 3|Volume Profile Auction 3]]
- [[_COMMUNITY_Core Utilities 4|Core Utilities 4]]
- [[_COMMUNITY_Research Pipeline 8|Research Pipeline 8]]
- [[_COMMUNITY_Alpha Signals|Alpha Signals]]
- [[_COMMUNITY_Temporal Contracts 4|Temporal Contracts 4]]
- [[_COMMUNITY_Core Utilities 5|Core Utilities 5]]
- [[_COMMUNITY_Core Utilities 6|Core Utilities 6]]
- [[_COMMUNITY_Core Utilities 7|Core Utilities 7]]
- [[_COMMUNITY_Ingestion|Ingestion]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Core Utilities 9|Core Utilities 9]]
- [[_COMMUNITY_Cross Market|Cross Market]]
- [[_COMMUNITY_Order Book Depth 18|Order Book Depth 18]]
- [[_COMMUNITY_Core Utilities 10|Core Utilities 10]]
- [[_COMMUNITY_Core Utilities 11|Core Utilities 11]]
- [[_COMMUNITY_Statistics Validation 4|Statistics Validation 4]]
- [[_COMMUNITY_Core Utilities 12|Core Utilities 12]]
- [[_COMMUNITY_Core Utilities 13|Core Utilities 13]]

## God Nodes (most connected - your core abstractions)
1. `ResearchReport` - 81 edges
2. `SSLPipelineResult` - 79 edges
3. `ResearchReport` - 79 edges
4. `ResearchAssistant` - 76 edges
5. `make_stream()` - 66 edges
6. `Evidence` - 64 edges
7. `PipelineConfig` - 60 edges
8. `make_generator()` - 59 edges
9. `TemporalPolicy` - 58 edges
10. `PipelineProgress` - 52 edges

## Surprising Connections (you probably didn't know these)
- `Simulation Layer` --implements--> `FeatureStore`  [EXTRACTED]
  docs/architecture.md → src/nq/features/store.py
- `FeatureStore` --implements--> `Self-Supervised Foundation Model`  [EXTRACTED]
  src/nq/features/store.py → docs/architecture.md
- `Statistical Testing` --implements--> `ResearchAssistant`  [EXTRACTED]
  docs/architecture.md → src/nq/research/assistant.py
- `Unified MBO To Report Pipeline` --conceptually_related_to--> `ResearchReport`  [INFERRED]
  README.md → src/nq/research/unified.py
- `test_make_generator_reproducible_and_isolated()` --calls--> `make_generator()`  [INFERRED]
  tests/test_core.py → src/nq/core/determinism.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Point-In-Time Research Contract** — readme_zero_temporal_leakage, docs_data_contracts_canonical_temporal_fields, docs_data_contracts_causal_order_event_ts_sequence, docs_data_contracts_availability_ts_contract, github_pull_request_template_governing_principles_checklist [INFERRED 0.85]
- **Unified Strategy Research Paths** — readme_unified_pipeline, readme_failed_fvg_research, readme_failed_breakout_research, readme_volume_profile_auction, readme_ssl_pipeline [INFERRED 0.85]
- **Quality Gate Contract** — readme_quantitative_rigor, github_pull_request_template_quality_gates, github_workflows_ci_quality_job, github_workflows_ci_ruff_mypy_pytest [EXTRACTED 1.00]

## Communities (81 total, 15 thin omitted)

### Community 0 - "Order Book Depth"
Cohesion: 0.17
Nodes (16): _paired_streams(), int, اختبارات مسار بحث Failed FVG عبر الخط الموحّد., test_run_fail_fvg_research_produces_report(), test_run_fail_fvg_research_uses_unified_features(), اختبارات المنسّق الموحّد: خط واحد من MBO إلى التقرير., test_run_research_pipeline_includes_auction_vp_signals(), test_run_research_pipeline_includes_failed_fvg_signal() (+8 more)

### Community 1 - "Statistics Validation"
Cohesion: 0.05
Nodes (62): Alternative, CorrectionMethod, DataFrame, float, str, float, FloatArray, floating (+54 more)

### Community 2 - "Failed Breakout"
Cohesion: 0.20
Nodes (19): مراحل الجلسة intraday (قيم صحيحة للتعلّم الآلي)., SessionPhase, _ensure_flow_columns(), failed_breakout_features(), failed_breakout_from_bars(), محاكي Failed Breakout السببي — من MBO فقط، بتركيز فوليوم.  منطق الإشارة (عند إغل, SMA على إغلاق إطار أعلى؛ متاح عند إغلاق شمعة SMA فقط., بوابة فوليوم سببية حسب وضع الفرضية. (+11 more)

### Community 3 - "Tests"
Cohesion: 0.08
Nodes (45): DataFrame, float, FloatArray, int, IntArray, integer, NDArray, str (+37 more)

### Community 4 - "Tests 2"
Cohesion: 0.14
Nodes (20): _ensure_flags_column(), is_databento_frame(), normalize_databento_frame(), تحويل مخرجات Databento MBO إلى عقد ``MBO_SCHEMA`` القانوني.  يُستدعى تلقائيًا من, يحوّل إطار Databento MBO إلى مخطط ``MBO_SCHEMA``., هل الإطار يحمل أعمدة Databento (وليس العقد القانوني بعد)؟, يُعيد تسمية أعمدة Databento دون إنشاء أعمدة مكرّرة., يحوّل أسعار float (دولار) إلى fixed-point Int64 وفق ``PRICE_SCALE``. (+12 more)

### Community 5 - "Failed Breakout 2"
Cohesion: 0.06
Nodes (96): AlphaDiscovery, مخرجات اكتشاف الألفا: تقييمات مفرزة، إشارات مختارة، وتقرير موثّق., AlphaDiscovery, CoverageReport, LanguageModel, مخرجات خط SSL: مقاييس، تمثيلات كامنة، وتقرير موثّق., SSLPipelineResult, LanguageModel (+88 more)

### Community 6 - "Statistics Validation 2"
Cohesion: 0.07
Nodes (45): align_forward_returns(), AlphaSignal, evaluate_signal(), evaluate_signal_intraday(), إشارات الألفا وتقييمها (Alpha Signals & Evaluation).  الإشارة سببية (تُحسب من ال, يفرز إشارات مرشّحة مع تصحيح التعدّد (BH) لعزل الألفا الحقيقي.      يُعيد إطارًا:, يقيّم إشارة بعوائد أمامية بعد عبور spread وانزلاق intraday., إشارة ألفا سببية مع طوابعها الزمنية ومصدرها (provenance). (+37 more)

### Community 7 - "Statistics Validation 3"
Cohesion: 0.12
Nodes (19): FeatureStore, مخزن الميزات point-in-time (Point-in-Time Feature Store).  المخطط القانوني الطوي, نسخة من كامل بيانات المخزن بالمخطط القانوني., يضيف رصدات ميزات (بالمخطط القانوني) إلى المخزن بعد التحقق., يحوّل مخرَج مُحاكٍ عريضًا إلى المخطط القانوني ثم يضيفه., قائمة الإصدارات المتوفّرة في المخزن., يُعيد أحدث قيمة لكل ميزة كانت متاحة عند ``timestamp`` (point-in-time)., يبني سلسلة لقطات point-in-time عريضة (feature-per-column) مملوءة أماميًا. (+11 more)

### Community 8 - "Research Pipeline"
Cohesion: 0.07
Nodes (45): AssertionError, generic, bool, float, FloatArray, int, NDArray, _causal_cumsum() (+37 more)

### Community 9 - "Order Book Depth 2"
Cohesion: 0.17
Nodes (23): DataFrame, float, int, str, apply_causal_ssl_gate(), default_fvg_grid(), exploratory_screen_candidates(), FvgHypothesisSearchResult (+15 more)

### Community 10 - "Failed FVG"
Cohesion: 0.12
Nodes (32): سياسة زمنية مركزية لمنع التسريب (Temporal Policy).  توحّد إعدادات walk-forward:, إعدادات زمنية ملزمة لمسارات SSL والمراقب والألفا., يقرأ ``[temporal]`` من ملف TOML (افتراضي: ``configs/default.toml``)., يبني سياسة لجلسة تشغيل مع نافذة SSL و``interval_ns`` معروفين., عدد عيّنات التدريب المُزالة قبل الاختبار بسبب تداخل النوافذ., فترة الحظر بنفس وحدات ``times`` (عادة nanoseconds).          * بيانات إنتاج (ns), TemporalPolicy, _times_in_nanoseconds() (+24 more)

### Community 11 - "Failed Breakout 3"
Cohesion: 0.16
Nodes (20): BreakoutHypothesisSpec, classic_breakout_grid(), core_breakout_grid(), default_breakout_grid(), materialize_breakout_hypotheses(), بحث فرضيات Failed Breakout — تركيز فوليوم بلا تسريب.  * شبكة واسعة من أوضاع الفو, الشبكة القديمة (جهد حجم فردي فقط) — للتوافق/المقارنة., الافتراضي = شبكة الفوليوم الواسعة. (+12 more)

### Community 12 - "SSL Models"
Cohesion: 0.13
Nodes (23): _align_markets(), cross_market_features(), _market_windows(), مُحاكي عبر السوقين (Cross-Market Simulator) — NQ ↔ MNQ.  يقارن السوق القائد (NQ،, يشتق ميزات عبر السوقين على شبكة زمنية موحّدة (متاحة عند ``bucket_end``)., يبني سلسلة نافذية لسوق واحد: سعر الإغلاق (mid) والدلتا العدوانية., ارتباط متدحرج سببي (يستخدم الماضي فقط) عبر نافذة بطول ``window``., يحاذي NQ مع MNQ مع تأخير سببي ``latency_ns`` (MNQ عند t−latency). (+15 more)

### Community 13 - "Temporal Contracts"
Cohesion: 0.09
Nodes (28): الحقول الزمنية القانونية (Canonical Temporal Fields).  كل صفّ بيانات في النظام —, أسماء الحقول الزمنية القانونية مُجمّعة للوصول البرمجي المتّسق., TemporalFields, الحتمية وقابلية إعادة الإنتاج (Determinism & Reproducibility).  كل نتيجة علمية ي, تثبّت كل مصادر العشوائية من بذرة واحدة وتُعيد مولّد ``numpy`` حتميًّا.      * ``, seed_everything(), assert_sorted_causal(), is_sorted_causal() (+20 more)

### Community 14 - "SSL Models 2"
Cohesion: 0.19
Nodes (33): CausalStandardScaler, تطبيع سببي (Causal Standardization).  يُلائَم المقياس (mean/std) على بيانات التد, مطبّع قياسي يُلائَم على الماضي (train) ويُطبّق للأمام.      يعمل على المحور الأخ, تقسيم زمني walk-forward مع purge/embargo (Purged Walk-Forward Split).  التقسيم ا, طيّة زمنية واحدة: مؤشّرات التدريب والاختبار., WalkForwardFold, _empty_ssl_result(), _evaluate_ssl_fold() (+25 more)

### Community 15 - "Failed FVG 2"
Cohesion: 0.14
Nodes (29): _as_float(), _as_int(), _base_signal_row(), build_ohlcv_bars(), detect_h1_fvgs(), _empty_signal_frame(), failed_fvg_features(), failed_fvg_from_bars() (+21 more)

### Community 16 - "Research Pipeline 2"
Cohesion: 0.12
Nodes (22): BaseException, _fmt_duration(), _fmt_rate(), iter_with_progress(), PipelineProgress, طباعة تقدّم الخط الموحّد — سطر بسطر من البداية للنهاية.  كل عملية تُطبع فورًا عل, يعلن بدء خطوة جديدة (ويُغلق زمنيًا الخطوة السابقة إن وُجدت)., ملاحظة داخل الخطوة الحالية. (+14 more)

### Community 17 - "SSL Models 3"
Cohesion: 0.10
Nodes (41): BoolArray, IntEnum, MaskedMatrix, mask_matrix(), masked_reconstruction_error(), MaskedMatrix, النمذجة المُقنّعة (Masked Modeling).  تُقنّع نسبة من عناصر المدخلات (masked even, مصفوفة مُقنّعة مع قناعها والقيم الأصلية (الأهداف). (+33 more)

### Community 18 - "Order Book Depth 3"
Cohesion: 0.17
Nodes (14): Encoder, PCAEncoder, مشفّر تمثيلي ذاتي الإشراف (Self-Supervised Representation Encoder).  ``Encoder``, واجهة المشفّر: يُلائَم على التدريب، ويشفّر ويعيد البناء., مشفّر تمثيلي أساسي عبر PCA (SVD)، يُلائَم على الماضي فقط., يتعلّم المحاور الرئيسية من بيانات التدريب (2-D: عيّنات × ميزات)., يشفّر المدخلات إلى الفضاء الكامن (market embeddings)., Protocol (+6 more)

### Community 19 - "SSL Models 4"
Cohesion: 0.12
Nodes (22): build_sequences(), build_tick_sequences(), تقطيع التسلسلات الزمنية السببية (Causal Sequence Windowing).  يبني عيّنات تسلسلي, يبني تسلسلات tick/event مع ``mask_path`` و ``market_phase`` لكل عيّنة., مجموعة تسلسلات سببية.      * ``x``: مصفوفة ``(n_samples, window, n_features)``., يفرد النوافذ إلى مصفوفة ثنائية ``(n_samples, window * n_features)``., يبني ``SequenceDataset`` سببيًا من إطار ميزات مرتّب زمنيًا.      يُفترض أن الإطا, SequenceDataset (+14 more)

### Community 20 - "Order Book Depth 4"
Cohesion: 0.13
Nodes (23): check_integrity(), فحوص سلامة تدفّق MBO (Stream Integrity Checks).  تُقاس هذه الفحوص على البيانات ك, يحسب فحوص السلامة المعتمدة على الإطار (per-instrument).      لا يشمل ``unknown_o, DataFrame, make_stream(), str, يبني إطار MBO صالحًا من قائمة أحداث مختصرة., test_mbo_window_descriptors_nonempty() (+15 more)

### Community 21 - "Order Book Depth 5"
Cohesion: 0.11
Nodes (35): DrawFn, Event, footprint_summary(), يلخّص البصمة لكل نافذة زمنية مع الدلتا التراكمية ونسبة الامتصاص.      الأعمدة: `, DataFrame, int, str, اختبارات latency في cross_market. (+27 more)

### Community 22 - "Research Pipeline 3"
Cohesion: 0.10
Nodes (27): Evidence, يقارن مقياسًا عبر الحالات ويُنتج استنتاجًا مؤسَّسًا على اختبار إحصائي., يقيّم دلالة عوائد إشارة (Sharpe + اختبار تبديل بقلب الإشارة)., يبني فرضية مرتبطة بأدلّة يُقدّمها المستخدم (تُسجَّل في السجلّ)., يبني خطة بحث مُرقّمة حتمية من سؤال وخطوات., يتحقّق من الاستنتاجات ويكتب تقريرًا لا يحوي إلا الموثّق منها., Evidence, EvidenceStore (+19 more)

### Community 23 - "Research Pipeline 4"
Cohesion: 0.09
Nodes (22): 0) التثبيت, 1) الخط الموحّد — من MBO إلى التقرير (`run_week`), 1) منع التسريب الزمني نهائيًا (Zero Temporal Leakage), 2) أمر منفصل: Failed FVG (`run_fail_fvg`), 2) صرامة كمية وعلمية بلا أخطاء, 3) أداء عالٍ لبيانات ضخمة, 3) أمر منفصل: Failed Breakout (`run_fail_breakout`), 4) MBO فقط (+14 more)

### Community 24 - "Order Book Depth 6"
Cohesion: 0.08
Nodes (31): MboSide, جانب دفتر الأوامر الذي يتعلق به الحدث., OrderBook, حالة دفتر الأوامر (Order Book State).  يتتبّع الدفتر لكل جانب (طلب/عرض) الحجم ال, أفضل طلب ``(price, size)`` أو ``None`` إن كان الجانب فارغًا., أفضل طلب ``(price, size)`` أو ``None`` إن كان الجانب فارغًا., أفضل عرض ``(price, size)`` أو ``None`` إن كان الجانب فارغًا., الفارق السعري (best_ask - best_bid) بالنقطة الثابتة، أو ``None``. (+23 more)

### Community 25 - "Order Book Depth 7"
Cohesion: 0.14
Nodes (24): MBO Raw Data, Order Book Reconstruction, Simulation Layer, يعيد بناء المدخلات من تمثيلها الكامن (للنمذجة المُقنّعة/التقييم)., IntegrityReport, تقرير سلامة تدفّق MBO., سليم عندما لا يوجد اختلال ترتيب ولا تسلسل غير رتيب ولا مراجع مجهولة., _empty_tob() (+16 more)

### Community 26 - "Tests 3"
Cohesion: 0.16
Nodes (16): ofi_by_bucket(), order_flow_imbalance(), order_flow_summary(), مُحاكي تدفّق الأوامر (Order Flow Simulator).  يقيس ضغط الشراء/البيع العدواني، با, يلخّص تدفّق الأوامر العدواني لكل نافذة زمنية (متاح عند ``bucket_end``)., يحسب OFI حدثًا بحدث من سلسلة top-of-book ومجموعه التراكمي.      المدخل يجب أن يح, يجمع OFI الحدثي إلى مجموع لكل نافذة زمنية (متاح عند ``bucket_end``)., DataFrame (+8 more)

### Community 27 - "Temporal Contracts 2"
Cohesion: 0.16
Nodes (17): MboEvent, عقد بيانات MBO (Market By Order) — المصدر الوحيد للحقيقة في النظام.  يُصمّم هذا, تمثيل مُكتمل التنميط (fully-typed) لحدث MBO مفرد.      يُستخدم أساسًا في الاختبا, يتحقق من مطابقة إطار Polars لعقد MBO بنيويًا ونقطيًا-زمنيًا.      الفحوص:      *, validate_mbo_frame(), DataFrame, DataFrame, اختبارات عقد بيانات MBO. (+9 more)

### Community 28 - "Research Pipeline 5"
Cohesion: 0.16
Nodes (33): discover_alpha_from_features(), FullResearchResult, اكتشاف الألفا من الميزات وخط البحث الكامل (Alpha Discovery & Pipeline).  يجمع كا, مخرجات الخط البحثي الكامل: تغطية + ألفا., يُفوِّض إلى الخط الموحّد ويُعيد تغطية + ألفا فقط., اختصار للخط الموحّد — يُعيد قناة الألفا فقط (للتوافق مع الاختبارات)., يقيّم ويفرز إشارات مرشّحة من إطار ميزات، ويكتب تقريرًا موثّقًا., run_full_research_pipeline() (+25 more)

### Community 29 - "Volume Profile Auction"
Cohesion: 0.15
Nodes (15): classify_nodes(), developing_value_area(), مُحاكي ملف الحجم (Volume Profile Simulator).  يوزّع الحجم المُنفَّذ على مستويات, يُضيف حجم صفقة إلى المستوى السعري., يُعيد ملف الحجم الحالي كإطار polars مرتّب بالسعر., POC/VAH/VAL من الحالة الحالية., ميزات VP سببية: مسافات POC/VAH/VAL + أعلام القرب/داخل المنطقة., يضيف علمَي ``is_hvn`` و ``is_lvn`` (قمم/قيعان محلية في التوزيع). (+7 more)

### Community 30 - "Order Book Depth 8"
Cohesion: 0.19
Nodes (18): DataType, DepthSnapshot, لقطة عمق معلّق عند نقطة زمنية واحدة., attach_depth_asof(), depth_at_bar_close(), depth_event_series(), _empty_depth_schema(), دورة حياة العمق السببية — دخول / مراقبة / تنفيذ / خروج.  يبني سلسلة لقطات عمق من (+10 more)

### Community 31 - "Tests 4"
Cohesion: 0.12
Nodes (19): add_time_bucket(), extract_trades(), أساس مشترك لطبقة المحاكاة (Simulation Common Foundation).  يوفّر:  * اصطلاح المُ, يضيف أعمدة نافذة زمنية سببية: ``bucket_start``, ``bucket_end``, ``availability_t, يستخرج الصفقات (``action == TRADE``) مع أحجام الشراء/البيع العدوانية.      يضيف, footprint_cells(), _imbalance(), مُحاكي البصمة السعرية (Footprint Simulator).  البصمة تُظهر الحجم العدواني المُنف (+11 more)

### Community 32 - "SSL Models 5"
Cohesion: 0.23
Nodes (12): augment_windows(), info_nce_loss(), _l2_normalize(), _logsumexp_rows(), التعلّم التبايني ذاتي الإشراف (Contrastive Self-Supervised Learning).  يولّد "من, يُنتج منظورًا مُحسَّنًا: تشويش غاوسي + إخفاء عشوائي لبعض الخلايا.      التحسين م, هدف InfoNCE (متماثل) لدفعة من المناظير الإيجابية المتناظرة.      لكل عيّنة ``i``, float (+4 more)

### Community 33 - "Order Book Depth 9"
Cohesion: 0.16
Nodes (34): CausalRegimeTracker, MboAction, أنواع أحداث MBO الذرية على دفتر الأوامر., DevelopingVolumeProfile, _book_depth_features(), _book_row(), build_tick_stream(), _log_size() (+26 more)

### Community 34 - "Order Book Depth 10"
Cohesion: 0.19
Nodes (17): add_session_columns(), _minutes_since_rth_open(), minutes_since_rth_open_from_ns(), _phase_for_time(), مراحل جلسة التداول intraday (Session Phases).  يُصنّف كل ``bucket_end`` / ``avai, يُرجع ``session_phase`` كعدد صحيح من طابع نانوثانية., تاريخ الجلسة (America/New_York) من طابع نانوثانية — سببي point-in-time., يضيف ``session_phase`` و ``minutes_since_rth_open`` و ``session_date``. (+9 more)

### Community 35 - "Volume Profile Auction 2"
Cohesion: 0.18
Nodes (16): auction_signal_frame(), auction_states(), مُحاكي المزاد (Auction Market Simulator).  يستند إلى نظرية المزاد ومنطقة القيمة, إشارات بحثية من Volume Profile + المزاد (توازن/اختلال/تمدّد).      جاهزة للدمج a, يصنّف حالة المزاد لكل نافذة زمنية (متاح عند ``bucket_end``).      الأعمدة تشمل:, DataFrame, float, int (+8 more)

### Community 36 - "SSL Models 6"
Cohesion: 0.14
Nodes (17): يتحقّق من استنتاج مقابل سجلّ الأدلّة (يفرض عدم اختلاق الأدلّة)., يقسّم الاستنتاجات إلى مقبولة ومرفوضة بعد التحقّق., verify_finding(), verify_report(), EvidenceStore, str, اختبارات المحطة 7: مساعد البحث المُؤسَّس على الأدلّة., _StubLM (+9 more)

### Community 37 - "SSL Models 7"
Cohesion: 0.15
Nodes (10): r2_score(), نموذج العالم التنبّئي (Predictive World Model).  يتعلّم خريطة من التمثيل الكامن, معامل التحديد R² (متعدّد المخرجات، مجمّع) — صيغة Campbell–Thompson OOS.      ``b, يلائم على التدريب فقط: ``(XᵀX + αI)⁻¹ Xᵀy`` مع عمود تحيّز., يتنبّأ بالحالة التالية للتمثيلات المُدخلة., float, FloatArray, Campbell OOS R²: ss_tot مقابل متوسط التدريب؛ متوسط الاختبار يحرّف المقياس. (+2 more)

### Community 38 - "Tests 5"
Cohesion: 0.19
Nodes (14): purged_walk_forward_split(), يُنتج طيّات walk-forward متوسّعة مع فترة حظر زمنية.      المعاملات:         time, يُنتج طيّات walk-forward متوسّعة مع فترة حظر زمنية.      المعاملات:         time, int, integer, NDArray, اختبارات التقسيم الزمني walk-forward., test_embargo_purges_adjacent_train() (+6 more)

### Community 39 - "Tests 6"
Cohesion: 0.18
Nodes (13): detect_icebergs(), liquidity_summary(), مُحاكي السيولة (Liquidity Simulator).  يقيس ديناميكية السيولة القائمة (resting), يلخّص إضافة/سحب السيولة لكل نافذة (متاح عند ``bucket_end``)., يكشف الأوامر المخفيّة (icebergs) لكل سعر عبر مسح سببي للأحداث.      يفترض إطارًا, DataFrame, float, int (+5 more)

### Community 40 - "Order Book Depth 11"
Cohesion: 0.05
Nodes (57): commission_rate(), انزلاق وتكاليف تنفيذ intraday مبسّطة (بدون محاكاة طابور)., قيمة الانزلاق المطلقة (نفس وحدات السعر)., عمولة كنسبة مناسبة للضرب في العائد (bps / 10_000)., slippage_amount(), depth_matrices_from_frame(), execution_forward_returns_depth(), _levels_at() (+49 more)

### Community 41 - "Temporal Contracts 3"
Cohesion: 0.17
Nodes (13): Availability Timestamp Contract, Canonical Temporal Fields, Causal Order By event_ts And sequence, Fixed-Point Prices, MBO Lifecycle Actions, MBO Schema, validate_mbo_frame Contract Enforcement, PR Governing Principles Checklist (+5 more)

### Community 42 - "Order Book Depth 12"
Cohesion: 0.19
Nodes (18): bytes, iter_mbo_batches(), load_mbo_frame(), _prepare_frame(), قارئ MBO التدفّقي (Streaming MBO Reader).  المصدر الوحيد للحقيقة هو تدفّق MBO ال, يسلّم بيانات MBO على دفعات سبقية متتابعة بذاكرة ثابتة., يُعالج أسعار null قبل إعادة بناء الدفتر (Clear/None → 0)., يُحمّل بيانات MBO ويتحقق من العقد ويرتّبها سبقيًا.      يقبل إطار Polars مباشرةً (+10 more)

### Community 43 - "SSL Models 8"
Cohesion: 0.17
Nodes (13): bool, DataFrame, float, str, _attach_embeddings(), generate_ssl_enhancement_candidates(), مولّد تعزيزات علمية من SSL/السياق فوق إشارة أساس سببية.  المبدأ: التعلم العميق *, يبني أعمدة تعزيز فوق ``base_columns`` ويعيد (frame, columns, specs).      المرشّ (+5 more)

### Community 44 - "Order Book Depth 13"
Cohesion: 0.23
Nodes (15): _book(), OrderBook, اختبارات حالة دفتر الأوامر (OrderBook)., test_add_aggregates_same_level(), test_best_bid_ask_and_spread(), test_cancel_can_partially_reduce_resting_order(), test_cancel_partial_then_full(), test_cancel_reduces_and_removes_level() (+7 more)

### Community 45 - "Order Book Depth 14"
Cohesion: 0.15
Nodes (15): make_generator(), يُنشئ ``numpy.random.Generator`` حتميًّا دون لمس الحالة العامة.      يُفضّل هذا, test_run_fail_breakout_research_uses_unified_pipeline(), test_search_fail_fvg_hypotheses_smoke(), اختبارات طباعة تقدّم الخط الموحّد., كل إشارة ألفا + كل مقياس M9 يُطبعان أثناء التشغيل التسلسلي., بحث FVG يمرّر progress إلى SSL-tick ويكتب progress.log., test_bucket_ssl_emits_fold_progress() (+7 more)

### Community 46 - "Tests 7"
Cohesion: 0.29
Nodes (11): build_streaming_research_features(), محرّك الميزات اللحظية (Streaming / State Machine) من MBO.  يحدّث الحالة من **كل, يبني إطار البحث من آلة حالة MBO لحظية (بديل الـ batch العريض)., إطار حدث-بحدث من آلة الحالة (متاح عند ``event_ts``)., آخر حالة لحظية داخل كل فاصل.      الحالة تُحدَّث حدثًا بحدث؛ عند أخذ عيّنة بحثية, sample_streaming_to_interval(), streaming_event_features(), DataFrame (+3 more)

### Community 47 - "Tests 8"
Cohesion: 0.29
Nodes (9): build_volume_profile(), يبني ملف الحجم: إجمالي الحجم المُنفَّذ لكل سعر (مرتّبًا تصاعديًا بالسعر)., _profile_stream(), DataFrame, اختبارات مُحاكي ملف الحجم., test_build_volume_profile(), test_classify_nodes_hvn_lvn(), test_value_area_empty_returns_none() (+1 more)

### Community 48 - "Tests 9"
Cohesion: 0.24
Nodes (9): DataFrame, int, random_add_cancel_stream(), مصنع بيانات MBO للاختبارات (test-only MBO builder)., يولّد تدفّق أوامر عشوائيًا (إضافة/إلغاء) صالحًا ومرتّبًا سببيًا., int, إثبات السببية ومنع التسريب الزمني في إعادة البناء.  القاعدة: حالة الدفتر عند الح, test_prefix_final_state_matches_full_run_snapshot() (+1 more)

### Community 49 - "Order Book Depth 15"
Cohesion: 0.29
Nodes (9): اختبارات محاكي Failed Breakout السببي + إصلاح دخول قابل للتنفيذ., تشويش شموع المستقبل لا يغيّر نسب الجهد الماضية., شموع اصطناعية فيها كسر فاشل واضح بعد فترة استقرار., _synthetic_signal_bars(), test_entry_ref_is_close_not_break_level(), test_failed_breakout_availability_at_bar_close(), test_failed_breakout_past_stable_when_future_perturbed(), test_volume_baselines_past_only_stable() (+1 more)

### Community 50 - "Research Pipeline 6"
Cohesion: 0.27
Nodes (11): DataFrame, Path, اختبارات مخزن الميزات point-in-time., test_as_of_returns_only_available_and_latest(), test_ingest_and_len_and_versions(), test_integration_with_real_simulator_output(), test_parquet_roundtrip(), test_point_in_time_join_no_future_leak() (+3 more)

### Community 51 - "SSL Models 9"
Cohesion: 0.31
Nodes (10): DataFrame, Path, اختبارات قارئ MBO التدفّقي., _stream(), test_invalid_batch_size_rejected(), test_load_from_dataframe_sorts_causal(), test_load_validates_contract(), test_roundtrip_arrow() (+2 more)

### Community 52 - "SSL Models 10"
Cohesion: 0.20
Nodes (7): DataFrame, اختبارات المحطة 9: مراقب التغطية البنيوية (Structural Coverage Monitor)., test_coverage_empty_frame(), test_distance_correlation_independent_near_zero(), test_run_all_metrics_on_cross_market_features(), test_run_coverage_pipeline_smoke(), test_run_full_research_pipeline_integration()

### Community 53 - "Failed FVG 3"
Cohesion: 0.25
Nodes (7): أنواع الأحداث (`MboAction`), الأسعار كنقطة ثابتة (Fixed-Point Prices), التحقق (Validation), الجوانب (`MboSide`), الحقول الزمنية القانونية (Canonical Temporal Fields), عقود البيانات (Data Contracts), مخطط MBO (`MBO_SCHEMA`)

### Community 54 - "Core Utilities"
Cohesion: 0.31
Nodes (8): DataFrame, int, اختبارات بحث فرضيات Failed FVG (walk-forward + منع التسريب)., test_default_fvg_grid_nonempty_and_causal_intervals(), test_failed_fvg_baseline_still_works(), test_materialize_hypotheses_past_stable_under_future_perturbation(), test_walk_forward_selects_train_best_candidate_not_first(), _trades_at_prices()

### Community 55 - "SSL Models 11"
Cohesion: 0.23
Nodes (11): اختبارات التعلّم ذاتي الإشراف: المشفّر، الإخفاء، نموذج العالم، والتباين., test_next_state_requires_fit(), test_pca_encoder_is_encoder_protocol(), test_pca_requires_fit_and_2d(), test_run_ssl_tick_pipeline_produces_report(), _paired_mbo(), اختبارات تدفّق tick/event (الأبعاد 1–4)., test_build_tick_stream_has_book_and_vp_columns() (+3 more)

### Community 56 - "Core Utilities 2"
Cohesion: 0.38
Nodes (4): يحسب المتوسّط والانحراف المعياري على بيانات التدريب فقط., يطبّق التطبيع باستخدام إحصاءات التدريب الملائَمة مسبقًا., يلائم على التدريب ثم يطبّق (يُستخدم على طيّة التدريب حصرًا)., FloatArray

### Community 57 - "Order Book Depth 16"
Cohesion: 0.40
Nodes (5): main(), DataFrame, int, قياس إنتاجية إعادة بناء دفتر الأوامر (throughput benchmark).  يُشغّل يدويًا (خار, _synthetic_stream()

### Community 58 - "Order Book Depth 17"
Cohesion: 0.40
Nodes (4): إثبات الالتزام بالمبادئ الحاكمة (Governing Principles Checklist), الوصف (What & Why), بوابات الجودة (Quality Gates), ملاحظات إضافية

### Community 59 - "Research Pipeline 7"
Cohesion: 0.29
Nodes (7): Depth-Based Execution Simulation, Failed Breakout Research Path, Failed FVG Research Path, SSL Tick Or Bucket Pipeline, Streaming Feature State Machine, Unified MBO To Report Pipeline, Volume Profile Auction Research Path

### Community 61 - "Core Utilities 3"
Cohesion: 0.33
Nodes (4): هل الدليل دالّ إحصائيًا عند مستوى ``alpha``؟ (يتطلّب ``pvalue``)., bool, float, object

### Community 62 - "Volume Profile Auction 3"
Cohesion: 0.33
Nodes (4): اختبارات دورة حياة العمق السببية (دخول/مراقبة/تنفيذ/خروج)., test_depth_bar_close_availability_at_bucket_end(), test_depth_event_series_causal_past_stable(), test_pipeline_attaches_depth_columns()

### Community 63 - "Core Utilities 4"
Cohesion: 0.67
Nodes (4): PR Quality Gates, CI Python 3.12, CI Quality Job, Ruff Mypy Pytest Gates

### Community 64 - "Research Pipeline 8"
Cohesion: 0.50
Nodes (3): اختبارات تركيز فرضيات Volume Profile / Auction., test_run_vp_auction_research_produces_report(), test_run_vp_auction_research_uses_unified_features()

### Community 71 - "Community 71"
Cohesion: 0.67
Nodes (3): Self-Supervised Foundation Model, Statistical Testing, Structural Coverage Monitor

## Knowledge Gaps
- **86 isolated node(s):** `int`, `DataFrame`, `FloatArray`, `DataFrame`, `DataFrame` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_stream()` connect `Order Book Depth 4` to `Statistics Validation 2`, `SSL Models`, `Failed FVG 2`, `Order Book Depth 5`, `Tests 3`, `Volume Profile Auction`, `Tests 4`, `Volume Profile Auction 2`, `Tests 6`, `Order Book Depth 12`, `Tests 8`, `Tests 9`, `Order Book Depth 15`, `Research Pipeline 6`, `SSL Models 9`, `SSL Models 10`, `Core Utilities`, `SSL Models 11`, `Volume Profile Auction 3`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `make_generator()` connect `Order Book Depth 14` to `Order Book Depth`, `SSL Models 5`, `Statistics Validation`, `Tests`, `SSL Models 6`, `SSL Models 7`, `Statistics Validation 2`, `Research Pipeline 8`, `Order Book Depth 11`, `SSL Models 8`, `Temporal Contracts`, `SSL Models 3`, `Order Book Depth 3`, `SSL Models 10`, `SSL Models 11`, `Volume Profile Auction 3`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `load_mbo_frame()` connect `Order Book Depth 12` to `Tests 2`, `Failed Breakout 2`, `Order Book Depth 2`, `Failed FVG`, `SSL Models 9`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 76 inferred relationships involving `ResearchReport` (e.g. with `AlphaDiscovery` and `FullResearchResult`) actually correct?**
  _`ResearchReport` has 76 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `SSLPipelineResult` (e.g. with `AlphaDiscovery` and `CoverageReport`) actually correct?**
  _`SSLPipelineResult` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 75 inferred relationships involving `ResearchReport` (e.g. with `AlphaDiscovery` and `FullResearchResult`) actually correct?**
  _`ResearchReport` has 75 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `ResearchAssistant` (e.g. with `AlphaDiscovery` and `FullResearchResult`) actually correct?**
  _`ResearchAssistant` has 65 INFERRED edges - model-reasoned connections that need verification._