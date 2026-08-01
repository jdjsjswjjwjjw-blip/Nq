# المخطط المعماري التفصيلي (Detailed Architecture)

يمثّل هذا المستند المصدر المرجعي لتدفّق النظام من بيانات MBO الخام
وصولًا إلى إشارات الألفا. كل طبقة تُشتق سببيًا وزمنيًا من الطبقة التي تسبقها،
دون أي تسريب زمني.

```
MBO Raw Data
      │
      ▼
Order Book Reconstruction
      │
      ▼
=============================
Simulation Layer
=============================
      │
      ├── Footprint Simulator
      │       ├── Bid / Ask Volume
      │       ├── Delta
      │       ├── Imbalance
      │       └── Absorption
      │
      ├── Volume Profile Simulator
      │       ├── POC
      │       ├── VAH / VAL
      │       ├── HVN / LVN
      │       └── Value Migration
      │
      ├── Order Flow Simulator
      │       ├── Aggressive Buying
      │       ├── Aggressive Selling
      │       ├── Trade Initiation
      │       └── Liquidity Consumption
      │
      ├── Liquidity Simulator
      │       ├── Resting Orders
      │       ├── Pulling Liquidity
      │       ├── Adding Liquidity
      │       ├── Iceberg Detection (wired into bottom-book / research)
      │       └── Depth Noise Filter (cancel storm / flicker / spoof)
      │
      ├── Depth Lifecycle + Bottom Book (L2–L5)
      │       ├── Bar-close snapshots (shared multi-interval pass)
      │       ├── Intra-bar path metrics (imbalance max/min, L2–L5 drain)
      │       └── Absorption / queue depletion / live iceberg hit
      │
      ├── Auction Market Simulator
      │       ├── Balance
      │       ├── Imbalance
      │       ├── Expansion
      │       └── Pullback Defense
      │
      └── Cross-Market Simulator
              ├── NQ vs MNQ Lead/Lag
              ├── Confirmation Failure
              ├── Divergence
              └── Trader Trap Detection

      │
      ▼
Feature Store
      │
      ▼
=====================================================
Self-Supervised Layer (implemented)
=====================================================

  الحالة الفعلية في الكود (ليست foundation model كامل):

      ├── Causal feature windows (tick / bucket)
      ├── PCAEncoder — تمثيلات منخفضة الأبعاد (z0…zk)
      ├── Walk-forward fit فقط على كتلة التدريب (purged + embargo)
      ├── Masked reconstruction MSE (خفيف)
      ├── Simple contrastive / world-model heads على z
      └── Causal SSL gates & enhancements
              ├── join_asof(backward) للتمثيلات
              ├── عتبات |z| من كمّية ماضية فقط (shift+rolling)
              └── اختيار المرشّحين بـ WF + selection-under-null

  ما هو *مخطط مستقبلي* وليس منفَّذًا كشبكة عميقة هنا:
  contrastive pair mining الكامل، world-model تنبؤي عميق،
  hierarchical multi-scale transformer، memory episodic طويل.

      │
      ▼
Latent Market Representations / Market States
      │
      ▼
=====================================================
Structural Coverage Monitor (Milestone 9)
=====================================================

      ├── MFIG  — Conditional Information Gap (MBO vs Features → Price)
      ├── CER   — Causal Exposure Residual (per simulator block)
      ├── PSG   — Predictive Sufficiency Gap (World Model surprise)
      ├── CRS   — Conditional Reconstruction Sufficiency (masked blocks)
      ├── LORI  — Latent Orphan Regime Index + Transition Surprise
      └── QDUF  — Queue Dynamics Unexplained Fraction

      │
      ▼
Statistical Testing
      │
      ├── Significance Testing (permutation)
      ├── Selection-aware null (re-select under shuffled labels)
      ├── Out-of-Sample Validation (purged walk-forward)
      ├── Regime Validation
      └── Hypothesis Verification
      │
      ▼
LLM Research Assistant
      │
      ├── Pattern Discovery
      ├── Representation Interpretation
      ├── Market Microstructure Reasoning
      ├── Hypothesis Generation
      ├── Research Planning
      └── Automatic Report Writing
      │
      ▼
Research Reports
Trading Hypotheses
Discovered Market Structures
Novel Alpha Signals
```

## مبادئ ملزمة

1. **صفر تسريب زمني** — PIT / causal / purged WF / asof backward / purge ≥ horizon
2. **صرامة كمية** — IC + permutation؛ للشبكات الكبيرة selection-under-null
3. **أداء** — Polars + مرور دفتر موحّد متعدد الفواصل؛ فلترة ضوضاء قبل المسار
4. **MBO فقط** — لا مصادر أسعار خارج عقد MBO
