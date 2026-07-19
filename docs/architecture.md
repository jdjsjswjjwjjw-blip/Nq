# المخطط المعماري التفصيلي (Detailed Architecture)

يمثّل هذا المستند المصدر المرجعي الكامل لتدفّق النظام من بيانات MBO الخام
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
      │       └── Iceberg Detection
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
Self-Supervised Foundation Model
=====================================================

      ├── Temporal Representation Learning
      │       ├── Event Sequence Encoding
      │       ├── Queue Evolution
      │       ├── Time Dependency Learning
      │       └── Event Transition Modeling
      │
      ├── Hierarchical Representation Learning
      │       ├── Event Level
      │       ├── Price Level
      │       ├── Order Book Level
      │       ├── Auction Level
      │       └── Session Level
      │
      ├── Multi-Scale Representation Learning
      │       ├── Microseconds
      │       ├── Milliseconds
      │       ├── Seconds
      │       ├── Minutes
      │       └── Multi-Horizon Context
      │
      ├── Contrastive Self-Supervised Learning
      │       ├── Positive Pair Mining
      │       ├── Negative Pair Mining
      │       ├── Regime Discrimination
      │       └── Representation Consistency
      │
      ├── Masked Modeling
      │       ├── Masked Event Prediction
      │       ├── Masked Order Reconstruction
      │       ├── Masked Queue Recovery
      │       └── Missing State Completion
      │
      ├── World Model / Predictive Modeling
      │       ├── Next State Prediction
      │       ├── Future Liquidity Prediction
      │       ├── Future Queue Evolution
      │       ├── Price Impact Prediction
      │       └── Counterfactual Simulation
      │
      ├── Memory Mechanism
      │       ├── Long Context Memory
      │       ├── Session Memory
      │       ├── Episodic Memory
      │       └── Persistent Market Memory
      │
      ├── Latent State Learning
      │       ├── Market Embeddings
      │       ├── Regime Embeddings
      │       ├── Liquidity Embeddings
      │       └── Auction Embeddings
      │
      ├── Cross-Market Representation Learning
      │       ├── NQ ↔ MNQ
      │       ├── Cross-Asset Correlation
      │       ├── Lead/Lag Embeddings
      │       └── Shared Latent Space
      │
      └── Causal Representation Learning
              ├── Causal Discovery
              ├── Intervention Modeling
              ├── Structural Dependencies
              └── Cause vs Correlation

      │
      ▼
Latent Market Representations / Market States
      │
      ▼
Statistical Testing
      │
      ├── Significance Testing
      ├── Robustness Testing
      ├── Out-of-Sample Validation
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
      ├── Explain Hidden Behaviors
      ├── Compare Market Regimes
      ├── Research Planning
      └── Automatic Report Writing
      │
      ▼
Research Reports
Trading Hypotheses
Discovered Market Structures
Novel Alpha Signals
```
