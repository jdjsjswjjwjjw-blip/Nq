# Phase 1 Graphify Baseline

Date: 2026-07-23

Repository path: `/Users/abdallah/Desktop/Trading/Nq-main`

## CLI Detection

Installed Graphify binary:

```text
/Users/abdallah/anaconda3/bin/graphify
```

Installed Graphify version:

```text
graphify 0.8.27
```

The installed CLI uses the newer command form:

```text
graphify extract <path>
graphify cluster-only <path>
```

It does not support the older/bare form `graphify . --backend openai` for this environment.

## Corpus Scope

Graphify baseline intentionally excluded `docs/deep_audit` so generated audit reports do not become evidence about the original system.

Command-equivalent detector scope:

```text
root = /Users/abdallah/Desktop/Trading/Nq-main
extra_excludes = ["docs/deep_audit"]
```

Detected corpus:

```text
total_files = 127
total_words = 52,925
code_files = 122
document_files = 5
paper_files = 0
image_files = 0
video_files = 0
skipped_sensitive = 0
```

Document files included:

```text
.github/pull_request_template.md
.github/workflows/ci.yml
README.md
docs/architecture.md
docs/data_contracts.md
```

No research PDFs, notebooks, images, or videos were present in the local tree.

## Extraction Commands

Supported full CLI attempt:

```text
graphify extract . --backend claude-cli --mode deep --exclude docs/deep_audit --max-concurrency 1 --api-timeout 600
```

Result:

```text
[graphify extract] found 122 code, 5 docs, 0 papers, 0 images
[graphify extract] AST extraction on 122 code files...
[graphify extract] semantic extraction on 5 files via claude-cli...
[graphify] chunk 1/1 failed: claude -p exited 1: error: unknown option '--no-session-persistence'
[graphify extract] error: all semantic chunks failed for backend 'claude-cli'
```

Root cause: Graphify `0.8.27` invokes the installed Claude CLI with `--no-session-persistence`, which this local Claude CLI (`2.0.0`) does not accept.

Fallback used:

- Graphify detector API with `extra_excludes=["docs/deep_audit"]`.
- Graphify AST extractor for all 122 code files.
- Manual semantic extraction for the 5 document files using Graphify's documented JSON schema.
- Graphify build/cluster/report/html APIs.
- Explicit installed CLI clustering pass:

```text
graphify cluster-only .
```

Cluster-only result:

```text
Graph: 1508 nodes, 3583 edges
Done - 83 communities. GRAPH_REPORT.md, graph.json and graph.html updated.
```

## Graph Outputs

```text
graphify-out/graph.json
graphify-out/GRAPH_REPORT.md
graphify-out/graph.html
graphify-out/GRAPH_HEALTH.txt
graphify-out/.graphify_detect.json
graphify-out/.graphify_ast.json
graphify-out/.graphify_semantic.json
graphify-out/.graphify_extract.json
graphify-out/.graphify_analysis.json
graphify-out/.graphify_labels.json
graphify-out/manifest.json
```

Artifact sizes:

```text
graphify-out/graph.json       1,812,999 bytes
graphify-out/GRAPH_REPORT.md     36,861 bytes
graphify-out/graph.html       1,660,913 bytes
graphify-out/GRAPH_HEALTH.txt     1,995 bytes
```

## Graph Statistics

Final graph:

```text
nodes = 1508
edges = 3583
communities = 83
```

Extraction components:

```text
AST nodes = 1500
AST edges = 4984
semantic nodes = 33
semantic edges = 35
semantic hyperedges = 3
```

Node file types:

```text
code = 1089
rationale = 396
concept = 23
```

Edge confidence:

```text
EXTRACTED = 2327
INFERRED = 1256
AMBIGUOUS = 0
```

Top edge relations:

```text
references = 856
uses = 832
calls = 779
contains = 534
rationale_for = 393
method = 91
imports = 40
imports_from = 30
implements = 13
inherits = 9
conceptually_related_to = 6
```

## Graph Health

Graphify diagnostic:

```text
missing_endpoint_edges = 0
self_loop_edges = 0
dangling_endpoint_edges = 602
exact_duplicate_edges = 484
directed_same_endpoint_collapsed_edges = 787
undirected_same_endpoint_collapsed_edges = 803
relation_variant_groups = 93
context_variant_groups = 198
post_build_graph_type = Graph
```

Interpretation:

- No missing endpoints or self-loops were found.
- `dangling_endpoint_edges` are mostly AST references to external/type nodes that are not modeled as project graph nodes.
- Same-endpoint edge collapse is significant because this graph is undirected after build; do not use raw Graphify edge count as a complete call-count metric.
- The graph is useful for navigation and architecture triage, but call-flow conclusions must still be verified from source code.

## God Nodes

From `graphify-out/GRAPH_REPORT.md`:

```text
1. ResearchReport - 80 edges
2. SSLPipelineResult - 75 edges
3. ResearchAssistant - 74 edges
4. make_stream() - 66 edges
5. Evidence - 61 edges
6. make_generator() - 58 edges
7. PipelineConfig - 58 edges
8. PipelineProgress - 51 edges
9. AlphaDiscovery - 48 edges
10. UnifiedResearchReport - 47 edges
```

## Surprising Connections

From `graphify-out/GRAPH_REPORT.md`:

```text
Simulation Layer --implements--> FeatureStore
Statistical Testing --implements--> ResearchAssistant
Unified MBO To Report Pipeline --conceptually_related_to--> ResearchReport
main() --calls--> run_vp_auction_research()
str --uses--> PipelineConfig
```

Audit note: the last item (`str --uses--> PipelineConfig`) is Graphify structural noise from type relationships, not a meaningful architecture conclusion.

## Communities

The graph contains 83 communities. Major labeled areas include:

- Order Book Depth
- Failed Breakout
- Failed FVG
- Volume Profile Auction
- SSL Models
- Temporal Contracts
- Research Pipeline
- Statistics Validation
- Alpha Signals
- Ingestion
- Cross Market
- Tests
- Core Utilities

The high community count reflects many test and type-helper clusters. Use Graphify for targeted queries, but keep direct source inspection as the authority for correctness.

## Ingestion Problems

Verified problems:

1. CLI semantic extraction through `claude-cli` failed because of an incompatible Claude CLI option.
2. No API-key backed Graphify backend was configured (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, and `GRAPHIFY_BACKEND` were unset).
3. No papers were available to ingest, despite the project request requiring research papers if present.
4. The graph health diagnostic shows material edge collapse in the undirected graph representation.

No parser crash was observed in AST extraction. No sensitive files were skipped.

## Audit Use Policy

Use Graphify as an architecture map only. For leakage, MBO semantics, validation, and execution realism, source code and tests remain the proof layer.
