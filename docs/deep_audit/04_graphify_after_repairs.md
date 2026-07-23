# Phase 21 Graphify After Repairs

Date: 2026-07-23

Repository path: `/Users/abdallah/Desktop/Trading/Nq-main`

## Commands

Graphify version:

```text
graphify 0.8.27
```

The full semantic CLI path remains blocked by the local Claude CLI incompatibility
documented in `01_graphify_baseline.md`. After code repairs, the supported code-only
update path was used:

```text
graphify update . --force
graphify cluster-only .
```

Because `graphify update` does not expose the same `--exclude docs/deep_audit` option
as `graphify extract`, a local `.graphifyignore` was added:

```text
docs/deep_audit/
```

An intermediate update had included generated audit reports in `graph.json`. Those
generated-report nodes were removed from the generated graph artifact, then clustering
was rerun:

```text
removed_nodes=53 nodes 1637->1584 links 3934->3885 hyperedges 3->3
graphify cluster-only .
```

Verification:

```text
jq '[.nodes[] | select((.source_file // .file // "") | contains("docs/deep_audit"))] | length' graphify-out/graph.json
0
```

## Before vs After

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Nodes | 1,508 | 1,584 | +76 |
| Edges | 3,583 | 3,885 | +302 |
| Communities | 83 | 81 | -2 |
| Hyperedges | 3 | 3 | 0 |

After report line:

```text
1584 nodes · 3885 edges · 81 communities (66 shown, 15 thin omitted)
```

Artifact sizes after repair:

```text
graphify-out/graph.json       1,954,868 bytes
graphify-out/GRAPH_REPORT.md     38,238 bytes
graphify-out/graph.html       1,783,792 bytes
```

## Health Diagnostic

After diagnostic:

```text
node_count = 1584
raw_edge_count = 3885
missing_endpoint_edges = 0
dangling_endpoint_edges = 0
self_loop_edges = 0
exact_duplicate_edges = 0
directed_same_endpoint_collapsed_edges = 0
undirected_same_endpoint_collapsed_edges = 0
relation_variant_groups = 0
source_file_variant_groups = 0
source_location_variant_groups = 0
context_variant_groups = 0
post_build_graph_type = Graph
post_build_edge_count = 3885
```

Interpretation:

- The after graph is structurally cleaner than the Phase 1 baseline diagnostic.
- Generated audit reports are excluded from the final after graph.
- The after graph is primarily a code-topology update plus retained original semantic
  docs. It is not a fresh full LLM semantic extraction because the local Graphify
  `claude-cli` semantic path still fails.

## Architecture Impact

No new orphan architecture was observed from the repairs. Expected changes appeared in
the graph around:

- `OrderBook.apply`
- `purged_walk_forward_split`
- `walk_forward_select_hypotheses`
- corresponding regression tests

The main communities remain:

- Order Book Depth
- Temporal Contracts
- Failed FVG
- Failed Breakout
- SSL Models
- Research Pipeline
- Statistics Validation
- Tests

## Remaining Graph Caveat

This comparison is valid for code topology and local graph health. It is not a full
semantic-docs re-ingestion because no compatible LLM backend was available through the
installed Graphify CLI.

