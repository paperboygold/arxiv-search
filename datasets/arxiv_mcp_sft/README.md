# arxiv-search MCP tool-use SFT dataset

A small, hand-curated supervised fine-tuning (SFT) dataset that teaches a model
to drive the **arxiv-search MCP server** natively - i.e. to reach for
`search` / `retrieve_paper` / `execute` / `hdrr` and call them with the exact
wire format the server expects, without being told how each time.

**Defining trait:** the user requests are deliberately *messy and
underspecified* - "go find me research on X", "papers on X for Y purpose to Z
effect", "go find the current SoTA on …", "ground yourself against arxiv and …",
"the rlhf kind of thing". The model's job is to **interpret** the request and
translate it into a precise query: domain words -> arXiv category (`vision` ->
`cs.CV`, `speech` -> `eess.AS`, `RL` -> `cs.LG`), "recent/lately/SoTA/right now"
-> `sort: date`, and intent -> abstract field terms. Every tool-call turn
carries a short note that makes that interpretation step explicit, so the model
learns to reason from a vague ask to a structured call.

Two request styles get dedicated coverage:

- **"find the current SoTA / state of the art on X"** -> date-sorted discovery
  that surfaces the leading *recent* work, with the honest caveat that arXiv is
  not a live benchmark leaderboard.
- **"ground yourself against arxiv" / "don't answer from memory, check first"**
  -> the anti-hallucination pattern: search/read the source *before* answering,
  cite the paper, and **correct the user** when the paper disagrees (e.g. that
  attention predates the Transformer; the Transformer was the first
  attention-*only* model).

- **Format:** OpenAI chat fine-tuning JSONL (one JSON object per line, with
  `messages`, `tools`, `parallel_tool_calls`).
- **Interface taught:** the four MCP tools exposed by `crates/native/src/tool.rs`.
- **Composition:** 50 hand-written "gold" conversations, plus ~200 more
  composed by a *controlled generator* inside `build_dataset.py` (a curated
  topic bank x messy phrasing templates x the same tool-call patterns the gold
  set uses). The generator is deterministic/reproducible, and every row - gold
  or composed - passes the same validator. See "Volume & anti-overfitting".

## Files

| File | Purpose |
|---|---|
| `arxiv_mcp_sft.jsonl` | The dataset. 256 conversations (50 hand-written + ~206 composed), ready for SFT. |
| `build_dataset.py` | The 50 gold examples, the controlled generator, plus serialization + validation. Re-run to regenerate the JSONL. |
| `README.md` | This file. |

Regenerate / validate:

```bash
python3 datasets/arxiv_mcp_sft/build_dataset.py
```

The script fails loudly if any example produces invalid JSON, an unknown tool,
a malformed `code` envelope, an orphaned tool result, or a conversation that
doesn't end on an assistant turn.

## The one thing the model must learn: the `code` envelope

Every arxiv-search tool takes a **single string argument named `code`** whose
value is itself a JSON document. So a tool call serializes like this (the
`arguments` string is JSON-encoded twice):

```json
{
  "role": "assistant",
  "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": "{\"code\": \"{\\\"q\\\": \\\"ti:attention AND au:vaswani\\\", \\\"n\\\": 5}\"}"
    }
  }]
}
```

That double nesting is the most common thing agents get wrong against this
server, so most examples reinforce it. The dataset also reinforces:

- preferring `q` over the `query` alias, and `paper_id` over `id`;
- arXiv field syntax in `q`: `ti:` `au:` `abs:` `cat:` with `AND` / `OR` / `ANDNOT`;
- arXiv IDs with an `arxiv:` prefix or `vN` version suffix are normalized server-side.

## Tools covered

| Tool | `code` shape | Returns |
|---|---|---|
| `search` | `{q, n<=50, offset, from, to, cats[], sort}` | `{papers[], total_results, start_index}` |
| `retrieve_paper` | `{paper_id, prune_references, chunk_chars, chunk_overlap, segmentation_k}` | a `PreparedPaper` (`paper`, `source`, `raw_markdown`, `pruned_markdown`, `chunks[]`) |
| `execute` | one `Operation` or an **array** of them: `{op, id, limit, ...}` where `op` ∈ `abstract` \| `download` \| `citations` \| `recs` \| `retrieve` | `{id, op, result}` (or an array thereof) |
| `hdrr` | `{q, limit_docs, limit_chunks}` | `{query, routed_documents[], chunks[]}` |

The set is **discovery-heavy** by design (tool-call mix: `search` 226,
`execute` 117, `retrieve_paper` 35, `hdrr` 2), because the target behavior is
"just go find papers off a poorly-structured instruction". Skills exercised:

- inferring arXiv **category** filters from domain words, and `sort: date` from
  recency cues ("recent", "lately", "SoTA", "right now");
- **SoTA requests**: surfacing the leading recent work while flagging that
  arXiv isn't a live leaderboard;
- **grounding/verification**: searching or reading the source before answering,
  citing it, and correcting the user when it contradicts their assumption;
- field-filtered queries (`ti:` `au:` `abs:` `cat:`, `ANDNOT` exclusions);
- resolving a **title or vague reference to an ID via `search`** instead of
  guessing the ID;
- multi-step chains: `search -> retrieve_paper` (deep read) and
  `search -> execute` (abstract / citations / recommendations);
- batched `execute` arrays and multi-paper synthesis with `hdrr` (plus the
  fallback to `search` + `execute` when the index isn't available);
- empty-result recovery (broaden, then ask which concept matters);
- interface mechanics phrased as casual asks: pagination via `offset`, the
  50-result cap, and `arxiv:`/version-suffix ID normalization;
- calibration: answering a purely conceptual question directly, with an offer
  to fetch the primary source, rather than firing a tool.

## Two fidelity notes worth knowing

1. **Tool outputs are illustrative.** The *call* shapes match the server
   exactly, and the examples use **real arXiv IDs and titles** (so the model
   never learns fake identifiers), but the returned abstracts/chunks/citations
   are synthetic stand-ins for what a live server would fetch. This is fine for
   teaching tool-use behavior; it is **not** a factual knowledge source.

2. **`hdrr` requires the `embedded-db` feature.** In `crates/native`, `hdrr`
   only works when the server is built with `--features embedded-db`;
   otherwise it returns `embedded-db feature not enabled`. The dataset includes
   both the happy path *and* a fallback example where the model degrades
   gracefully to `search` + `execute` when the index isn't available, plus an
   empty-routing example. Keep this in mind for how heavily you want the model
   to lean on `hdrr`.

   `retrieve_paper`'s returned `paper` field is intentionally sparse (title ==
   id, no authors/abstract) because the content path doesn't fetch the Atom
   metadata record - the examples reflect that, so the model learns to use
   `search` or `execute op=abstract` when it needs metadata.

## Volume & anti-overfitting

The 50 hand-written rows come first; the rest are composed by a deterministic
generator (`build_dataset.py`, bottom section) that crosses a curated bank of
~30 real-paper topics with the messy phrasing templates and the proven
tool-call patterns. Because templated bulk is exactly where a fine-tune
overfits, the generator is built to stay diverse:

- every boilerplate string (intros, SoTA caveats, grounding notes, closers,
  offers, headers) is drawn from a **rotated pool of 4-8 variants**, so no
  single sentence dominates - in the current build no answer sentence repeats
  in more than ~3% of rows (worst case 8/256), and user turns are near-unique;
- each `(topic, intent)` pair appears at most twice, with different phrasing,
  paper subset, and parameters;
- `TARGET_GENERATED` (default 200) controls volume. Raise it for more data, but
  watch the repetition - adding more topics/phrasings keeps it healthier than
  simply turning the dial up.

Treat the composed rows as breadth (phrasing/topic robustness) and the 50 gold
rows as the quality anchor for richer multi-step behavior.

## Using it for fine-tuning

The JSONL is in OpenAI's chat tool-calling format, which is consumed directly
by OpenAI fine-tuning and by most open-source SFT stacks (Axolotl,
LLaMA-Factory, Unsloth, Together, etc.).

- Assistant tool-call turns carry a short natural-language `content` note
  alongside `tool_calls` (narration the model learns to emit before acting). A
  few strict validators want `content` to be null/empty when `tool_calls` are
  present; if your trainer is one of them, blank the `content` field on
  tool-call turns before training.
- 256 rows is a solid starter set, not a finished one. For a durable "native"
  capability, mix this with a larger/real corpus, add topics and phrasings, and
  consider mixing in some genuinely non-arxiv conversations so the model doesn't
  learn to reach for these tools on every prompt.

## Source of truth

The tool names, parameters, aliases, and response shapes mirror
`crates/native/src/tool.rs` and `crates/core/src/{arxiv,content,paper,semantic_scholar}.rs`.
If those change, update `TOOLS`, the builders, and the examples here to match.
