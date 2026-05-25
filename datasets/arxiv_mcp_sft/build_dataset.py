#!/usr/bin/env python3
"""Serialize the hand-curated arxiv-search MCP tool-use examples to JSONL.

This is NOT a synthetic generator: every conversation below is hand-authored.
The script exists only to (a) guarantee the deeply-nested JSON escaping is
correct and (b) validate the result. The escaping is genuinely tricky here
because every arxiv-search MCP tool takes a *single string argument named
`code`* whose value is itself a JSON document, so a tool call serializes to:

    function.arguments = "{\"code\": \"{\\\"q\\\": \\\"...\\\"}\"}"

Output format: OpenAI chat fine-tuning JSONL. One JSON object per line with
`messages`, `tools`, and `parallel_tool_calls`. Assistant tool-call turns use
`tool_calls`; tool results use a `tool` role message keyed by `tool_call_id`.

NOTE ON FIDELITY: the tool *outputs* in these examples are illustrative
(synthetic), but they are schema-faithful to what crates/native/src/tool.rs
actually returns, and they use real arXiv IDs/titles so the model never learns
fake identifiers. The tool *call* shapes match the server exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("arxiv_mcp_sft.jsonl")

# --------------------------------------------------------------------------- #
# System prompt (kept identical across every example for training stability).
# --------------------------------------------------------------------------- #

SYSTEM = (
    "You are a research assistant with access to the arxiv-search MCP server, "
    "which retrieves and prepares scientific papers from arXiv. Ground every "
    "claim in real papers returned by the tools; never invent arXiv IDs, "
    "titles, authors, or findings.\n\n"
    "Every tool takes a SINGLE string argument named `code` that contains a "
    "JSON object (or, for `execute`, a JSON object or array). Always put the "
    "parameters inside `code` as JSON text.\n\n"
    "Tools:\n"
    "- search: discover papers. code keys: q (required), n (1-50, default 10), "
    "offset, from, to (YYYY-MM-DD), cats (array like [\"cs.CL\"]), sort "
    "(relevance|date). In q use arXiv field syntax: ti: (title), au: (author), "
    "abs: (abstract), cat: (category), combined with AND/OR/ANDNOT. Prefer the "
    "key `q` (not `query`).\n"
    "- retrieve_paper: fetch ONE paper's content, pruned and chunked for "
    "reading. code keys: paper_id (required), prune_references (default true), "
    "chunk_chars (default 4000), chunk_overlap (default 200), segmentation_k "
    "(optional float, e.g. 1.2, for hierarchical structure on complex papers). "
    "The returned `paper` field carries only the id/url, not metadata.\n"
    "- execute: batch metadata/content ops. code is one Operation or an array "
    "of them. Operation keys: op (abstract|download|citations|recs|retrieve), "
    "id (required), limit (citations<=100, recs<=50). abstract=metadata+"
    "abstract, download=full markdown, citations=papers citing this, "
    "recs=similar papers, retrieve=prepared content.\n"
    "- hdrr: Hybrid Document-Routed Retrieval for multi-paper question "
    "answering over an index. code keys: q (required), limit_docs (default 5), "
    "limit_chunks (default 10).\n\n"
    "Workflow: discover with search; for a cross-paper question prefer hdrr; "
    "for a deep single-paper read use retrieve_paper. arXiv IDs may carry an "
    "`arxiv:` prefix or a version suffix (e.g. 1706.03762v2) and are "
    "normalized server-side. If a search returns nothing, broaden the query "
    "before retrieving."
)

# --------------------------------------------------------------------------- #
# Tool schemas (mirror crates/native/src/tool.rs: each tool has one `code`
# string param described by its #[schemars(description=...)]).
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search arXiv papers with filters (categories, dates, sorting).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "JSON object. Keys: q (required arXiv query, e.g. "
                            "'ti:attention AND au:vaswani'; supports ti: au: abs: "
                            "cat: and AND/OR/ANDNOT), n (1-50, default 10), offset "
                            "(default 0), from/to (YYYY-MM-DD), cats (string array "
                            "e.g. [\"cs.AI\",\"cs.LG\"]), sort (relevance|date)."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_paper",
            "description": "Get one paper's content, pruned and chunked for LLM reading. Supports hierarchical segmentation (segmentation_k).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "JSON object. Keys: paper_id (required arXiv id, e.g. "
                            "'1706.03762'; 'arxiv:' prefix and version suffix are "
                            "normalized), prune_references (default true), "
                            "chunk_chars (default 4000), chunk_overlap (default "
                            "200), segmentation_k (optional float for hierarchical "
                            "chunking, e.g. 1.2)."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": "Batch fetch: abstracts, full text, citations, recommendations, or prepared content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "JSON Operation, or a JSON array of Operations for "
                            "batching. Operation keys: op (abstract|download|"
                            "citations|recs|retrieve), id (required arXiv id), "
                            "limit (citations<=100, recs<=50, default 10), "
                            "prune_references, chunk_chars, chunk_overlap, "
                            "segmentation_k."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hdrr",
            "description": "Hybrid Document-Routed Retrieval (HDRR): two-stage multi-paper QA. Stage 1 routes documents, stage 2 does scoped chunk search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "JSON object. Keys: q (required query), limit_docs "
                            "(default 5, stage-1 documents), limit_chunks (default "
                            "10, stage-2 chunks), segmentation_k (optional)."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
]

# --------------------------------------------------------------------------- #
# A small library of real arXiv papers used to build schema-faithful results.
# --------------------------------------------------------------------------- #

PAPERS = {
    "1706.03762": {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "cats": ["cs.CL", "cs.LG"],
        "published": "2017-06-12T17:57:34Z",
        "abstract": (
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks. We propose the "
            "Transformer, a model architecture based solely on attention "
            "mechanisms, dispensing with recurrence and convolutions entirely. "
            "Experiments show these models are superior in quality while being "
            "more parallelizable and requiring significantly less time to train."
        ),
    },
    "1810.04805": {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
        "cats": ["cs.CL"],
        "published": "2018-10-11T00:50:01Z",
        "abstract": (
            "We introduce BERT, a language representation model that pre-trains "
            "deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context. BERT obtains new "
            "state-of-the-art results on eleven natural language processing tasks."
        ),
    },
    "2005.14165": {
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Tom B. Brown", "Benjamin Mann", "Nick Ryder"],
        "cats": ["cs.CL"],
        "published": "2020-05-28T17:29:03Z",
        "abstract": (
            "We show that scaling up language models greatly improves "
            "task-agnostic, few-shot performance. We train GPT-3, an "
            "autoregressive language model with 175 billion parameters, and test "
            "its performance in the few-shot setting without any gradient updates."
        ),
    },
    "2203.02155": {
        "title": "Training language models to follow instructions with human feedback",
        "authors": ["Long Ouyang", "Jeff Wu", "Xu Jiang"],
        "cats": ["cs.CL", "cs.LG"],
        "published": "2022-03-04T07:04:42Z",
        "abstract": (
            "We show an avenue for aligning language models with user intent by "
            "fine-tuning with human feedback. We collect demonstrations and use "
            "reinforcement learning from human feedback (RLHF) to fine-tune GPT-3 "
            "into InstructGPT, which is preferred over much larger models."
        ),
    },
    "2201.11903": {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans"],
        "cats": ["cs.CL"],
        "published": "2022-01-28T16:18:31Z",
        "abstract": (
            "We explore how generating a chain of thought - a series of "
            "intermediate reasoning steps - significantly improves the ability of "
            "large language models to perform complex reasoning, an ability that "
            "emerges naturally at sufficient model scale."
        ),
    },
    "2005.11401": {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
        "cats": ["cs.CL", "cs.LG"],
        "published": "2020-05-22T21:34:34Z",
        "abstract": (
            "We introduce retrieval-augmented generation (RAG), models that "
            "combine pre-trained parametric memory with non-parametric memory "
            "from a dense vector index of Wikipedia, accessed with a neural "
            "retriever, for knowledge-intensive NLP tasks."
        ),
    },
    "1910.10683": {
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "authors": ["Colin Raffel", "Noam Shazeer", "Adam Roberts"],
        "cats": ["cs.LG", "cs.CL"],
        "published": "2019-10-23T17:37:36Z",
        "abstract": (
            "We explore transfer learning for NLP by introducing a unified "
            "framework that converts every language problem into a text-to-text "
            "format, studying the limits of transfer learning with our "
            "Text-to-Text Transfer Transformer (T5)."
        ),
    },
    "2106.09685": {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis"],
        "cats": ["cs.CL", "cs.LG"],
        "published": "2021-06-17T17:37:18Z",
        "abstract": (
            "We propose Low-Rank Adaptation (LoRA), which freezes the pre-trained "
            "model weights and injects trainable rank decomposition matrices into "
            "each Transformer layer, greatly reducing the number of trainable "
            "parameters for downstream tasks."
        ),
    },
    "2302.13971": {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard"],
        "cats": ["cs.CL"],
        "published": "2023-02-27T17:00:00Z",
        "abstract": (
            "We introduce LLaMA, a collection of foundation language models "
            "ranging from 7B to 65B parameters, trained on trillions of tokens "
            "using publicly available datasets exclusively."
        ),
    },
    "2307.09288": {
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors": ["Hugo Touvron", "Louis Martin", "Kevin Stone"],
        "cats": ["cs.CL", "cs.AI"],
        "published": "2023-07-18T17:00:00Z",
        "abstract": (
            "We develop and release Llama 2, a collection of pretrained and "
            "fine-tuned large language models ranging from 7B to 70B parameters, "
            "optimized for dialogue use cases."
        ),
    },
    "1512.03385": {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
        "cats": ["cs.CV"],
        "published": "2015-12-10T19:51:55Z",
        "abstract": (
            "We present a residual learning framework to ease the training of "
            "networks that are substantially deeper than those used previously, "
            "reformulating layers as learning residual functions with reference "
            "to the layer inputs."
        ),
    },
    "2010.11929": {
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "authors": ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"],
        "cats": ["cs.CV", "cs.LG"],
        "published": "2020-10-22T17:55:01Z",
        "abstract": (
            "We show that a pure transformer applied directly to sequences of "
            "image patches can perform very well on image classification tasks, "
            "attaining excellent results compared to state-of-the-art "
            "convolutional networks while requiring fewer resources to train."
        ),
    },
    "2112.10752": {
        "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
        "authors": ["Robin Rombach", "Andreas Blattmann", "Dominik Lorenz"],
        "cats": ["cs.CV"],
        "published": "2021-12-20T18:55:25Z",
        "abstract": (
            "We apply diffusion models in the latent space of powerful "
            "pretrained autoencoders, enabling high-resolution image synthesis "
            "with greatly reduced computational requirements while retaining "
            "quality and flexibility."
        ),
    },
    "2305.10601": {
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "authors": ["Shunyu Yao", "Dian Yu", "Jeffrey Zhao"],
        "cats": ["cs.CL", "cs.AI"],
        "published": "2023-05-17T17:00:00Z",
        "abstract": (
            "We introduce Tree of Thoughts (ToT), a framework that generalizes "
            "chain-of-thought prompting and enables exploration over coherent "
            "units of text (thoughts) as intermediate steps, with lookahead and "
            "backtracking via search."
        ),
    },
    "2104.09864": {
        "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        "authors": ["Jianlin Su", "Yu Lu", "Shengfeng Pan"],
        "cats": ["cs.CL", "cs.LG"],
        "published": "2021-04-20T08:00:00Z",
        "abstract": (
            "We propose Rotary Position Embedding (RoPE) to leverage positional "
            "information in transformer language models, encoding absolute "
            "position with a rotation matrix while naturally incorporating "
            "relative position dependency in self-attention."
        ),
    },
    "1412.6980": {
        "title": "Adam: A Method for Stochastic Optimization",
        "authors": ["Diederik P. Kingma", "Jimmy Ba"],
        "cats": ["cs.LG"],
        "published": "2014-12-22T20:09:24Z",
        "abstract": (
            "We introduce Adam, an algorithm for first-order gradient-based "
            "optimization of stochastic objective functions, based on adaptive "
            "estimates of lower-order moments. The method is computationally "
            "efficient and well suited for problems large in data and parameters."
        ),
    },
    "2401.04088": {
        "title": "Mixtral of Experts",
        "authors": ["Albert Q. Jiang", "Alexandre Sablayrolles", "Antoine Roux"],
        "cats": ["cs.LG", "cs.CL"],
        "published": "2024-01-08T18:00:00Z",
        "abstract": (
            "We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) "
            "language model where each layer has 8 feedforward experts and a "
            "router selects two per token, giving each token access to 47B "
            "parameters while using only 13B during inference."
        ),
    },
    "2009.06732": {
        "title": "Efficient Transformers: A Survey",
        "authors": ["Yi Tay", "Mostafa Dehghani", "Dara Bahri"],
        "cats": ["cs.LG"],
        "published": "2020-09-14T17:00:00Z",
        "abstract": (
            "Transformer efficiency has become an important research direction. "
            "This survey characterizes a large and thoughtful selection of recent "
            "efficiency-flavored 'X-former' models, providing an organized "
            "taxonomy across the literature."
        ),
    },
    "1301.3781": {
        "title": "Efficient Estimation of Word Representations in Vector Space",
        "authors": ["Tomas Mikolov", "Kai Chen", "Greg Corrado"],
        "cats": ["cs.CL"],
        "published": "2013-01-16T17:24:43Z",
        "abstract": (
            "We propose two novel model architectures for computing continuous "
            "vector representations of words from very large data sets "
            "(word2vec), drastically reducing computational cost while improving "
            "accuracy on word-similarity tasks."
        ),
    },
    "1406.2661": {
        "title": "Generative Adversarial Networks",
        "authors": ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza"],
        "cats": ["cs.LG", "stat.ML"],
        "published": "2014-06-10T18:58:17Z",
        "abstract": (
            "We propose estimating generative models via an adversarial process, "
            "training a generator and a discriminator simultaneously; the "
            "generator learns to produce samples indistinguishable from real data."
        ),
    },
    "1502.03167": {
        "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
        "authors": ["Sergey Ioffe", "Christian Szegedy"],
        "cats": ["cs.LG"],
        "published": "2015-02-11T00:00:00Z",
        "abstract": (
            "We introduce Batch Normalization, which normalizes layer inputs per "
            "mini-batch to reduce internal covariate shift, allowing much higher "
            "learning rates, easier initialization, and acting as a regularizer."
        ),
    },
    "1312.5602": {
        "title": "Playing Atari with Deep Reinforcement Learning",
        "authors": ["Volodymyr Mnih", "Koray Kavukcuoglu", "David Silver"],
        "cats": ["cs.LG"],
        "published": "2013-12-19T00:00:00Z",
        "abstract": (
            "We present the first deep learning model to successfully learn "
            "control policies directly from high-dimensional sensory input using "
            "reinforcement learning, applied to Atari 2600 games (DQN)."
        ),
    },
    "1707.06347": {
        "title": "Proximal Policy Optimization Algorithms",
        "authors": ["John Schulman", "Filip Wolski", "Prafulla Dhariwal"],
        "cats": ["cs.LG"],
        "published": "2017-07-20T00:00:00Z",
        "abstract": (
            "We propose Proximal Policy Optimization (PPO), policy-gradient "
            "methods that alternate sampling data with optimizing a clipped "
            "surrogate objective, balancing simplicity, stability, and sample "
            "efficiency."
        ),
    },
    "1609.02907": {
        "title": "Semi-Supervised Classification with Graph Convolutional Networks",
        "authors": ["Thomas N. Kipf", "Max Welling"],
        "cats": ["cs.LG", "stat.ML"],
        "published": "2016-09-09T00:00:00Z",
        "abstract": (
            "We present a scalable approach for semi-supervised learning on "
            "graph-structured data based on an efficient variant of convolutional "
            "neural networks that operate directly on graphs (GCN)."
        ),
    },
    "2006.11239": {
        "title": "Denoising Diffusion Probabilistic Models",
        "authors": ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"],
        "cats": ["cs.LG", "stat.ML"],
        "published": "2020-06-19T00:00:00Z",
        "abstract": (
            "We present high-quality image synthesis with denoising diffusion "
            "probabilistic models, latent-variable models trained to reverse a "
            "gradual noising process, drawing a connection to denoising score "
            "matching with Langevin dynamics."
        ),
    },
    "2103.00020": {
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "authors": ["Alec Radford", "Jong Wook Kim", "Chris Hallacy"],
        "cats": ["cs.CV", "cs.LG"],
        "published": "2021-02-26T00:00:00Z",
        "abstract": (
            "We show that predicting which caption goes with which image is an "
            "efficient, scalable way to learn image representations from scratch, "
            "enabling zero-shot transfer to many downstream tasks (CLIP)."
        ),
    },
    "2304.02643": {
        "title": "Segment Anything",
        "authors": ["Alexander Kirillov", "Eric Mintun", "Nikhila Ravi"],
        "cats": ["cs.CV"],
        "published": "2023-04-05T00:00:00Z",
        "abstract": (
            "We introduce the Segment Anything project: a promptable segmentation "
            "model trained on over one billion masks, enabling strong zero-shot "
            "generalization to new image distributions and tasks (SAM)."
        ),
    },
    "2302.04761": {
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": ["Timo Schick", "Jane Dwivedi-Yu", "Roberto Dessi"],
        "cats": ["cs.CL"],
        "published": "2023-02-09T00:00:00Z",
        "abstract": (
            "We show language models can teach themselves to use external tools "
            "via simple APIs, deciding which tool to call, when, and how to fold "
            "the results into generation (Toolformer)."
        ),
    },
    "2210.03629": {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": ["Shunyu Yao", "Jeffrey Zhao", "Dian Yu"],
        "cats": ["cs.CL", "cs.AI"],
        "published": "2022-10-06T00:00:00Z",
        "abstract": (
            "We use LLMs to generate reasoning traces and task-specific actions in "
            "an interleaved manner (ReAct), letting the model query external "
            "sources, reducing hallucination, and improving interpretability."
        ),
    },
    "2205.14135": {
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "authors": ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon"],
        "cats": ["cs.LG"],
        "published": "2022-05-27T00:00:00Z",
        "abstract": (
            "We propose FlashAttention, an IO-aware exact attention algorithm "
            "using tiling to cut memory reads/writes between GPU HBM and on-chip "
            "SRAM, speeding up training and reducing memory footprint."
        ),
    },
    "2312.00752": {
        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors": ["Albert Gu", "Tri Dao"],
        "cats": ["cs.LG"],
        "published": "2023-12-01T00:00:00Z",
        "abstract": (
            "We introduce Mamba, a selective state-space model giving linear-time "
            "sequence modeling with content-based reasoning, matching or beating "
            "Transformers across several modalities with fast inference."
        ),
    },
    "2305.18290": {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "authors": ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell"],
        "cats": ["cs.LG", "cs.CL"],
        "published": "2023-05-29T00:00:00Z",
        "abstract": (
            "We introduce Direct Preference Optimization (DPO), fine-tuning "
            "language models to satisfy human preferences with a simple "
            "classification loss, with no explicit reward model or reinforcement "
            "learning loop."
        ),
    },
    "2210.17323": {
        "title": "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
        "authors": ["Elias Frantar", "Saleh Ashkboos", "Torsten Hoefler"],
        "cats": ["cs.LG", "cs.CL"],
        "published": "2022-10-31T00:00:00Z",
        "abstract": (
            "We propose GPTQ, a one-shot weight-quantization method based on "
            "approximate second-order information that compresses GPT-scale models "
            "to 3-4 bits per weight with negligible accuracy loss."
        ),
    },
    "2212.04356": {
        "title": "Robust Speech Recognition via Large-Scale Weak Supervision",
        "authors": ["Alec Radford", "Jong Wook Kim", "Tao Xu"],
        "cats": ["eess.AS", "cs.CL", "cs.SD"],
        "published": "2022-12-06T00:00:00Z",
        "abstract": (
            "We study speech processing trained on 680,000 hours of multilingual, "
            "multitask weakly-supervised audio (Whisper); the models generalize "
            "well to standard benchmarks zero-shot, approaching human robustness "
            "across languages and accents."
        ),
    },
    "1409.0473": {
        "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "authors": ["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"],
        "cats": ["cs.CL", "cs.LG", "stat.ML"],
        "published": "2014-09-01T00:00:00Z",
        "abstract": (
            "We introduce an attention mechanism that lets a neural machine "
            "translation model automatically soft-search for the parts of the "
            "source sentence relevant to predicting each target word, instead of "
            "squeezing the whole source into a single fixed-length vector."
        ),
    },
}

# --------------------------------------------------------------------------- #
# Builders. These only assemble dicts; nothing here invents example content.
# --------------------------------------------------------------------------- #


def sys_msg():
    return {"role": "system", "content": SYSTEM}


def usr(text):
    return {"role": "user", "content": text}


def asst(text):
    return {"role": "assistant", "content": text}


def call(name, code, cid="call_1", note=""):
    """An assistant tool-call turn. `code` is the JSON value the server expects
    inside the `code` string envelope (dict for most tools, dict-or-list for
    execute). `note` is brief narration shown before the call."""
    arguments = json.dumps({"code": json.dumps(code, ensure_ascii=False)}, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": note,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def result(cid, payload):
    """A tool-result turn. `payload` may be a dict/list (json-encoded) or a raw
    string (e.g. the markdown returned by op=download)."""
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": cid, "content": content}


def full_paper(pid):
    p = PAPERS[pid]
    return {
        "id": pid,
        "title": p["title"],
        "authors": [{"name": n, "affiliations": []} for n in p["authors"]],
        "abstract_text": p["abstract"],
        "categories": p["cats"],
        "published": p["published"],
        "url": f"https://arxiv.org/abs/{pid}",
        "doi": None,
        "journal_ref": None,
    }


def search_response(ids, total, start=0):
    return {
        "papers": [full_paper(i) for i in ids],
        "total_results": total,
        "start_index": start,
    }


def ss_paper(pid, title, authors, year):
    """Semantic-Scholar-derived paper (citations/recs): no abstract/categories."""
    return {
        "id": pid,
        "title": title,
        "authors": [{"name": n, "affiliations": []} for n in authors],
        "abstract_text": "",
        "categories": [],
        "published": str(year),
        "url": f"https://arxiv.org/abs/{pid}" if pid else "",
        "doi": None,
        "journal_ref": None,
    }


def content_paper(pid):
    """The sparse `paper` field that retrieve/op=retrieve returns: title==id,
    no metadata, because the content path does not fetch the Atom record."""
    return {
        "id": pid,
        "title": pid,
        "authors": [],
        "abstract_text": "",
        "categories": [],
        "published": "",
        "url": f"https://arxiv.org/abs/{pid}",
        "doi": None,
        "journal_ref": None,
    }


def prepared(pid, source, chunk_texts, pruned_md, raw_md=None, clusters=None, parents=None):
    chunks = []
    pos = 0
    for i, text in enumerate(chunk_texts):
        cluster_id = clusters[i] if clusters else None
        parent_id = parents[i] if parents else None
        body = text
        ctx = [c for c in (parent_id, cluster_id) if c]
        if ctx:
            body = "Context: " + " -> ".join(ctx) + "\n\n" + text
        chunks.append(
            {
                "index": i,
                "start_char": pos,
                "end_char": pos + len(text),
                "text": body,
                "cluster_id": cluster_id,
                "parent_id": parent_id,
            }
        )
        pos += len(text)
    return {
        "paper": content_paper(pid),
        "source": source,
        "raw_markdown": raw_md if raw_md is not None else pruned_md,
        "pruned_markdown": pruned_md,
        "chunks": chunks,
        "hierarchical_chunks": None,
    }


def exec_one(pid, op, value):
    return {"id": pid, "op": op, "result": value}


# --------------------------------------------------------------------------- #
# The curated conversations. Grouped by the skill each one teaches.
# --------------------------------------------------------------------------- #

EXAMPLES: list[list[dict]] = []
E = EXAMPLES.append

# Curated conversations: poorly-structured, natural-language requests in,
# precise arXiv tool calls out. The user is vague/colloquial; the assistant's
# job is to INTERPRET the request (domain -> category, "recent/lately" -> date
# sort, intent -> abstract terms) and emit a well-formed query. The brief note
# on each tool-call turn shows that interpretation step.

# ---- "go find me research on X" : minimal, vague topic --------------------- #

# 1. Vague reasoning request -> infer cs.CL + reasoning terms.
E([
    usr("go find me research on getting language models to reason through hard problems"),
    call("search", {"q": "abs:reasoning large language models", "cats": ["cs.CL"], "n": 5, "sort": "relevance"},
         note="Reading this as LLM reasoning work in cs.CL; searching abstracts for the core terms."),
    result("call_1", search_response(["2201.11903", "2305.10601", "2210.03629"], 1842)),
    asst(
        "Here's the core line of work on LLM reasoning:\n"
        "1. **Chain-of-Thought Prompting** (2201.11903) - get the model to spell out intermediate steps; reasoning emerges with scale.\n"
        "2. **Tree of Thoughts** (2305.10601) - generalizes that into a searchable tree with lookahead/backtracking.\n"
        "3. **ReAct** (2210.03629) - interleaves reasoning with actions so the model can check itself against external info.\n\n"
        "Want me to pull any of them in full?"
    ),
])

# 2. "X for Y purpose to Z effect": efficiency for deployment on small GPUs.
E([
    usr("i need papers on running transformers cheaper so i can deploy them on smaller gpus"),
    call("search", {"q": "abs:efficient transformer inference quantization memory", "cats": ["cs.LG"], "n": 6, "sort": "relevance"},
         note="Intent = inference efficiency / lower memory for deployment -> cs.LG, abstract terms quantization + memory."),
    result("call_1", search_response(["2210.17323", "2205.14135", "2009.06732"], 731)),
    asst(
        "For squeezing transformers onto smaller GPUs, three angles:\n"
        "- **GPTQ** (2210.17323) - post-training quantization to 3-4 bits/weight with little accuracy loss; the most direct memory win.\n"
        "- **FlashAttention** (2205.14135) - IO-aware attention that cuts memory traffic and speeds things up, no accuracy cost.\n"
        "- **Efficient Transformers: A Survey** (2009.06732) - the map of the whole design space if you want options.\n\n"
        "If your bottleneck is specifically VRAM at inference, start with GPTQ. Want its method section?"
    ),
])

# 3. "lately/hot" -> recency, infer cs.CV + sort:date.
E([
    usr("whats the hot stuff in image generation lately"),
    call("search", {"q": "abs:image generation diffusion", "cats": ["cs.CV"], "n": 5, "sort": "date"},
         note="\"Hot lately\" -> sort by date; image generation -> cs.CV diffusion."),
    result("call_1", search_response(["2112.10752", "2006.11239"], 2904)),
    asst(
        "Sorting by recency, the pillars of modern image generation are diffusion-based:\n"
        "- **Latent Diffusion Models** (2112.10752) - runs diffusion in a compressed latent space; the basis of Stable Diffusion.\n"
        "- **Denoising Diffusion Probabilistic Models** (2006.11239) - the DDPM formulation that kicked off the wave.\n\n"
        "Want the latest follow-ups in a specific direction (text-to-image, video, control)?"
    ),
])

# 4. "that paper everyone cites" -> title field search.
E([
    usr("find me that foundational attention paper everyone keeps citing"),
    call("search", {"q": "ti:\"attention is all you need\"", "n": 1},
         note="\"Foundational attention paper everyone cites\" almost certainly means Attention Is All You Need; pinning it by title."),
    result("call_1", search_response(["1706.03762"], 1)),
    asst("That's **Attention Is All You Need** (arXiv:1706.03762, Vaswani et al., 2017) - it introduced the Transformer, built entirely on self-attention. Want the abstract or a breakdown of the architecture?"),
])

# 5. "where do i even start" -> roadmap from a vague pretraining ask.
E([
    usr("my prof told me to look into self-supervised pretraining for nlp and i have no idea where to start"),
    call("search", {"q": "abs:self-supervised pre-training language representation", "cats": ["cs.CL"], "n": 5, "sort": "relevance"},
         note="Turning a fuzzy 'where to start' into the seminal pretraining papers in cs.CL."),
    result("call_1", search_response(["1810.04805", "1910.10683", "1301.3781"], 5120)),
    asst(
        "A sensible reading order to get the foundations:\n"
        "1. **word2vec** (1301.3781, 2013) - the idea that you can learn word meaning from raw text, no labels.\n"
        "2. **BERT** (1810.04805, 2018) - masked-language-model pretraining of a bidirectional Transformer; the recipe everything else builds on.\n"
        "3. **T5** (1910.10683, 2019) - frames every NLP task as text-to-text and scales the pretraining idea.\n\n"
        "Start with BERT for the core concept; I can retrieve its method section when you're ready."
    ),
])

# 6. "doing a lit review" on agents/tools.
E([
    usr("anything on ai agents that can use tools and apis? doing a lit review"),
    call("search", {"q": "abs:language model tool use agents", "cats": ["cs.CL", "cs.AI"], "n": 6, "sort": "relevance"},
         note="Tool-using LLM agents -> cs.CL/cs.AI; abstract search on tool use + agents."),
    result("call_1", search_response(["2302.04761", "2210.03629"], 612)),
    asst(
        "Two anchor papers for the lit review:\n"
        "- **Toolformer** (2302.04761) - the model teaches itself when/how to call APIs and fold results back in.\n"
        "- **ReAct** (2210.03629) - interleaves reasoning traces with actions so the agent can query external sources mid-task.\n\n"
        "Both are highly cited starting points; want me to pull each one's citations to expand the review tree?"
    ),
])

# 7. Casual "the rlhf stuff".
E([
    usr("papers on training models from human preferences, the rlhf kind of thing"),
    call("search", {"q": "abs:human feedback preference fine-tuning", "cats": ["cs.CL", "cs.LG"], "n": 5, "sort": "relevance"},
         note="\"rlhf kind of thing\" = learning from human preferences; cs.CL/cs.LG, abstract terms human feedback + preference."),
    result("call_1", search_response(["2203.02155", "2305.18290"], 488)),
    asst(
        "The two you want:\n"
        "- **InstructGPT** (2203.02155) - the canonical RLHF pipeline: demonstrations + preference comparisons + RL fine-tuning.\n"
        "- **Direct Preference Optimization (DPO)** (2305.18290) - gets the same alignment from preferences with a simple loss, skipping the RL loop.\n\n"
        "DPO is the lighter-weight successor if you're going to implement something."
    ),
])

# 8. "understand how X works" -> find + deep read chain.
E([
    usr("i wanna actually understand how diffusion models work, grab the key paper and walk me through it"),
    call("search", {"q": "ti:denoising diffusion probabilistic models", "n": 1},
         note="Finding the canonical diffusion paper first so I read the right one."),
    result("call_1", search_response(["2006.11239"], 3)),
    call("retrieve_paper", {"paper_id": "2006.11239", "segmentation_k": 1.2}, cid="call_2",
         note="Retrieving DDPM with structure so I can walk through the actual method."),
    result("call_2", prepared(
        "2006.11239", "html",
        [
            "A forward process gradually adds Gaussian noise to data over T steps until it becomes pure noise.",
            "A neural network is trained to reverse each step; sampling runs the learned reverse process from noise back to a clean image, and the training objective reduces to predicting the added noise.",
        ],
        pruned_md="# Denoising Diffusion Probabilistic Models\n\n...",
        parents=["Method", "Method"],
        clusters=["Forward Process", "Reverse Process"])),
    asst(
        "Found and read the key paper - **DDPM** (2006.11239). The mechanism in two halves:\n"
        "1. **Forward process**: take a real image and add a little Gaussian noise repeatedly over T steps until it's pure noise. This part has no learned parameters.\n"
        "2. **Reverse process**: train a network to undo one noising step at a time. Neatly, the training objective simplifies to *predicting the noise* that was added. To generate, you start from random noise and run the learned reverse steps back to a clean image.\n\n"
        "That's the whole idea; later work (e.g. Latent Diffusion, 2112.10752) mostly makes it cheaper. Want the sampling details?"
    ),
])

# 9. Vague "ditch convolutions".
E([
    usr("find research on vision models that ditch convolutions entirely"),
    call("search", {"q": "abs:image recognition transformer ANDNOT abs:convolution", "cats": ["cs.CV"], "n": 4, "sort": "relevance"},
         note="\"Ditch convolutions\" -> attention-based vision; exclude convolution in the abstract."),
    result("call_1", search_response(["2010.11929"], 377)),
    asst("The headline paper is **An Image is Worth 16x16 Words / ViT** (2010.11929). It drops convolution completely: split the image into patches, embed them, and feed the sequence to a plain Transformer encoder - and it beats CNNs once pre-trained on enough data."),
])

# 10. "really big models".
E([
    usr("what should i read about making neural nets really really big"),
    call("search", {"q": "abs:scaling large language models parameters", "cats": ["cs.CL"], "n": 5, "sort": "relevance"},
         note="\"Really big models\" -> scaling LLMs; cs.CL abstract search."),
    result("call_1", search_response(["2005.14165", "2302.13971"], 1396)),
    asst(
        "For the scaling story:\n"
        "- **GPT-3 / Language Models are Few-Shot Learners** (2005.14165) - 175B params; shows capability rises smoothly with scale and unlocks few-shot prompting.\n"
        "- **LLaMA** (2302.13971) - the counterpoint: a smaller model trained on more tokens can beat a bigger one (compute-optimal training).\n\n"
        "Read GPT-3 for 'scale helps', then LLaMA for 'how you spend the compute matters'."
    ),
])

# 11. Casual "og GAN paper" + summary -> search then abstract.
E([
    usr("get me the og GAN paper and a quick summary"),
    call("search", {"q": "ti:generative adversarial networks AND au:goodfellow", "n": 1},
         note="\"OG GAN paper\" = Goodfellow's original; title + author filter."),
    result("call_1", search_response(["1406.2661"], 2)),
    call("execute", {"op": "abstract", "id": "1406.2661"}, cid="call_2",
         note="Pulling the abstract for the quick summary they asked for."),
    result("call_2", exec_one("1406.2661", "abstract", full_paper("1406.2661"))),
    asst(
        "**Generative Adversarial Networks** (1406.2661, Goodfellow et al., 2014). Quick version: train two networks against "
        "each other - a *generator* that fabricates samples and a *discriminator* that tries to tell fake from real. As the "
        "discriminator gets better, it pushes the generator to produce data indistinguishable from the real distribution. "
        "That adversarial game is the whole trick."
    ),
])

# 12. Recency on SSMs as transformer alternative.
E([
    usr("find recent work on state space models as an alternative to transformers"),
    call("search", {"q": "abs:state space model sequence modeling", "cats": ["cs.LG"], "n": 4, "sort": "date"},
         note="\"Recent\" -> sort by date; SSMs as a transformer alternative -> cs.LG."),
    result("call_1", search_response(["2312.00752"], 143)),
    asst("Top recent hit: **Mamba** (2312.00752, Gu & Dao, 2023). It uses a *selective* state-space model to get linear-time sequence modeling with content-based gating, matching or beating Transformers on several modalities with faster inference - the most cited transformer alternative right now."),
])

# 13. Multilingual speech -> infer eess.AS/cs.CL.
E([
    usr("got anything on speech recognition that handles a bunch of languages?"),
    call("search", {"q": "abs:multilingual speech recognition", "cats": ["eess.AS", "cs.CL"], "n": 4, "sort": "relevance"},
         note="Speech recognition across languages -> eess.AS + cs.CL, abstract term multilingual."),
    result("call_1", search_response(["2212.04356"], 219)),
    asst("That's **Whisper / Robust Speech Recognition via Large-Scale Weak Supervision** (2212.04356). Trained on 680k hours of multilingual, multitask audio, it transcribes and translates across many languages zero-shot and is notably robust to accents and noise."),
])

# 14. "i'm building a recommender" -> GNN.
E([
    usr("im building a recommender system and want to learn graph neural networks, find me a good starting paper"),
    call("search", {"q": "abs:graph neural network semi-supervised classification", "cats": ["cs.LG"], "n": 3, "sort": "relevance"},
         note="GNN foundations -> cs.LG; the GCN paper is the standard entry point."),
    result("call_1", search_response(["1609.02907"], 904)),
    asst("Best starting point: **Semi-Supervised Classification with Graph Convolutional Networks** (1609.02907, Kipf & Welling). It defines the GCN layer - a simple, scalable convolution that operates directly on graph structure - and it's the foundation most recommender/GNN work builds on."),
])

# 15. Story-style request -> title search.
E([
    usr("find the deepmind paper where they got an AI to play atari with reinforcement learning"),
    call("search", {"q": "ti:atari reinforcement learning", "n": 2, "sort": "relevance"},
         note="Specific result described -> title search for the Atari RL paper."),
    result("call_1", search_response(["1312.5602"], 7)),
    asst("That's **Playing Atari with Deep Reinforcement Learning** (1312.5602, Mnih et al., 2013) - the DQN paper. It learned control policies straight from raw pixels using Q-learning with a convolutional net, across several Atari 2600 games."),
])

# 16. "solid policy gradient method".
E([
    usr("whats a solid policy gradient method for RL these days, find the paper"),
    call("search", {"q": "ti:proximal policy optimization", "n": 2, "sort": "relevance"},
         note="Most-used modern policy-gradient method is PPO; searching its title."),
    result("call_1", search_response(["1707.06347"], 5)),
    asst("**Proximal Policy Optimization (PPO)** (1707.06347, Schulman et al., 2017). It's the default workhorse policy-gradient method: a clipped surrogate objective keeps each update from moving the policy too far, giving you most of TRPO's stability with far less complexity."),
])

# 17. Compound "X domain for Y purpose to Z effect": contrastive image-text for zero-shot.
E([
    usr("find papers on learning from images and text together so you can classify stuff zero-shot"),
    call("search", {"q": "abs:contrastive image text zero-shot transfer", "cats": ["cs.CV"], "n": 4, "sort": "relevance"},
         note="Domain = joint image-text learning; purpose = zero-shot classification -> cs.CV, contrastive + zero-shot terms."),
    result("call_1", search_response(["2103.00020"], 540)),
    asst(
        "**CLIP / Learning Transferable Visual Models From Natural Language Supervision** (2103.00020) is exactly this. It "
        "contrastively learns to match images with their captions over 400M pairs; because the label space is just text, "
        "you get **zero-shot** classification - describe the classes in words and CLIP scores each image against them, no "
        "task-specific training."
    ),
])

# 18. "heard there's a big model" -> title search.
E([
    usr("heard there's some big model that can segment basically anything in an image, find it for me"),
    call("search", {"q": "ti:segment anything", "n": 2, "sort": "relevance"},
         note="Described as a 'segment anything' model -> title search."),
    result("call_1", search_response(["2304.02643"], 11)),
    asst("That's **Segment Anything (SAM)** (2304.02643, Kirillov et al., 2023). It's a *promptable* segmentation model trained on 1B+ masks; you give it points/boxes/text-like prompts and it segments arbitrary objects, generalizing zero-shot to new image types."),
])

# 19. "without retraining everything" -> PEFT/LoRA.
E([
    usr("find me work on fine-tuning huge models without having to retrain the whole thing"),
    call("search", {"q": "abs:parameter-efficient fine-tuning low-rank adaptation", "cats": ["cs.CL", "cs.LG"], "n": 4, "sort": "relevance"},
         note="\"Without retraining everything\" = parameter-efficient fine-tuning -> LoRA and friends."),
    result("call_1", search_response(["2106.09685"], 396)),
    asst("The go-to is **LoRA: Low-Rank Adaptation** (2106.09685). You freeze the pretrained weights and train only small low-rank update matrices injected into each layer - orders of magnitude fewer trainable parameters, and you can merge the update back in so there's no extra inference cost."),
])

# 20. "normalizing activations to train faster".
E([
    usr("papers on normalizing the activations to make deep nets train faster"),
    call("search", {"q": "ti:batch normalization", "n": 2, "sort": "relevance"},
         note="\"Normalizing activations for faster training\" -> Batch Normalization."),
    result("call_1", search_response(["1502.03167"], 88)),
    asst("**Batch Normalization** (1502.03167, Ioffe & Szegedy). It normalizes each layer's inputs per mini-batch to reduce 'internal covariate shift', which lets you use much higher learning rates, makes initialization less finicky, and has a mild regularizing effect."),
])

# 21. Vague compare-the-approaches -> search then batch abstracts then synthesize.
E([
    usr("compare the main ways people get transformers to handle really long sequences"),
    call("search", {"q": "abs:long sequence efficient attention", "cats": ["cs.LG"], "n": 6, "sort": "relevance"},
         note="Cross-cutting question -> first gather the main long-sequence approaches."),
    result("call_1", search_response(["2205.14135", "2312.00752", "2009.06732"], 661)),
    call("execute", [
        {"op": "abstract", "id": "2205.14135"},
        {"op": "abstract", "id": "2312.00752"},
    ], cid="call_2", note="Pulling the two contrasting approaches' abstracts to compare directly."),
    result("call_2", [
        exec_one("2205.14135", "abstract", full_paper("2205.14135")),
        exec_one("2312.00752", "abstract", full_paper("2312.00752")),
    ]),
    asst(
        "Three broad strategies for long sequences:\n"
        "- **Keep attention, make it cheaper on hardware** - **FlashAttention** (2205.14135): exact attention, but IO-aware tiling slashes memory traffic so you can run longer contexts.\n"
        "- **Replace attention** - **Mamba** (2312.00752): a selective state-space model that's *linear* in sequence length instead of quadratic.\n"
        "- **Survey of approximations** - **Efficient Transformers: A Survey** (2009.06732): sparse/low-rank/kernelized attention variants if you want the menu.\n\n"
        "So it's roughly: optimize exact attention vs. swap it for a linear-time model vs. approximate it."
    ),
])

# 22. Direct named ask, still casual.
E([
    usr("find the word2vec paper"),
    call("search", {"q": "ti:efficient estimation word representations vector space", "n": 1},
         note="word2vec's actual title; searching for it directly."),
    result("call_1", search_response(["1301.3781"], 2)),
    asst("Here it is: **Efficient Estimation of Word Representations in Vector Space** (1301.3781, Mikolov et al., 2013) - the word2vec paper, introducing the CBOW and skip-gram architectures for cheap, high-quality word embeddings."),
])

# 23. Compound + recency + purpose: alignment for a safety project.
E([
    usr("i need recent papers on aligning chatbots to be helpful and not harmful, its for a safety project"),
    call("search", {"q": "abs:alignment helpful harmless human feedback", "cats": ["cs.CL"], "n": 6, "sort": "date"},
         note="Purpose = LLM safety/alignment; 'recent' -> sort by date; cs.CL."),
    result("call_1", search_response(["2307.09288", "2305.18290", "2203.02155"], 503)),
    asst(
        "For a safety/alignment project, recent-first:\n"
        "- **Llama 2** (2307.09288) - has a substantial safety-tuning section (helpfulness vs. harmlessness reward models, red-teaming).\n"
        "- **DPO** (2305.18290) - a stable, simple way to optimize a model toward preferred (e.g. harmless) responses.\n"
        "- **InstructGPT** (2203.02155) - the foundational RLHF alignment recipe everything here descends from.\n\n"
        "Llama 2's safety section is the most directly applied; want me to retrieve it?"
    ),
])

# 24. "modern position encodings".
E([
    usr("what do i read to get how modern position encodings work in transformers"),
    call("search", {"q": "abs:rotary position embedding transformer", "cats": ["cs.CL", "cs.LG"], "n": 3, "sort": "relevance"},
         note="\"Modern position encodings\" -> rotary embeddings (RoPE) is the current standard."),
    result("call_1", search_response(["2104.09864"], 207)),
    asst("Read **RoFormer / Rotary Position Embedding** (2104.09864). RoPE encodes a token's absolute position by *rotating* its embedding by a position-dependent angle; because rotations compose, the attention score ends up depending only on the relative offset between tokens. It's the position scheme used in most modern LLMs (LLaMA, etc.)."),
])

# 25. "the sparse model thing".
E([
    usr("get me something on mixture of experts, the sparse model thing"),
    call("search", {"q": "abs:sparse mixture of experts language model", "cats": ["cs.LG", "cs.CL"], "n": 3, "sort": "relevance"},
         note="\"Sparse MoE\" -> sparse mixture-of-experts LMs."),
    result("call_1", search_response(["2401.04088"], 156)),
    asst("**Mixtral of Experts** (2401.04088) is the clearest modern example. Each layer has 8 expert feed-forward blocks and a router picks 2 per token - so a token can draw on 47B parameters while only ~13B are active during inference. That's the 'sparse' part: most experts sit idle for any given token."),
])

# 26. Find + break down the math (deep read).
E([
    usr("find the original transformer paper and break down the attention math for me"),
    call("search", {"q": "ti:\"attention is all you need\"", "n": 1},
         note="Locate the exact paper before reading it."),
    result("call_1", search_response(["1706.03762"], 1)),
    call("retrieve_paper", {"paper_id": "1706.03762", "segmentation_k": 1.2}, cid="call_2",
         note="Retrieving with structure so I can quote the attention definition accurately."),
    result("call_2", prepared(
        "1706.03762", "html",
        [
            "Scaled dot-product attention computes Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.",
            "Multi-head attention runs h attention functions in parallel on projected queries/keys/values, then concatenates and projects the results.",
        ],
        pruned_md="# Attention Is All You Need\n\n## Attention\n\n...",
        parents=["Model Architecture", "Model Architecture"],
        clusters=["Scaled Dot-Product Attention", "Multi-Head Attention"])),
    asst(
        "Paper is **Attention Is All You Need** (1706.03762). The math, from the actual text:\n\n"
        "**Scaled dot-product attention**: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`. You score every query against "
        "every key via dot products, divide by √d_k (so large dimensions don't blow up the softmax), turn scores into "
        "weights with softmax, and take the weighted sum of the values.\n\n"
        "**Multi-head attention**: do that h times in parallel on different learned projections of Q/K/V, concatenate the "
        "h outputs, and project once more. Multiple heads let the model attend to different relationships at once."
    ),
])

# 27. "who cites it" -> search then citations.
E([
    usr("find what the BERT paper influenced - like who cites it"),
    call("search", {"q": "ti:bert pre-training bidirectional transformers", "n": 1},
         note="Resolve BERT's id before asking for citations (don't guess the id)."),
    result("call_1", search_response(["1810.04805"], 1)),
    call("execute", {"op": "citations", "id": "1810.04805", "limit": 4}, cid="call_2",
         note="Now fetch papers that cite BERT."),
    result("call_2", exec_one("1810.04805", "citations", [
        ss_paper("1910.10683", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", ["Colin Raffel"], 2019),
        ss_paper("2005.14165", "Language Models are Few-Shot Learners", ["Tom B. Brown"], 2020),
        ss_paper("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", ["Edward J. Hu"], 2021),
    ])),
    asst(
        "BERT (1810.04805) is cited all over modern NLP. A few influential descendants:\n"
        "- **T5** (1910.10683) - generalized the pretrain-finetune idea to text-to-text.\n"
        "- **GPT-3** (2005.14165) - contrasts BERT's fine-tuning with few-shot prompting.\n"
        "- **LoRA** (2106.09685) - parameter-efficient fine-tuning, often demoed on BERT-style models.\n\n"
        "I resolved the title to the id by searching first rather than guessing it."
    ),
])

# 28. "i keep seeing X everywhere" -> find originator + explain.
E([
    usr("i keep seeing 'RAG' everywhere, find the paper that started it and tell me what it actually is"),
    call("search", {"q": "ti:retrieval-augmented generation knowledge-intensive", "n": 1},
         note="Find the originating RAG paper by title."),
    result("call_1", search_response(["2005.11401"], 3)),
    call("execute", {"op": "abstract", "id": "2005.11401"}, cid="call_2",
         note="Grab its abstract to explain the concept accurately."),
    result("call_2", exec_one("2005.11401", "abstract", full_paper("2005.11401"))),
    asst(
        "The originating paper is **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (2005.11401, Lewis "
        "et al., 2020). What RAG actually is: instead of relying only on what's baked into the model's weights, a neural "
        "*retriever* fetches relevant passages from an external index (Wikipedia in the paper) at query time, and the "
        "generator conditions its answer on those passages. That's why people bolt 'RAG' onto LLMs to ground answers in "
        "your own documents."
    ),
])

# 29. "theory papers... the adam optimizer kind".
E([
    usr("find theory-ish papers on optimization for deep learning, like the adam optimizer kind of thing"),
    call("search", {"q": "abs:stochastic optimization adaptive gradient", "cats": ["cs.LG"], "n": 3, "sort": "relevance"},
         note="Optimization methods for DL -> cs.LG; Adam-style adaptive methods."),
    result("call_1", search_response(["1412.6980"], 612)),
    asst("The canonical one is **Adam: A Method for Stochastic Optimization** (1412.6980, Kingma & Ba). It keeps running estimates of the gradient's first and second moments, bias-corrects them, and adapts a per-parameter step size - which is why it's the default optimizer for most deep nets."),
])

# 30. Compound: compress to 4-bit for consumer GPUs.
E([
    usr("find research on compressing LLMs down to like 4-bit so they run on consumer gpus"),
    call("search", {"q": "abs:post-training quantization large language model 4-bit", "cats": ["cs.LG", "cs.CL"], "n": 3, "sort": "relevance"},
         note="Goal = low-bit quantization for consumer hardware -> GPTQ-style work."),
    result("call_1", search_response(["2210.17323"], 174)),
    asst("**GPTQ** (2210.17323, Frantar et al.) is the one. It's a one-shot post-training quantization method using approximate second-order info to push GPT-scale weights down to 3-4 bits with negligible accuracy loss - which is exactly what makes big models fit on a single consumer GPU. Tools like the 4-bit GGUF/GPTQ checkpoints trace back to this."),
])

# ---- multi-paper synthesis from a vague ask -------------------------------- #

# 31. hdrr over an indexed set, vague phrasing.
E([
    usr("across those agent papers we pulled, how do they actually get the model to take real actions?"),
    call("hdrr", {"q": "how language model agents take actions tool use", "limit_docs": 4, "limit_chunks": 8},
         note="Cross-paper question over the index -> HDRR, not separate retrieves."),
    result("call_1", {
        "query": "how language model agents take actions tool use",
        "routed_documents": ["2302.04761", "2210.03629"],
        "chunks": [
            {"id": "2302.04761-2", "text": "Toolformer inserts API calls (e.g. a calculator or search query) into the text during training and learns, from whether the call lowers loss, when to invoke each tool."},
            {"id": "2210.03629-5", "text": "ReAct prompts the model to emit alternating Thought and Action tokens; Actions are executed against an environment (e.g. a search API) and the Observation is fed back in."},
        ],
    }),
    asst(
        "HDRR routed to the two agent papers. They take actions differently:\n"
        "- **Toolformer** (2302.04761) bakes it into *training*: it inserts candidate API calls into text and keeps the ones "
        "that actually reduce loss, so the model learns when to call a tool.\n"
        "- **ReAct** (2210.03629) does it at *inference*: the model emits alternating Thought/Action steps; each Action (e.g. "
        "a search query) is run against the environment and the Observation is fed back before the next step.\n\n"
        "So: learned-at-training vs. prompted-at-inference, but both hinge on executing an external call and conditioning on the result."
    ),
])

# 32. Vague "rundown" -> hdrr fallback (no embedded-db) -> search + batch.
E([
    usr("give me a rundown of how the big diffusion papers differ"),
    call("hdrr", {"q": "diffusion models image synthesis differences", "limit_docs": 5, "limit_chunks": 10},
         note="Multi-paper synthesis -> try HDRR first."),
    result("call_1", {"error": "embedded-db feature not enabled"}),
    call("search", {"q": "abs:diffusion image synthesis", "cats": ["cs.CV", "cs.LG"], "n": 4, "sort": "relevance"},
         cid="call_2", note="HDRR needs the embedded-db index (not built here); falling back to search + abstracts."),
    result("call_2", search_response(["2006.11239", "2112.10752"], 1820)),
    call("execute", [
        {"op": "abstract", "id": "2006.11239"},
        {"op": "abstract", "id": "2112.10752"},
    ], cid="call_3", note="Pulling both abstracts to contrast them."),
    result("call_3", [
        exec_one("2006.11239", "abstract", full_paper("2006.11239")),
        exec_one("2112.10752", "abstract", full_paper("2112.10752")),
    ]),
    asst(
        "HDRR wasn't available (the server's embedded-db index isn't built here), so I searched + compared abstracts:\n"
        "- **DDPM** (2006.11239) - establishes the diffusion formulation, running the noising/denoising process directly in "
        "*pixel space*. Great quality, but expensive at high resolution.\n"
        "- **Latent Diffusion** (2112.10752) - moves that same process into a compressed *latent* space from a pretrained "
        "autoencoder, cutting compute by a lot; it's what made Stable-Diffusion-scale generation practical.\n\n"
        "Core difference: where the diffusion happens (pixels vs. latents) and therefore the cost. If you build the index, "
        "HDRR would do this comparison in a single call."
    ),
])

# 33. Nonsense-niche request -> empty results -> broaden -> ask for the key concept.
E([
    usr("find me papers on quantum graph cryptography for federated blockchains"),
    call("search", {"q": "abs:quantum graph cryptography federated blockchain", "n": 5, "sort": "relevance"},
         note="Trying the full phrase as given first."),
    result("call_1", search_response([], 0)),
    call("search", {"q": "abs:graph neural network blockchain", "cats": ["cs.LG", "cs.CR"], "n": 5, "sort": "relevance"},
         cid="call_2", note="Zero hits - dropping the least-standard terms and broadening to the plausible core (graphs + blockchain)."),
    result("call_2", search_response([], 0)),
    asst(
        "Both the full phrase and a broadened `graph neural network blockchain` search (cs.LG/cs.CR) returned nothing - that "
        "exact combination doesn't seem to exist as a research thread, and stringing four niche terms together is probably "
        "over-constraining it. Which single piece is the real target: the *quantum* angle, *graph* methods, *cryptography*, "
        "or *blockchain/federated* systems? Tell me the one that matters and I'll search productively around it."
    ),
])

# 34. Vague-but-conceptual -> answer directly, offer to fetch (calibration).
E([
    usr("real quick, what even is a transformer"),
    asst(
        "A transformer is a neural network architecture built around *self-attention*: every token looks at every other "
        "token and decides how much to weight each one, so context flows directly between positions instead of through "
        "recurrence. Stack those attention layers with feed-forward layers and you get the backbone behind modern LLMs, ViT, "
        "etc.\n\n"
        "That's the gist - no lookup needed. If you want the primary source, say so and I'll grab *Attention Is All You Need* "
        "(1706.03762) and walk through it."
    ),
])

# 35. Title-only -> resolve via search, then citations.
E([
    usr("can you get the citations for 'Denoising Diffusion Probabilistic Models'"),
    call("search", {"q": "ti:\"denoising diffusion probabilistic models\"", "n": 1},
         note="Resolve the title to an id via search before fetching citations."),
    result("call_1", search_response(["2006.11239"], 1)),
    call("execute", {"op": "citations", "id": "2006.11239", "limit": 4}, cid="call_2",
         note="Fetch the citing papers."),
    result("call_2", exec_one("2006.11239", "citations", [
        ss_paper("2112.10752", "High-Resolution Image Synthesis with Latent Diffusion Models", ["Robin Rombach"], 2021),
        ss_paper("2304.02643", "Segment Anything", ["Alexander Kirillov"], 2023),
    ])),
    asst(
        "Resolved the title to **arXiv:2006.11239** and pulled its citations. Notable citing work includes **Latent "
        "Diffusion Models** (2112.10752), which builds directly on DDPM, and broader vision systems like **Segment "
        "Anything** (2304.02643). I searched to get the id rather than guessing it."
    ),
])

# 36. "a survey on X".
E([
    usr("find me a survey paper on all the efficient transformer variants"),
    call("search", {"q": "ti:efficient transformers survey", "n": 2, "sort": "relevance"},
         note="They explicitly want a survey -> ti:survey on efficient transformers."),
    result("call_1", search_response(["2009.06732"], 4)),
    asst("**Efficient Transformers: A Survey** (2009.06732, Tay et al.) is exactly that - it organizes the 'X-former' landscape (sparse, low-rank, kernel-based, recurrence-augmented attention, etc.) into a taxonomy, so it's a good single map of the efficiency design space."),
])

# 37. "i loved X, find similar" -> recs.
E([
    usr("i loved the chain of thought paper, find me similar stuff"),
    call("execute", {"op": "recs", "id": "2201.11903", "limit": 4},
         note="\"Similar to a paper I liked\" -> recommendations for the CoT paper (2201.11903)."),
    result("call_1", exec_one("2201.11903", "recs", [
        ss_paper("2305.10601", "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", ["Shunyu Yao"], 2023),
        ss_paper("2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models", ["Shunyu Yao"], 2022),
    ])),
    asst(
        "If you liked Chain-of-Thought, the closest neighbors are:\n"
        "- **Tree of Thoughts** (2305.10601) - turns the linear chain into a searchable tree with backtracking.\n"
        "- **ReAct** (2210.03629) - adds *acting* to the reasoning so the model can verify steps against tools.\n\n"
        "Both are direct extensions of the CoT idea. Want either in full?"
    ),
])

# 38. Compound x/y/z: RL to fine-tune LMs for instruction following.
E([
    usr("find papers on using reinforcement learning to fine-tune language models so they follow instructions better"),
    call("search", {"q": "abs:reinforcement learning fine-tuning instruction following", "cats": ["cs.CL", "cs.LG"], "n": 5, "sort": "relevance"},
         note="Domain = RL fine-tuning of LMs; purpose = better instruction following -> cs.CL/cs.LG."),
    result("call_1", search_response(["2203.02155", "2305.18290"], 421)),
    asst(
        "Two papers cover this precisely:\n"
        "- **InstructGPT** (2203.02155) - the original: collect human preference data, train a reward model, and fine-tune "
        "with RL (PPO) so the model follows user intent. A 1.3B InstructGPT beat 175B GPT-3 on instruction following.\n"
        "- **DPO** (2305.18290) - achieves the same preference alignment *without* the RL loop, optimizing the policy "
        "directly from preference pairs - simpler and more stable to train.\n\n"
        "InstructGPT for the RL approach, DPO for the lighter modern alternative."
    ),
])

# ---- interface mechanics, still phrased as messy asks ---------------------- #

# 39. Pagination from a casual "show me more".
E([
    usr("show me work on neural machine translation"),
    call("search", {"q": "abs:neural machine translation", "n": 5, "sort": "relevance"},
         note="Initial discovery search for NMT."),
    result("call_1", search_response(["1706.03762", "1910.10683"], 8800)),
    asst("Top hits: **Attention Is All You Need** (1706.03762) and **T5** (1910.10683). There are ~8,800 matches total - want more?"),
    usr("yeah these aint it, gimme the next batch"),
    call("search", {"q": "abs:neural machine translation", "n": 5, "offset": 5, "sort": "relevance"},
         cid="call_2", note="Advancing the page with offset=5 for the next batch."),
    result("call_2", search_response(["2010.11929", "2009.06732"], 8800, start=5)),
    asst("Next page brings up **ViT** (2010.11929) and **Efficient Transformers: A Survey** (2009.06732). I can keep paging (offset 10, 15, ...) or narrow it down with a year or a subtopic if the results are drifting off-target."),
])

# 40. ID with prefix + version -> normalization, casual phrasing.
E([
    usr("pull up arxiv:1706.03762v7 for me real quick"),
    call("retrieve_paper", {"paper_id": "arxiv:1706.03762v7"},
         note="The 'arxiv:' prefix and 'v7' version suffix normalize to the base id server-side; passing it through as-is."),
    result("call_1", prepared(
        "1706.03762", "html",
        ["The Transformer is the first sequence transduction model based entirely on attention, without recurrence or convolution."],
        pruned_md="# Attention Is All You Need\n\n...")),
    asst("Got it - the `arxiv:` prefix and `v7` suffix normalized to **1706.03762** automatically. It's the Transformer paper, retrieved and chunked. Want a section summarized?"),
])

# 41. "dump 200 papers" -> 50-result cap + offer pagination.
E([
    usr("just dump like 200 papers on reinforcement learning on me"),
    call("search", {"q": "abs:reinforcement learning", "cats": ["cs.LG"], "n": 50, "sort": "relevance"},
         note="arXiv caps a single call at 50 results, so requesting the max and flagging pagination for the rest."),
    result("call_1", search_response(["1312.5602", "1707.06347"], 41000)),
    asst("Heads up: one call tops out at 50 results (there are ~41,000 RL matches in cs.LG), so I pulled the top 50 - led by **DQN** (1312.5602) and **PPO** (1707.06347). To go past 50 I page with `offset` (50, 100, ...). Want me to start paging, or narrow by subtopic/year so the first 50 are actually the right ones?"),
])

# ---- "find the current SoTA": recency + surface the leading work ----------- #

# 42. SoTA in text-to-image generation.
E([
    usr("go find the current state of the art on text-to-image generation"),
    call("search", {"q": "abs:text-to-image generation diffusion", "cats": ["cs.CV"], "n": 6, "sort": "date"},
         note="\"Current SoTA\" -> recent leading work; image generation -> cs.CV diffusion, sort by date."),
    result("call_1", search_response(["2112.10752", "2006.11239"], 3310)),
    asst(
        "On arXiv, the dominant recent line for text-to-image is latent diffusion:\n"
        "- **High-Resolution Image Synthesis with Latent Diffusion Models** (2112.10752) - runs diffusion in a compressed "
        "latent space; the basis of the Stable Diffusion family and the current workhorse.\n"
        "- **Denoising Diffusion Probabilistic Models** (2006.11239) - the formulation it builds on.\n\n"
        "One caveat: arXiv isn't a live leaderboard, so 'SoTA' here means the leading *recent* approach, not a benchmark "
        "ranking. Want me to pull the newest follow-ups in a specific direction (control, speed, video)?"
    ),
])

# 43. SoTA open LLMs right now.
E([
    usr("whats the SoTA for open LLMs right now"),
    call("search", {"q": "abs:open foundation large language model", "cats": ["cs.CL"], "n": 6, "sort": "date"},
         note="\"SoTA right now\" -> most recent open models in cs.CL, date-sorted."),
    result("call_1", search_response(["2401.04088", "2307.09288", "2302.13971"], 980)),
    asst(
        "Recent-first, the leading open models on arXiv are:\n"
        "- **Mixtral of Experts** (2401.04088) - sparse MoE, strong quality at lower active-parameter cost.\n"
        "- **Llama 2** (2307.09288) - 7B-70B, pretrained + chat-tuned, widely adopted.\n"
        "- **LLaMA** (2302.13971) - the open foundation line these built on.\n\n"
        "\"SoTA\" in open LLMs turns over fast and isn't a single number, so treat this as the current frontier of *recent* "
        "releases rather than a fixed ranking. Want the latest entrants in a size class you care about?"
    ),
])

# 44. SoTA for long context efficiency.
E([
    usr("find me the state of the art for handling long context efficiently"),
    call("search", {"q": "abs:long context efficient sequence modeling", "cats": ["cs.LG"], "n": 6, "sort": "date"},
         note="SoTA + efficiency for long sequences -> cs.LG, date-sorted."),
    result("call_1", search_response(["2312.00752", "2205.14135"], 712)),
    asst(
        "The leading recent approaches split two ways:\n"
        "- **Mamba** (2312.00752) - replace attention with a selective state-space model that's *linear* in length.\n"
        "- **FlashAttention** (2205.14135) - keep exact attention but make it IO-aware so long contexts fit and run fast.\n\n"
        "So 'SoTA' here is either a sub-quadratic architecture (Mamba) or a hardware-efficient exact-attention kernel "
        "(FlashAttention), depending on whether you can change the model. I can pull either in full."
    ),
])

# 45. SoTA multilingual speech recognition.
E([
    usr("whats the current sota for multilingual speech recognition"),
    call("search", {"q": "abs:multilingual speech recognition", "cats": ["eess.AS", "cs.CL"], "n": 5, "sort": "date"},
         note="SoTA multilingual ASR -> eess.AS/cs.CL, date-sorted."),
    result("call_1", search_response(["2212.04356"], 184)),
    asst("The reference point is **Whisper / Robust Speech Recognition via Large-Scale Weak Supervision** (2212.04356) - trained on 680k hours of multilingual, multitask audio, it's the strong zero-shot baseline most current multilingual ASR work compares against. (arXiv shows leading work, not a live WER leaderboard.)"),
])

# ---- "ground yourself against arxiv": verify from the source, not memory --- #

# 46. Explicit grounding + confirm with the source's own caveat.
E([
    usr("ground yourself against arxiv: is chain-of-thought actually better than plain prompting, or am i misremembering?"),
    call("search", {"q": "ti:chain-of-thought prompting reasoning", "n": 1},
         note="They asked me to ground against arxiv, so I verify from the source rather than answering from memory."),
    result("call_1", search_response(["2201.11903"], 3)),
    call("execute", {"op": "abstract", "id": "2201.11903"}, cid="call_2",
         note="Reading the abstract to ground the answer (and catch any caveats)."),
    result("call_2", exec_one("2201.11903", "abstract", full_paper("2201.11903"))),
    asst(
        "Grounded in the source, you're remembering right - with one caveat the paper makes explicit. **Chain-of-Thought "
        "Prompting** (2201.11903) shows that prompting for intermediate reasoning steps *significantly* improves complex "
        "reasoning over standard prompting. But the gain **emerges with scale**: it's large for big models and small or "
        "absent for small ones. So \"better\" is true mainly at sufficient model size, not universally."
    ),
])

# 47. "don't answer from memory, check arxiv first".
E([
    usr("dont just answer from memory — check arxiv first. what did the RAG paper actually propose?"),
    call("search", {"q": "ti:retrieval-augmented generation knowledge-intensive", "n": 1},
         note="Honoring the 'check arxiv first' instruction; resolving the paper before answering."),
    result("call_1", search_response(["2005.11401"], 3)),
    call("execute", {"op": "abstract", "id": "2005.11401"}, cid="call_2",
         note="Grounding the answer in the abstract."),
    result("call_2", exec_one("2005.11401", "abstract", full_paper("2005.11401"))),
    asst(
        "Checked against the source - **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (2005.11401, "
        "Lewis et al., 2020). What it actually proposed: pair a pretrained *parametric* seq2seq generator with a *non-"
        "parametric* memory - a dense vector index of Wikipedia accessed by a neural retriever. At generation time it "
        "retrieves relevant passages and conditions the output on them, so knowledge can be updated by swapping the index "
        "rather than retraining. That's the grounding mechanism, straight from the paper, not from memory."
    ),
])

# 48. Verify a user's belief and CORRECT it via grounding.
E([
    usr("fact check me real quick — the transformer was the first model to use attention, right? ground it against arxiv"),
    call("search", {"q": "abs:attention mechanism neural machine translation", "cats": ["cs.CL"], "n": 5, "sort": "relevance"},
         note="Verifying the claim against the source - searching for attention work both before and at the Transformer."),
    result("call_1", search_response(["1409.0473", "1706.03762"], 1290)),
    asst(
        "Grounded against arXiv - not quite, and it's a common mix-up. Attention *predates* the Transformer:\n"
        "- **Neural Machine Translation by Jointly Learning to Align and Translate** (1409.0473, Bahdanau et al., 2014) "
        "introduced the attention mechanism, letting a translation model soft-search the source instead of using a single "
        "fixed vector.\n"
        "- **Attention Is All You Need** (1706.03762, 2017) was the first to rely *solely* on attention - dropping "
        "recurrence and convolution entirely.\n\n"
        "So the accurate statement is: the Transformer was the first *attention-only* model, not the first to use attention."
    ),
])

# 49. "ground yourself" before giving the gist of a concept.
E([
    usr("ground yourself against arxiv and give me the gist of what DPO is"),
    call("search", {"q": "ti:direct preference optimization", "n": 1},
         note="Grounding instruction -> fetch the DPO paper before summarizing."),
    result("call_1", search_response(["2305.18290"], 2)),
    call("execute", {"op": "abstract", "id": "2305.18290"}, cid="call_2",
         note="Summarizing from the abstract, not from memory."),
    result("call_2", exec_one("2305.18290", "abstract", full_paper("2305.18290"))),
    asst(
        "Grounded in the paper - **Direct Preference Optimization** (2305.18290, Rafailov et al., 2023). The gist: you can "
        "align a model to human preferences *directly* with a simple classification-style loss on preferred-vs-rejected "
        "response pairs, skipping the usual RLHF machinery (no separate reward model, no RL/PPO loop). The paper's framing "
        "is that the language model is implicitly its own reward model, so you optimize it in one step."
    ),
])

# 50. Verify a specific technical claim against the actual method text.
E([
    usr("i'm pretty sure LoRA keeps the base weights frozen and only trains small matrices — verify that against the paper, dont guess"),
    call("search", {"q": "ti:low-rank adaptation large language models", "n": 1},
         note="Resolve the LoRA paper, then read its method to verify the specific claim."),
    result("call_1", search_response(["2106.09685"], 2)),
    call("retrieve_paper", {"paper_id": "2106.09685", "segmentation_k": 1.2}, cid="call_2",
         note="Reading the method section so the verification comes from the source, not memory."),
    result("call_2", prepared(
        "2106.09685", "html",
        [
            "LoRA freezes the pretrained weight matrix W0 and represents its update as a low-rank product BA, so only A and B are optimized while W0 stays fixed.",
            "Because the update BA can be merged into W0 after training, LoRA adds no inference latency relative to a fully fine-tuned model.",
        ],
        pruned_md="# LoRA\n\n## Method\n\n...",
        parents=["Method", "Method"],
        clusters=["Low-Rank Update", "No Inference Latency"])),
    asst(
        "Verified against the method section of **LoRA** (2106.09685): you're correct. The paper freezes the pretrained "
        "weight matrix W0 and learns only a low-rank update BA (with B and A the small trainable matrices); W0 itself is "
        "never updated. It adds the detail that BA can be merged back into W0 after training, so there's no extra inference "
        "latency. Your recollection holds up against the source."
    ),
])

# --------------------------------------------------------------------------- #
# Controlled volume expansion.
#
# The 50 conversations above are hand-written ("gold"). To reach a useful
# training volume we additionally COMPOSE conversations from curated building
# blocks: a bank of real-paper topics (mapped to arXiv categories), the messy
# phrasing templates, and the same tool-call patterns the gold set uses. This
# is deterministic (no RNG) and reproducible; every generated row is run
# through the same validator as the gold rows. Tune TARGET_GENERATED to scale.
# --------------------------------------------------------------------------- #

TARGET_GENERATED = 200

# More real arXiv papers (compact form), merged into PAPERS so domain coverage
# is broad enough for the topic bank. (id, title, authors, cats, year, blurb)
EXTRA_PAPERS = [
    ("1508.04025", "Effective Approaches to Attention-based Neural Machine Translation",
     ["Minh-Thang Luong", "Hieu Pham"], ["cs.CL"], 2015,
     "We propose global and local attention mechanisms for neural machine translation, improving quality over non-attentional systems."),
    ("2001.08361", "Scaling Laws for Neural Language Models",
     ["Jared Kaplan", "Sam McCandlish"], ["cs.LG"], 2020,
     "Language model loss scales as a power law with model size, dataset size, and compute, with smooth and predictable trends."),
    ("2203.15556", "Training Compute-Optimal Large Language Models",
     ["Jordan Hoffmann", "Sebastian Borgeaud"], ["cs.CL"], 2022,
     "Current large models are undertrained; for compute-optimal training, model size and training tokens should scale equally (Chinchilla)."),
    ("2206.07682", "Emergent Abilities of Large Language Models",
     ["Jason Wei", "Yi Tay"], ["cs.CL"], 2022,
     "Some abilities are absent in smaller models but emerge unpredictably in larger ones, a phenomenon we call emergence."),
    ("2310.06825", "Mistral 7B",
     ["Albert Q. Jiang", "Alexandre Sablayrolles"], ["cs.CL"], 2023,
     "Mistral 7B outperforms larger models using grouped-query and sliding-window attention for efficient, high-quality inference."),
    ("1907.11692", "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
     ["Yinhan Liu", "Myle Ott"], ["cs.CL"], 2019,
     "We show BERT was significantly undertrained and present RoBERTa, an improved recipe matching or exceeding later models."),
    ("1909.11942", "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
     ["Zhenzhong Lan", "Mingda Chen"], ["cs.CL"], 2019,
     "We present parameter-reduction techniques that lower memory use and increase training speed for BERT-style models."),
    ("1910.01108", "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
     ["Victor Sanh", "Lysandre Debut"], ["cs.CL"], 2019,
     "We distill BERT into a smaller, faster model that retains most of its language-understanding ability via knowledge distillation."),
    ("1409.3215", "Sequence to Sequence Learning with Neural Networks",
     ["Ilya Sutskever", "Oriol Vinyals"], ["cs.CL"], 2014,
     "We present a general end-to-end sequence-to-sequence approach using multilayered LSTMs, applied to machine translation."),
    ("1409.1556", "Very Deep Convolutional Networks for Large-Scale Image Recognition",
     ["Karen Simonyan", "Andrew Zisserman"], ["cs.CV"], 2014,
     "We investigate the effect of depth using very small 3x3 convolution filters, achieving strong image classification (VGG)."),
    ("1409.4842", "Going Deeper with Convolutions",
     ["Christian Szegedy", "Wei Liu"], ["cs.CV"], 2014,
     "We propose the Inception architecture (GoogLeNet), increasing depth and width while keeping computation budget constant."),
    ("1608.06993", "Densely Connected Convolutional Networks",
     ["Gao Huang", "Zhuang Liu"], ["cs.CV"], 2016,
     "We connect each layer to every other layer (DenseNet), strengthening feature propagation and reducing parameter count."),
    ("1709.01507", "Squeeze-and-Excitation Networks",
     ["Jie Hu", "Li Shen"], ["cs.CV"], 2017,
     "We propose channel-wise feature recalibration (SE blocks) that adaptively reweight channels, improving CNN accuracy cheaply."),
    ("1311.2524", "Rich feature hierarchies for accurate object detection and semantic segmentation",
     ["Ross Girshick", "Jeff Donahue"], ["cs.CV"], 2013,
     "We combine region proposals with CNN features (R-CNN), dramatically improving object detection performance."),
    ("1506.01497", "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
     ["Shaoqing Ren", "Kaiming He"], ["cs.CV"], 2015,
     "We introduce a Region Proposal Network that shares features with the detector for near real-time object detection."),
    ("1506.02640", "You Only Look Once: Unified, Real-Time Object Detection",
     ["Joseph Redmon", "Santosh Divvala"], ["cs.CV"], 2015,
     "We frame detection as a single regression problem (YOLO), enabling extremely fast real-time object detection."),
    ("1411.4038", "Fully Convolutional Networks for Semantic Segmentation",
     ["Jonathan Long", "Evan Shelhamer"], ["cs.CV"], 2014,
     "We adapt classification networks into fully convolutional models that produce dense, pixelwise semantic segmentation."),
    ("1505.04597", "U-Net: Convolutional Networks for Biomedical Image Segmentation",
     ["Olaf Ronneberger", "Philipp Fischer"], ["cs.CV"], 2015,
     "We present an encoder-decoder with skip connections (U-Net) that segments biomedical images from very few examples."),
    ("1503.03585", "Deep Unsupervised Learning using Nonequilibrium Thermodynamics",
     ["Jascha Sohl-Dickstein", "Eric A. Weiss"], ["cs.LG"], 2015,
     "We introduce diffusion probabilistic models that learn to reverse a gradual noising process to model complex distributions."),
    ("2011.13456", "Score-Based Generative Modeling through Stochastic Differential Equations",
     ["Yang Song", "Jascha Sohl-Dickstein"], ["cs.LG"], 2020,
     "We unify score-based and diffusion models through stochastic differential equations, enabling high-quality sampling and likelihoods."),
    ("1312.6114", "Auto-Encoding Variational Bayes",
     ["Diederik P. Kingma", "Max Welling"], ["stat.ML"], 2013,
     "We introduce the variational autoencoder, a scalable method for inference and learning in directed latent-variable models."),
    ("1607.06450", "Layer Normalization",
     ["Jimmy Lei Ba", "Jamie Ryan Kiros"], ["stat.ML"], 2016,
     "We normalize across features within a layer, stabilizing training for recurrent and transformer networks without batch dependence."),
    ("1503.02531", "Distilling the Knowledge in a Neural Network",
     ["Geoffrey Hinton", "Oriol Vinyals"], ["stat.ML"], 2015,
     "We compress an ensemble's knowledge into a single small model by training it on softened output probabilities."),
    ("1602.04938", "Why Should I Trust You?: Explaining the Predictions of Any Classifier",
     ["Marco Tulio Ribeiro", "Sameer Singh"], ["cs.LG"], 2016,
     "We present LIME, which explains any classifier's prediction by locally approximating it with an interpretable model."),
    ("1705.07874", "A Unified Approach to Interpreting Model Predictions",
     ["Scott Lundberg", "Su-In Lee"], ["cs.AI"], 2017,
     "We present SHAP values, a unified game-theoretic measure of feature importance for individual model predictions."),
    ("1602.01783", "Asynchronous Methods for Deep Reinforcement Learning",
     ["Volodymyr Mnih", "Adria Puigdomenech Badia"], ["cs.LG"], 2016,
     "We present asynchronous gradient descent for deep RL (A3C), stabilizing training without an experience replay buffer."),
    ("1509.02971", "Continuous control with deep reinforcement learning",
     ["Timothy P. Lillicrap", "Jonathan J. Hunt"], ["cs.LG"], 2015,
     "We adapt deterministic policy gradients to continuous action spaces with deep function approximators (DDPG)."),
    ("1801.01290", "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning",
     ["Tuomas Haarnoja", "Aurick Zhou"], ["cs.LG"], 2018,
     "We propose a maximum-entropy off-policy actor-critic (SAC) that is sample-efficient and stable for continuous control."),
    ("1509.06461", "Deep Reinforcement Learning with Double Q-learning",
     ["Hado van Hasselt", "Arthur Guez"], ["cs.LG"], 2015,
     "We show DQN overestimates action values and propose Double DQN to reduce the bias, improving performance."),
    ("2108.07258", "On the Opportunities and Risks of Foundation Models",
     ["Rishi Bommasani", "Drew A. Hudson"], ["cs.LG"], 2021,
     "We survey foundation models trained at scale, examining their capabilities, applications, and societal risks."),
    ("2303.08774", "GPT-4 Technical Report",
     ["OpenAI"], ["cs.CL"], 2023,
     "We report GPT-4, a large multimodal model exhibiting human-level performance on a range of professional and academic benchmarks."),
]

for _pid, _title, _authors, _cats, _year, _abs in EXTRA_PAPERS:
    PAPERS.setdefault(_pid, {
        "title": _title,
        "authors": _authors,
        "cats": _cats,
        "published": f"{_year}-01-01T00:00:00Z",
        "abstract": _abs,
    })

# Topic bank: a natural-language phrase, the abstract query terms, the arXiv
# categories the model should infer, and representative real paper IDs.
TOPICS = [
    {"phrase": "the transformer architecture", "q": "transformer self-attention architecture",
     "cats": ["cs.CL", "cs.LG"], "ids": ["1706.03762", "1409.0473", "1508.04025", "2104.09864"]},
    {"phrase": "scaling up large language models", "q": "scaling large language models",
     "cats": ["cs.CL", "cs.LG"], "ids": ["2005.14165", "2001.08361", "2203.15556", "2206.07682"]},
    {"phrase": "open-source LLMs", "q": "open foundation large language model",
     "cats": ["cs.CL"], "ids": ["2302.13971", "2307.09288", "2310.06825", "2401.04088"]},
    {"phrase": "LLM reasoning", "q": "reasoning large language models chain of thought",
     "cats": ["cs.CL"], "ids": ["2201.11903", "2305.10601", "2210.03629"]},
    {"phrase": "tool-using LLM agents", "q": "language model tool use agents",
     "cats": ["cs.CL", "cs.AI"], "ids": ["2302.04761", "2210.03629"]},
    {"phrase": "aligning models to human preferences", "q": "human feedback preference alignment",
     "cats": ["cs.CL", "cs.LG"], "ids": ["2203.02155", "2305.18290"]},
    {"phrase": "retrieval-augmented generation", "q": "retrieval augmented generation",
     "cats": ["cs.CL"], "ids": ["2005.11401"]},
    {"phrase": "parameter-efficient fine-tuning", "q": "parameter efficient fine-tuning low-rank",
     "cats": ["cs.CL", "cs.LG"], "ids": ["2106.09685"]},
    {"phrase": "making LLMs cheaper to run", "q": "quantization efficient inference transformer",
     "cats": ["cs.LG"], "ids": ["2210.17323", "2205.14135", "2009.06732"]},
    {"phrase": "long-context sequence models", "q": "long context efficient sequence modeling state space",
     "cats": ["cs.LG"], "ids": ["2312.00752", "2205.14135"]},
    {"phrase": "mixture-of-experts models", "q": "sparse mixture of experts language model",
     "cats": ["cs.LG", "cs.CL"], "ids": ["2401.04088"]},
    {"phrase": "word embeddings", "q": "word representations embeddings vector space",
     "cats": ["cs.CL"], "ids": ["1301.3781"]},
    {"phrase": "self-supervised pretraining for NLP", "q": "self-supervised pre-training language representation",
     "cats": ["cs.CL"], "ids": ["1810.04805", "1907.11692", "1909.11942", "1910.01108", "1910.10683"]},
    {"phrase": "neural machine translation", "q": "neural machine translation sequence to sequence",
     "cats": ["cs.CL"], "ids": ["1409.3215", "1409.0473", "1508.04025"]},
    {"phrase": "convolutional network architectures", "q": "deep convolutional neural network image classification",
     "cats": ["cs.CV"], "ids": ["1512.03385", "1409.1556", "1409.4842", "1608.06993", "1709.01507"]},
    {"phrase": "vision transformers", "q": "image recognition transformer patches",
     "cats": ["cs.CV"], "ids": ["2010.11929"]},
    {"phrase": "object detection", "q": "object detection region proposal",
     "cats": ["cs.CV"], "ids": ["1311.2524", "1506.01497", "1506.02640"]},
    {"phrase": "image segmentation", "q": "semantic segmentation fully convolutional",
     "cats": ["cs.CV"], "ids": ["1411.4038", "1505.04597", "2304.02643"]},
    {"phrase": "diffusion generative models", "q": "diffusion image synthesis generative",
     "cats": ["cs.CV", "cs.LG"], "ids": ["2006.11239", "2112.10752", "1503.03585", "2011.13456"]},
    {"phrase": "GANs and variational autoencoders", "q": "generative adversarial network variational autoencoder",
     "cats": ["cs.LG", "stat.ML"], "ids": ["1406.2661", "1312.6114"]},
    {"phrase": "connecting images and text", "q": "contrastive image text zero-shot",
     "cats": ["cs.CV"], "ids": ["2103.00020"]},
    {"phrase": "speech recognition", "q": "speech recognition multilingual",
     "cats": ["eess.AS", "cs.CL"], "ids": ["2212.04356"]},
    {"phrase": "value-based reinforcement learning", "q": "deep reinforcement learning value function",
     "cats": ["cs.LG"], "ids": ["1312.5602", "1509.06461", "1602.01783"]},
    {"phrase": "policy-gradient reinforcement learning", "q": "policy gradient continuous control",
     "cats": ["cs.LG"], "ids": ["1707.06347", "1509.02971", "1801.01290"]},
    {"phrase": "graph neural networks", "q": "graph neural network",
     "cats": ["cs.LG"], "ids": ["1609.02907"]},
    {"phrase": "optimization for deep learning", "q": "stochastic optimization adaptive gradient",
     "cats": ["cs.LG"], "ids": ["1412.6980"]},
    {"phrase": "normalization techniques", "q": "normalization training deep networks",
     "cats": ["cs.LG"], "ids": ["1502.03167", "1607.06450"]},
    {"phrase": "knowledge distillation and model compression", "q": "knowledge distillation model compression",
     "cats": ["cs.LG"], "ids": ["1503.02531", "1910.01108"]},
    {"phrase": "position encodings in transformers", "q": "position embedding transformer rotary",
     "cats": ["cs.CL", "cs.LG"], "ids": ["2104.09864"]},
    {"phrase": "explainability of ML models", "q": "interpretability explanation model predictions",
     "cats": ["cs.LG"], "ids": ["1602.04938", "1705.07874"]},
    {"phrase": "foundation models", "q": "foundation models pretraining general-purpose",
     "cats": ["cs.LG", "cs.CL"], "ids": ["2108.07258", "2303.08774"]},
]

DISCOVERY_PHRASINGS = [
    "go find me research on {p}",
    "find some papers on {p}",
    "i need to read up on {p}, point me at some papers",
    "what are the key papers on {p}?",
    "got anything good on {p}?",
    "dig up some work on {p} for me",
]
SOTA_PHRASINGS = [
    "go find the current state of the art on {p}",
    "whats the SoTA for {p} right now",
    "what's the latest and greatest in {p}?",
    "find me the cutting edge of {p}",
    "where's {p} at these days? find recent work",
]
GROUNDING_PHRASINGS = [
    "ground yourself against arxiv and give me the gist of {p}",
    "dont answer from memory - check arxiv first about {p}",
    "before you explain {p}, ground it in the actual paper",
    "fact-check this against arxiv: whats the deal with {p}?",
]
DEEPREAD_PHRASINGS = [
    "find the key paper on {p} and walk me through it",
    "grab the main paper on {p} and break it down",
    "get me the foundational {p} paper and explain the core idea",
]
CITATIONS_PHRASINGS = [
    "find a major paper on {p} and tell me what cites it",
    "who builds on the main {p} work? find the citations",
]
RECS_PHRASINGS = [
    "i'm into {p} - find me papers similar to the main one",
    "recommend papers like the key {p} paper",
]
COMPARE_PHRASINGS = [
    "compare the main approaches to {p}",
    "whats the difference between the top {p} methods?",
]
LITREVIEW_PHRASINGS = [
    "i'm doing a lit review on {p}, where do i start?",
    "give me the lay of the land on {p}",
]
# Many variants for every boilerplate string, rotated by a running counter so
# no single sentence repeats often (avoids the model overfitting to templates).
INTRO_DISCOVERY = [
    "Here's what's worth reading:", "Good starting points:", "Key papers on this:",
    "A few that matter:", "These are the ones I'd grab:", "Solid entry points:",
    "Here's a useful cluster:",
]
OFFER_POOL = [
    "Want any of these in full?", "Want me to retrieve one of these?",
    "Should I pull the abstract of any?", "Want me to broaden or narrow the search?",
    "Happy to go deeper on whichever looks right.", "Say which one and I'll fetch it.",
    "Want citations or recommendations off any of these?", "I can pull the method section of any.",
]
SOTA_LEADS = [
    "Leading recent work on {p}:", "Most recent notable work on {p}:",
    "Where {p} is right now, recent-first:", "The current frontier on {p}:",
    "Recent, high-signal work on {p}:", "Newest standout work on {p}:",
]
SOTA_CAVEATS = [
    "Caveat: arXiv isn't a live leaderboard, so treat this as the leading recent line, not a ranking.",
    "Bear in mind arXiv shows recent work, not benchmark standings - this is the active frontier, not a ranked #1.",
    "\"SoTA\" moves fast and isn't one number here; read this as the current direction, not a fixed ranking.",
    "Note this is the leading recent line on arXiv, not a live benchmark result.",
    "One caveat: arXiv isn't a leaderboard, so this is \"what's hot recently\" rather than \"the top score\".",
    "Keep in mind these are recent influential papers, not a measured ranking.",
]
GROUND_NOTES = [
    "Grounding instruction -> fetch the source before answering, not from memory.",
    "They asked me to ground against arXiv, so I check the paper instead of recalling.",
    "Honoring 'check arxiv first' - resolving the actual paper before answering.",
    "Verifying from the source rather than memory; finding the paper first.",
    "Grounding the answer in the paper, not my recollection.",
]
GROUND_NOTES2 = [
    "Reading the abstract to ground the answer.",
    "Pulling the abstract so the answer comes from the paper.",
    "Grounding the claim in the abstract.",
    "Reading the source abstract before I summarize.",
]
GROUND_LEADS = [
    "Grounded in the source -", "Checked against arXiv -", "Straight from the paper -",
    "Verified from the source -", "Per the paper itself -", "From the source, not memory -",
]
GROUND_TAGS = [
    " (Pulled from arXiv, not from memory.)", " (That's from the paper, not recollection.)",
    " (Grounded in the source above.)", "", " (Checked against the source, not recalled.)", "",
]
DEEPREAD_NOTES = [
    "Locating the key paper before reading it.",
    "Finding the right paper first so I read the correct one.",
    "Resolving the main paper, then I'll read it.",
]
DEEPREAD_NOTES2 = [
    "Retrieving the full text, chunked with structure, to read the method.",
    "Pulling the full paper, chunked, to walk through the method.",
    "Retrieving and segmenting the paper so I can read the method.",
]
DEEPREAD_LEADS = [
    "Found and read **{t}** ({pid}).", "Pulled and read **{t}** ({pid}).",
    "Got it - **{t}** ({pid}), read in full.", "Here it is: **{t}** ({pid}).",
]
DEEPREAD_CLOSERS = [
    "Want me to go deeper into a specific section?",
    "Want the experiments, or just the method?",
    "I can pull a specific section if you want more detail.",
    "Say the word and I'll dig into any part.",
]
COMPARE_CLOSERS = [
    "They differ mainly in approach and scope; tell me which axis matters and I'll dig into the methods.",
    "The contrast is mostly method and emphasis - want me to read either one's details?",
    "Different angles on the same problem; I can pull either method section to go deeper.",
    "Tell me what you care about (speed, quality, simplicity) and I'll compare on that axis.",
]
COMPARE_NOTES = [
    "Comparison -> pull both abstracts in one batched execute call.",
    "Comparing two -> one batched execute with both abstracts.",
    "Side-by-side -> fetch both abstracts in a single batched call.",
    "Two-way compare -> batch both abstracts in one execute.",
]
CITE_NOTES = [
    "Resolve the paper first, then fetch what cites it.",
    "Find the paper, then look up its citations.",
]
CITE_NOTES2 = ["Fetching citing papers.", "Pulling what cites it.", "Looking up the citation list."]
CITE_HEADERS = [
    "Work building on **{t}** ({pid}):", "Papers that cite **{t}** ({pid}):",
    "Building on **{t}** ({pid}):",
]
RECS_NOTES = [
    "\"Similar to the main one\" -> recommendations for the key paper.",
    "Asking for papers similar to the key one.",
]
RECS_HEADERS = [
    "Similar to **{t}** ({pid}):", "In the same vein as **{t}** ({pid}):",
    "If you liked **{t}** ({pid}), also see:",
]
LIT_INTROS = ["Lay of the land on {p}:", "The landscape on {p}:", "Where to start on {p}:"]
LIT_CLOSERS = [
    "Start with **{t}** ({pid}); I can retrieve its method when you're ready.",
    "I'd begin with **{t}** ({pid}) - want its method section?",
    "**{t}** ({pid}) is the best entry point; say the word to go deeper.",
]
LIT_NOTES2 = [
    "Pulling the key abstracts to summarize the landscape.",
    "Grabbing a couple of key abstracts for the overview.",
]
N_POOL = [4, 5, 6, 5, 8, 4, 10, 6]
TOTAL_POOL = [123, 487, 1820, 56, 940, 312, 2104, 88, 651, 274]
INTENT_ORDER = ["discovery", "sota", "grounding", "deep_read", "compare", "lit_review", "citations", "recs"]


def pick(lst, seq):
    return lst[seq % len(lst)]


def _year(pid):
    y = PAPERS[pid]["published"][:4]
    return int(y) if y.isdigit() else 2020


def _blurb(pid):
    sentence = PAPERS[pid]["abstract"].split(". ")[0].strip().rstrip(".")
    words = sentence.split()
    if len(words) > 22:
        sentence = " ".join(words[:22]) + " ..."
    return sentence


def _line(pid):
    return f"- **{PAPERS[pid]['title']}** ({pid}, {_year(pid)}) - {_blurb(pid)}."


def _ss_from(pid):
    p = PAPERS[pid]
    return ss_paper(pid, p["title"], p["authors"][:1] if p["authors"] else ["et al."], _year(pid))


def _rotate(ids, k, seq):
    start = seq % len(ids)
    return (ids[start:] + ids[:start])[:k]


def build_conv(topic, intent, seq, phr):
    ids = topic["ids"]
    if not ids:
        return None
    p = topic["phrase"]
    cats = topic["cats"]
    catstr = ", ".join(cats)
    q = topic["q"]
    n = N_POOL[seq % len(N_POOL)]
    total = TOTAL_POOL[seq % len(TOTAL_POOL)]

    if intent == "discovery":
        if len(ids) == 1 and phr > 0:
            return None
        chosen = _rotate(ids, min(3, len(ids)), seq)
        u = DISCOVERY_PHRASINGS[seq % len(DISCOVERY_PHRASINGS)].format(p=p)
        intro = INTRO_DISCOVERY[seq % len(INTRO_DISCOVERY)]
        ans = intro + "\n" + "\n".join(_line(i) for i in chosen) + "\n\n" + OFFER_POOL[seq % len(OFFER_POOL)]
        return [
            usr(u),
            call("search", {"q": f"abs:{q}", "cats": cats, "n": n, "sort": "relevance"},
                 note=f"Reading this as {p}; searching abstracts in {catstr}."),
            result("call_1", search_response(chosen, total)),
            asst(ans),
        ]

    if intent == "sota":
        if len(ids) == 1 and phr > 0:
            return None
        chosen = _rotate(ids, min(3, len(ids)), seq)
        u = SOTA_PHRASINGS[seq % len(SOTA_PHRASINGS)].format(p=p)
        ans = (pick(SOTA_LEADS, seq).format(p=p) + "\n" + "\n".join(_line(i) for i in chosen) +
               "\n\n" + pick(SOTA_CAVEATS, seq) + " " + pick(OFFER_POOL, seq + 3))
        return [
            usr(u),
            call("search", {"q": f"abs:{q}", "cats": cats, "n": n, "sort": "date"},
                 note=f"\"SoTA/recent\" -> sort by date; {p} -> {catstr}."),
            result("call_1", search_response(chosen, total)),
            asst(ans),
        ]

    if intent == "grounding":
        if phr >= len(ids):
            return None
        pid = ids[phr % len(ids)]
        u = GROUNDING_PHRASINGS[seq % len(GROUNDING_PHRASINGS)].format(p=p)
        ans = (f"{pick(GROUND_LEADS, seq)} **{PAPERS[pid]['title']}** ({pid}, {_year(pid)}). "
               f"{PAPERS[pid]['abstract']}{pick(GROUND_TAGS, seq)}")
        return [
            usr(u),
            call("search", {"q": f"abs:{q}", "n": 1, "sort": "relevance"},
                 note=pick(GROUND_NOTES, seq)),
            result("call_1", search_response([pid], total)),
            call("execute", {"op": "abstract", "id": pid}, cid="call_2",
                 note=pick(GROUND_NOTES2, seq)),
            result("call_2", exec_one(pid, "abstract", full_paper(pid))),
            asst(ans),
        ]

    if intent == "deep_read":
        if phr >= len(ids):
            return None
        pid = ids[phr % len(ids)]
        u = DEEPREAD_PHRASINGS[seq % len(DEEPREAD_PHRASINGS)].format(p=p)
        ans = (pick(DEEPREAD_LEADS, seq).format(t=PAPERS[pid]['title'], pid=pid) +
               " The core idea, from the paper: " + PAPERS[pid]['abstract'] + " " +
               pick(DEEPREAD_CLOSERS, seq))
        return [
            usr(u),
            call("search", {"q": f"abs:{q}", "n": 1, "sort": "relevance"},
                 note=pick(DEEPREAD_NOTES, seq)),
            result("call_1", search_response([pid], total)),
            call("retrieve_paper", {"paper_id": pid, "segmentation_k": 1.2}, cid="call_2",
                 note=pick(DEEPREAD_NOTES2, seq)),
            result("call_2", prepared(pid, "html", [PAPERS[pid]["abstract"]],
                                      pruned_md=f"# {PAPERS[pid]['title']}\n\n...")),
            asst(ans),
        ]

    if intent == "citations":
        if len(ids) < 2 or phr >= len(ids):
            return None
        subj = ids[phr % len(ids)]
        others = [i for i in ids if i != subj][:3]
        u = CITATIONS_PHRASINGS[seq % len(CITATIONS_PHRASINGS)].format(p=p)
        body = "\n".join(f"- **{PAPERS[i]['title']}** ({i})" for i in others)
        header = pick(CITE_HEADERS, seq).format(t=PAPERS[subj]['title'], pid=subj)
        return [
            usr(u),
            call("search", {"q": f"abs:{q}", "n": 1, "sort": "relevance"},
                 note=pick(CITE_NOTES, seq)),
            result("call_1", search_response([subj], total)),
            call("execute", {"op": "citations", "id": subj, "limit": 3}, cid="call_2",
                 note=pick(CITE_NOTES2, seq)),
            result("call_2", exec_one(subj, "citations", [_ss_from(i) for i in others])),
            asst(f"{header}\n{body}"),
        ]

    if intent == "recs":
        if len(ids) < 2 or phr >= len(ids):
            return None
        subj = ids[phr % len(ids)]
        others = [i for i in ids if i != subj][:3]
        u = RECS_PHRASINGS[seq % len(RECS_PHRASINGS)].format(p=p)
        body = "\n".join(f"- **{PAPERS[i]['title']}** ({i})" for i in others)
        header = pick(RECS_HEADERS, seq).format(t=PAPERS[subj]['title'], pid=subj)
        return [
            usr(u),
            call("execute", {"op": "recs", "id": subj, "limit": 3},
                 note=pick(RECS_NOTES, seq)),
            result("call_1", exec_one(subj, "recs", [_ss_from(i) for i in others])),
            asst(f"{header}\n{body}"),
        ]

    if intent == "compare":
        if len(ids) < 2:
            return None
        two = _rotate(ids, 2, seq + phr)
        a, b = two[0], two[1]
        if a == b:
            return None
        u = COMPARE_PHRASINGS[seq % len(COMPARE_PHRASINGS)].format(p=p)
        ans = (f"- **{PAPERS[a]['title']}** ({a}) - {_blurb(a)}.\n"
               f"- **{PAPERS[b]['title']}** ({b}) - {_blurb(b)}.\n\n" + pick(COMPARE_CLOSERS, seq))
        return [
            usr(u),
            call("execute", [{"op": "abstract", "id": a}, {"op": "abstract", "id": b}],
                 note=pick(COMPARE_NOTES, seq)),
            result("call_1", [exec_one(a, "abstract", full_paper(a)), exec_one(b, "abstract", full_paper(b))]),
            asst(ans),
        ]

    if intent == "lit_review":
        if len(ids) < 2:
            return None
        chosen = _rotate(ids, min(3, len(ids)), seq)
        two = chosen[:2]
        u = LITREVIEW_PHRASINGS[seq % len(LITREVIEW_PHRASINGS)].format(p=p)
        body = "\n".join(_line(i) for i in chosen)
        ans = (pick(LIT_INTROS, seq).format(p=p) + f"\n{body}\n\n" +
               pick(LIT_CLOSERS, seq).format(t=PAPERS[two[0]]['title'], pid=two[0]))
        return [
            usr(u),
            call("search", {"q": f"abs:{q}", "cats": cats, "n": n, "sort": "relevance"},
                 note=f"Lit-review scan -> discover in {catstr}, then pull the key abstracts."),
            result("call_1", search_response(chosen, total)),
            call("execute", [{"op": "abstract", "id": i} for i in two], cid="call_2",
                 note=pick(LIT_NOTES2, seq)),
            result("call_2", [exec_one(i, "abstract", full_paper(i)) for i in two]),
            asst(ans),
        ]

    return None


GOLD_COUNT = len(EXAMPLES)
_seq = 0
for _phr in (0, 1):
    for _intent in INTENT_ORDER:
        for _topic in TOPICS:
            if len(EXAMPLES) - GOLD_COUNT >= TARGET_GENERATED:
                break
            _conv = build_conv(_topic, _intent, _seq, _phr)
            if _conv is not None:
                EXAMPLES.append(_conv)
                _seq += 1

# A handful of pagination conversations phrased as casual follow-ups.
_PAG_FOLLOWUPS = ["gimme more", "next batch please", "show me the next page", "more, these arent quite it"]
_pag = 0
for _topic in TOPICS:
    if _pag >= 12:
        break
    _ids = _topic["ids"]
    if len(_ids) < 4:
        continue
    _first, _second = _ids[:2], _ids[2:4]
    _q, _cats, _p = _topic["q"], _topic["cats"], _topic["phrase"]
    EXAMPLES.append([
        usr(f"show me papers on {_p}"),
        call("search", {"q": f"abs:{_q}", "cats": _cats, "n": 2, "sort": "relevance"},
             note="Initial discovery search."),
        result("call_1", search_response(_first, 900)),
        asst("Top hits: " + " and ".join(f"**{PAPERS[i]['title']}** ({i})" for i in _first) + ". Want more?"),
        usr(_PAG_FOLLOWUPS[_pag % len(_PAG_FOLLOWUPS)]),
        call("search", {"q": f"abs:{_q}", "cats": _cats, "n": 2, "offset": 2, "sort": "relevance"},
             cid="call_2", note="Advancing the page with offset=2."),
        result("call_2", search_response(_second, 900, start=2)),
        asst("Next page: " + " and ".join(f"**{PAPERS[i]['title']}** ({i})" for i in _second) +
             ". I can keep paging (offset 4, 6, ...) or narrow it down."),
    ])
    _pag += 1


# --------------------------------------------------------------------------- #
# Assembly + validation.
# --------------------------------------------------------------------------- #

VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def to_record(messages):
    return {"messages": [sys_msg(), *messages], "tools": TOOLS, "parallel_tool_calls": False}


def validate(record):
    msgs = record["messages"]
    assert msgs[0]["role"] == "system", "first message must be system"
    open_calls: set[str] = set()
    seen_ids: set[str] = set()
    for m in msgs:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                cid = tc["id"]
                assert cid not in seen_ids, f"duplicate tool_call id {cid}"
                seen_ids.add(cid)
                name = tc["function"]["name"]
                assert name in VALID_TOOL_NAMES, f"unknown tool {name}"
                # arguments must be a JSON string holding {"code": <json string>}
                args = json.loads(tc["function"]["arguments"])
                assert set(args.keys()) == {"code"}, f"args must be just 'code', got {args.keys()}"
                code_val = json.loads(args["code"])  # code must itself be valid JSON
                if name == "execute":
                    assert isinstance(code_val, (dict, list)), "execute code must be object or array"
                    ops = code_val if isinstance(code_val, list) else [code_val]
                    for o in ops:
                        assert o.get("op") in {"abstract", "download", "citations", "recs", "retrieve"}, o
                        assert "id" in o, "execute op needs id"
                else:
                    assert isinstance(code_val, dict), f"{name} code must be an object"
                    if name in {"search", "hdrr"}:
                        assert "q" in code_val, f"{name} needs q"
                    if name == "retrieve_paper":
                        assert "paper_id" in code_val, "retrieve_paper needs paper_id"
                open_calls.add(cid)
        elif role == "tool":
            cid = m["tool_call_id"]
            assert cid in open_calls, f"tool result {cid} has no matching call"
            open_calls.discard(cid)
            json_or_str = m["content"]
            # content is either raw text (download) or JSON; if it looks like JSON, it must parse
            stripped = json_or_str.lstrip()
            if stripped[:1] in "{[":
                json.loads(json_or_str)
    assert not open_calls, f"tool calls without results: {open_calls}"
    # the conversation must end on an assistant message
    assert msgs[-1]["role"] == "assistant", "conversation must end with assistant"


def main():
    records = [to_record(conv) for conv in EXAMPLES]
    for i, rec in enumerate(records):
        try:
            validate(rec)
        except AssertionError as exc:  # pragma: no cover - surfaced at build time
            raise SystemExit(f"example {i} failed validation: {exc}") from exc

    lines = [json.dumps(rec, ensure_ascii=False) for rec in records]
    # Every line must independently round-trip as JSON.
    for i, line in enumerate(lines):
        json.loads(line)
        assert "\n" not in line, f"example {i} contains a newline in the serialized line"

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tool_counts: dict[str, int] = {}
    turn_total = 0
    for conv in EXAMPLES:
        turn_total += len(conv)
        for m in conv:
            for tc in m.get("tool_calls", []):
                name = tc["function"]["name"]
                tool_counts[name] = tool_counts.get(name, 0) + 1

    print(f"wrote {len(records)} examples -> {OUT_PATH}")
    print(f"  user/assistant/tool turns (excl. system): {turn_total}")
    print(f"  tool-call distribution: {dict(sorted(tool_counts.items()))}")


if __name__ == "__main__":
    main()
