# Generative AI Project MVP

This repository contains a runnable, modular Python MVP with YAML-based
configuration, retrieval-augmented generation (RAG), a persistent SQLite vector
store, Gemini, OpenAI, and Anthropic clients, and an optional local Hugging Face
Transformers model.

## Project layout

```text
config/                 Application and logging YAML configuration
examples/               Runnable completion example
src/llm/                Provider interface, clients, and provider factory
src/rag/                Document chunking, embeddings, SQLite vectors, and RAG
src/cognition/          Cognitive memory and contextual Wikipedia discovery
src/utils/wiki_extractor.py Wikipedia API extraction, learning levels, and JSON export
src/web/                Local browser chat interface and HTTP handler
src/prompt_engineering/ Prompt templates and chain
src/utils/              Logger and in-memory cache
src/handlers/           Application error handling
tests/                  Unit tests
main.py                 Application entry point
requirements.txt        Python dependencies
Dockerfile              Container image definition
```

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Gemini is the provider used by the standalone model-generation example
(`gemini-2.5-flash`). To configure its key without entering it into a terminal,
open the workspace `.env` file in VS Code and set
`GEMINI_API_KEY=your-new-key`. The application loads this file automatically
on each startup. `.env` is excluded by `.gitignore`; do not commit it. You can
start from `.env.example`.
The browser chat and `--ask` do not require this key. They search the official
Wikipedia API for every question, save the retrieved article into
`data/vectors-wikipedia.sqlite3`, and return attributed source excerpts without
calling Gemini. Local lexical embeddings (`hash`) maintain the index without
an embedding API key or quota. The standalone model-generation example uses the configured provider. To use
the local Flan-T5-Small model instead, set `default_provider: local` in
`config/model_config.yaml`. The first run downloads the model from Hugging Face;
after it is cached, inference runs on the configured local device (`-1` means
CPU, and `0` selects the first CUDA GPU). This uses local compute and memory.
The browser Wikipedia chat continues returning the complete sourced article
text directly from Wikipedia; it does not send that text to an LLM.
To run the standalone local Wikipedia Q&A example, use
`.venv/bin/python examples/run_local_ai.py`. Wikipedia search still requires
network access, but the question and retrieved context are passed to the local
model rather than a hosted LLM API.
When the configured provider is used for generated answers, matching lexicon
entries are added to the prompt as supplemental memory. The lexicon does not
change model weights and is not fine-tuning; factual answers must remain
grounded in cited source excerpts.

The standalone RAG example indexes a short document and answers a question
using retrieved chunks. The web chat uses the configured local database path.
The `rag` section also controls chunk size, overlap, and retrieval count.

## Progressive Wikipedia RAG

Fetch and index a Portuguese Wikipedia article, then answer with excerpts from
the official Wikipedia API:

```bash
python main.py --wiki-topic "Inteligência artificial" \
  --ask "O que é inteligência artificial?" --level basico
```

Use `--level intermediario` or `--level avancado` to select matching sections.
Each question is searched live through the official MediaWiki API, and the
article is added or refreshed in the local SQLite index. Extracted sections
are saved with source and CC BY-SA attribution metadata in `data/prompts/`.
Section levels are heuristic: the introduction is basic, recognizable
mechanism/history/application sections are intermediate, and advanced topics
or later sections are advanced.

To ingest the curated universal curriculum (science, society, computing, and
humanities) into the three level directories and the vector database:

```bash
python main.py --universal-curriculum
```

The starter catalog includes 11 foundational articles and four advanced
related articles (machine learning, neural networks, quantum mechanics, and
game theory). Files are written to
`data/universal_curriculum/nivel_1_basico/`,
`data/universal_curriculum/nivel_2_intermediario/`, and
`data/universal_curriculum/nivel_3_avancado/` (one JSON per article and level;
some files can have an empty `sections` list). Advanced related articles are
placed in the advanced track; sections in the foundational articles use the
extractor's title-and-position heuristic. Override the output directory with
`--curriculum-output PATH`. The command accesses Wikipedia and indexes text
using local lexical embeddings, without Gemini embedding requests.

This process builds a local knowledge index; it does not fine-tune or train
language-model weights.

`--ask` performs a live Wikipedia search for each question and refreshes the
local index with the matching article:

```bash
python main.py --ask "O que é o método científico?" --level basico
python main.py --ask "Como funciona a aprendizagem de máquina?" --level avancado
```

Run the unit tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

The MVP's browser chat interface is a local-only HTTP service, not a public
production API. Start it with:

```bash
python main.py --chat
```

Then open `http://127.0.0.1:8000`. For Codespaces, open the forwarded port 8000
in the Ports panel. Use `--port 8080` to choose a different port. The server
binds to loopback only; requests from unrelated browser origins are rejected.
Each question triggers a live Wikipedia search and adds the matching article
to the local index. Replies contain source excerpts and attribution, without
Gemini generation.
The reply displays all text sections retrieved for the selected article; it
does not truncate section length or omit sections based on the selected level.
The level selector still controls the cognitive metadata recorded for the
interaction.
When an article has a Wikimedia thumbnail, the chat also shows the image with
alt text and a link to the file description/license page. Images are optional:
articles without a thumbnail still return their complete text. The browser
loads the image directly from `upload.wikimedia.org`; the chat only renders
HTTPS images from that Wikimedia host and does not use HTML injection.

The chat also estimates a learning level from the question, assesses whether
local context is available, and stores the sourced interaction in
`data/cognitive_memory.sqlite3` for up to seven days. The cognitive assessment
uses simple heuristics; it is not model training or a guarantee that a response
is correct. Semantic Wikipedia discovery extracts query concepts, searches the
official MediaWiki full-text API using the question and its key terms, and
reranks candidate snippets locally with typo-tolerant term matching. Short
anaphoric follow-ups can use prior question context; a new topical question is
searched independently. This is contextual lexical matching, not
embedding-based semantic similarity. The local SQLite vector index retains
retrieved Wikipedia passages separately.
For each successful Wikipedia chat answer, the configured
`cognition.vocabulary_path` (default `data/personal_lexicon.json`) also stores
the article title, lead text, question context, source URL, exposure count, and
highest learning level observed from the selected or inferred level. Repeated
exposure increases the count; it does not automatically certify mastery. For
generated answers, relevant records are retrieved into the prompt as local
context. The JSON lexicon persists across runs; this contextual learning does
not update language-model weights or constitute fine-tuning.
Try the standalone vocabulary demonstration with
`.venv/bin/python examples/run_vocabulary_growth.py`; it fetches a Portuguese
Wikipedia article and records its introduction and source in the same lexicon.
The browser keeps a short recent conversation in memory so follow-up questions
can refer to prior turns; this history is sent with chat requests but is not
persisted after the page is reloaded.
SQLite vector search uses exact cosine similarity and is intended for local MVP
workloads, not large-scale concurrent production traffic.