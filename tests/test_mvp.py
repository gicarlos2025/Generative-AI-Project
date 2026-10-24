import json
import os
import tempfile
import time
from types import ModuleType
import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import main as app_main
from examples.run_local_ai import run_local_pipeline
from examples.run_vocabulary_growth import run_vocabulary_demo
from src.llm.anthropic_client import AnthropicClient
from src.llm.base import BaseLLMClient
from src.llm.factory import create_llm_client
from src.llm.gemini_client import GeminiClient
from src.llm.gpt_client import GPTClient
from src.llm.local_engine import LocalAIEngine
from src.prompt_engineering.chainer import PromptChain
from src.prompt_engineering.templates import PromptTemplate
from src.rag.chunking import chunk_text
from src.rag.embeddings import GeminiEmbeddingProvider, HashEmbeddingProvider, create_embedding_provider
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import SQLiteVectorStore
from src.cognition.cognitive_learner import CognitiveLearnerSkill, SQLiteCognitiveMemory
from src.cognition.semantic_wiki_discoverer import SemanticWikiDiscovererSkill
from src.cognition.vocabulary_builder import PersonalVocabularyBuilder
from src.utils.cache import SimpleCache
from src.utils.universal_curriculum import (
    CurriculumTopic,
    ingest_universal_curriculum,
)
from src.utils.secrets import load_local_secrets
from src.utils.wiki_extractor import WikipediaArticle, WikipediaExtractor, WikipediaSection
from src.utils.wikipedia_chat import WikipediaChat, _is_context_dependent
from src.web.chat_server import create_chat_server


class CountingClient(BaseLLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt = ""

    @property
    def is_configured(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return f"generated: {prompt}"


class MvpTests(unittest.TestCase):
    def test_local_secrets_load_keys_without_logging_or_overriding_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secrets_file = Path(directory) / ".env"
            secrets_file.write_text(
                "# local keys\nOPENAI_API_KEY='file-key-value'\nANTHROPIC_API_KEY=\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "existing-environment-key"}, clear=True):
                load_local_secrets(secrets_file)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "existing-environment-key")
                self.assertNotIn("ANTHROPIC_API_KEY", os.environ)

            with patch.dict(os.environ, {}, clear=True):
                load_local_secrets(secrets_file)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "file-key-value")
                self.assertNotIn("ANTHROPIC_API_KEY", os.environ)

    def test_local_secrets_load_gemini_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secrets_file = Path(directory) / ".env"
            secrets_file.write_text("GEMINI_API_KEY=gemini-test-key\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                load_local_secrets(secrets_file)

                self.assertEqual(os.environ["GEMINI_API_KEY"], "gemini-test-key")

    def test_local_secrets_reports_invalid_line_number_not_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secrets_file = Path(directory) / ".env"
            secrets_file.write_text("not-an-assignment sensitive-value\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1") as error:
                load_local_secrets(secrets_file)

            self.assertNotIn("sensitive-value", str(error.exception))

    def test_cache_expires_after_ttl(self) -> None:
        cache = SimpleCache(default_ttl=0)
        cache.set("key", "value")
        time.sleep(0.001)
        self.assertIsNone(cache.get("key"))

    def test_cognitive_memory_persists_sourced_answers_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "cognition.sqlite3")
            learner = CognitiveLearnerSkill(SQLiteCognitiveMemory(database_path))
            perception = learner.perceive("Explique a arquitetura de um modelo")
            assessment = learner.evaluate_cognition(perception)
            self.assertTrue(assessment["has_knowledge_gap"])
            self.assertEqual(assessment["recommended_action"], "search_wikipedia")

            learner.learn_and_store(
                "Explique a arquitetura de um modelo",
                "Resumo baseado em fontes verificadas.",
                level=perception["detected_complexity"],
                sources=["https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial"],
                assessment=assessment,
            )

            recalled = CognitiveLearnerSkill(SQLiteCognitiveMemory(database_path)).recall(
                "EXPLIQUE A ARQUITETURA DE UM MODELO"
            )

        self.assertIsNotNone(recalled)
        self.assertEqual(recalled["level"], "avancado")
        self.assertIn("pt.wikipedia.org", recalled["sources"][0])
        self.assertTrue(recalled["assessment"]["has_knowledge_gap"])

    def test_personal_vocabulary_persists_exposures_and_maturity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lexicon_path = Path(directory) / "nested" / "personal_lexicon.json"
            builder = PersonalVocabularyBuilder(lexicon_path)
            builder.acquire_sign(
                term="Inteligência Artificial",
                meaning="Campo da computação que estuda sistemas inteligentes.",
                context="O que é inteligência artificial?",
                level="basico",
                source="https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial",
            )
            builder.acquire_sign(
                term="inteligencia artificial",
                meaning="Campo da computação que estuda sistemas inteligentes.",
                context="Como funciona a inteligência artificial?",
                level="avancado",
                source="https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial",
            )

            restored = PersonalVocabularyBuilder(lexicon_path)
            entry = restored.consult_lexicon("INTELIGÊNCIA ARTIFICIAL")

        self.assertIsNotNone(entry)
        self.assertEqual(entry["term"], "Inteligência Artificial")
        self.assertEqual(entry["exposure_count"], 2)
        self.assertEqual(entry["maturity_level"], "avancado")
        self.assertEqual(len(entry["meanings"]), 1)
        self.assertEqual(len(entry["contexts"]), 2)
        self.assertEqual(len(entry["sources"]), 1)

    def test_personal_vocabulary_rejects_corrupt_file_without_resetting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lexicon_path = Path(directory) / "personal_lexicon.json"
            lexicon_path.write_text("{invalid json", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Could not load personal lexicon"):
                PersonalVocabularyBuilder(lexicon_path)

            self.assertEqual(lexicon_path.read_text(encoding="utf-8"), "{invalid json")

    def test_personal_vocabulary_normalizes_legacy_accented_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lexicon_path = Path(directory) / "personal_lexicon.json"
            lexicon_path.write_text(
                json.dumps(
                    {
                        "inteligência artificial": {
                            "term": "Inteligência Artificial",
                            "meanings": ["Conceito"],
                            "contexts": ["Uma pesquisa"],
                            "maturity_level": "basico",
                            "exposure_count": 1,
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            entry = PersonalVocabularyBuilder(lexicon_path).consult_lexicon(
                "inteligencia artificial"
            )

        self.assertIsNotNone(entry)
        self.assertEqual(entry["exposure_count"], 1)
        self.assertEqual(entry["sources"], [])

    def test_wikipedia_chat_consolidates_article_into_personal_vocabulary(self) -> None:
        article = WikipediaArticle(
            title="Algoritmo",
            language="pt",
            source_url="https://pt.wikipedia.org/wiki/Algoritmo",
            retrieved_at="2026-10-24T00:00:00Z",
            sections=[
                WikipediaSection(
                    "Introdução",
                    "Um algoritmo é uma sequência finita de instruções para resolver um problema.",
                    "basico",
                    0,
                )
            ],
        )
        discoverer = type(
            "FakeDiscoverer",
            (),
            {
                "fetch_best_article": lambda self, query: article,
                "fetch_article_image": lambda self, title: None,
            },
        )()

        with tempfile.TemporaryDirectory() as directory:
            pipeline = RAGPipeline(
                SQLiteVectorStore(
                    Path(directory) / "vectors.sqlite3",
                    HashEmbeddingProvider(),
                ),
                CountingClient(),
            )
            vocabulary = PersonalVocabularyBuilder(
                Path(directory) / "personal_lexicon.json"
            )
            chat = WikipediaChat(
                extractor=WikipediaExtractor(language="pt"),
                pipeline=pipeline,
                wiki_discoverer=discoverer,
                vocabulary_builder=vocabulary,
            )

            chat.answer("O que é um algoritmo?", learning_level="basico")
            chat.answer("Como funciona um algoritmo?", learning_level="intermediario")
            entry = PersonalVocabularyBuilder(
                Path(directory) / "personal_lexicon.json"
            ).consult_lexicon("algoritmo")

        self.assertEqual(entry["exposure_count"], 2)
        self.assertEqual(entry["maturity_level"], "intermediario")
        self.assertEqual(
            entry["meanings"],
            ["Um algoritmo é uma sequência finita de instruções para resolver um problema."],
        )
        self.assertEqual(len(entry["contexts"]), 2)
        self.assertEqual(entry["sources"], ["https://pt.wikipedia.org/wiki/Algoritmo"])

    def test_vocabulary_demo_records_wikipedia_lead_and_source(self) -> None:
        article = WikipediaArticle(
            title="Algoritmo",
            language="pt",
            source_url="https://pt.wikipedia.org/wiki/Algoritmo",
            retrieved_at="2026-10-24T00:00:00Z",
            sections=[
                WikipediaSection("Introdução", "Definição obtida da Wikipédia.", "basico", 0)
            ],
        )
        record = {
            "term": "Algoritmo",
            "maturity_level": "basico",
            "exposure_count": 1,
            "meanings": ["Definição obtida da Wikipédia."],
            "contexts": ["Pesquisa: Algoritmo"],
            "sources": ["https://pt.wikipedia.org/wiki/Algoritmo"],
        }

        with (
            patch("examples.run_vocabulary_growth.PersonalVocabularyBuilder") as builder_type,
            patch("examples.run_vocabulary_growth.SemanticWikiDiscovererSkill") as discoverer_type,
            redirect_stdout(StringIO()),
        ):
            builder_type.return_value.consult_lexicon.return_value = record
            discoverer_type.return_value.fetch_best_article.return_value = article
            result = run_vocabulary_demo()

        builder_type.return_value.acquire_sign.assert_called_once_with(
            term="Algoritmo",
            meaning="Definição obtida da Wikipédia.",
            context="Pesquisa: Algoritmo",
            level="basico",
            source="https://pt.wikipedia.org/wiki/Algoritmo",
        )
        self.assertEqual(result, record)

    def test_prompt_template_reports_missing_variables(self) -> None:
        template = PromptTemplate("Hello {name}")
        with self.assertRaisesRegex(ValueError, "name"):
            template.format()

    def test_chain_uses_cached_result(self) -> None:
        client = CountingClient()
        chain = PromptChain(client, PromptTemplate("Summarize: {text}"), SimpleCache())

        first = chain.run(text="sample")
        second = chain.run(text="sample")

        self.assertEqual(first, second)
        self.assertEqual(client.calls, 1)

    def test_gpt_client_uses_mock_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = GPTClient(config={"model_name": "test-model"})
            response = client.generate("hello")

        self.assertIn("Mock GPT response - test-model", response)
        self.assertFalse(client.is_configured)

    def test_anthropic_client_uses_mock_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = AnthropicClient(config={"model_name": "test-model"})
            response = client.generate("hello")

        self.assertIn("Mock Anthropic response - test-model", response)
        self.assertFalse(client.is_configured)

    def test_rag_returns_retrieved_sources_without_llm_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            pipeline = RAGPipeline(
                SQLiteVectorStore(
                    os.path.join(directory, "vectors.sqlite3"),
                    HashEmbeddingProvider(),
                ),
                GPTClient(config={"model_name": "test-model"}),
            )
            pipeline.index_text(
                "science.txt",
                "The scientific method tests ideas through observation and experiments.",
                {"section_title": "Scientific method"},
            )

            answer = pipeline.ask("How does the scientific method test ideas?")

            self.assertIn("Resposta local baseada", answer)
            self.assertIn("tests ideas through observation and experiments", answer)
            self.assertIn("Fonte: science.txt", answer)
            self.assertNotIn("Mock GPT response", answer)

    def test_factory_uses_configured_provider(self) -> None:
        client = create_llm_client(
            config={
                "default_provider": "anthropic",
                "anthropic": {"model_name": "test-model"},
            }
        )

        self.assertIsInstance(client, AnthropicClient)

    def test_factory_creates_configured_gemini_client(self) -> None:
        client = create_llm_client(
            config={"default_provider": "gemini", "gemini": {"model_name": "gemini-test"}}
        )

        self.assertIsInstance(client, GeminiClient)
        self.assertEqual(client.model_name, "gemini-test")

    def test_factory_creates_configured_local_client(self) -> None:
        with patch("src.llm.factory.LocalAIEngine") as local_engine:
            client = create_llm_client(
                config={
                    "default_provider": "local",
                    "local": {"model_name": "google/flan-t5-small"},
                }
            )

        self.assertIs(client, local_engine.return_value)
        local_engine.assert_called_once()

    def test_local_engine_loads_and_generates_with_configured_model(self) -> None:
        calls: dict[str, object] = {}

        def fake_pipeline(task: str, **kwargs: object):
            calls["load"] = (task, kwargs)

            def generate(prompt: str, **generation_kwargs: object):
                calls["generate"] = (prompt, generation_kwargs)
                return [{"generated_text": " resposta local "}]

            return generate

        fake_transformers = ModuleType("transformers")
        fake_transformers.pipeline = fake_pipeline
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            engine = LocalAIEngine(
                config={
                    "model_name": "google/flan-t5-small",
                    "device": -1,
                    "temperature": 0,
                    "max_new_tokens": 80,
                }
            )
            answer = engine.generate("prompt")

        self.assertTrue(engine.is_configured)
        self.assertEqual(answer, "resposta local")
        self.assertEqual(
            calls["load"],
            ("text2text-generation", {"model": "google/flan-t5-small", "device": -1}),
        )
        self.assertEqual(
            calls["generate"],
            ("prompt", {"num_return_sequences": 1, "do_sample": False, "max_new_tokens": 80}),
        )

    def test_local_engine_reports_missing_transformers(self) -> None:
        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "torch, transformers, and sentencepiece"):
                LocalAIEngine(config={"model_name": "test-model"})

    def test_local_example_uses_wikipedia_context_and_local_provider(self) -> None:
        class FakeEngine:
            def generate(self, prompt: str, **kwargs: object) -> str:
                self.prompt = prompt
                self.kwargs = kwargs
                return "resposta local"

        engine = FakeEngine()
        with (
            patch(
                "examples.run_local_ai.load_model_config",
                return_value={
                    "local": {"model_name": "test-model", "max_new_tokens": 90},
                    "wikipedia": {"language": "pt", "timeout_seconds": 10},
                    "cognition": {"vocabulary_path": "/tmp/lexicon.json"},
                },
            ),
            patch("examples.run_local_ai.WikipediaExtractor") as extractor_type,
            patch("examples.run_local_ai.SemanticWikiDiscovererSkill") as discoverer_type,
            patch("examples.run_local_ai.PersonalVocabularyBuilder") as vocabulary_type,
            patch("examples.run_local_ai.create_llm_client", return_value=engine) as factory,
        ):
            discoverer_type.return_value.fetch_best_context.return_value = "Fonte Wikipédia"
            vocabulary_type.return_value.format_relevant_context.return_value = (
                "Term: IA\nMeaning: Sistema que executa tarefas."
            )
            answer = run_local_pipeline("Pergunta de teste")

        self.assertEqual(answer, "resposta local")
        factory.assert_called_once_with(
            config={
                "default_provider": "local",
                "local": {"model_name": "test-model", "max_new_tokens": 90},
            }
        )
        discoverer_type.return_value.fetch_best_context.assert_called_once_with(
            "Pergunta de teste"
        )
        self.assertIn("Fonte Wikipédia", engine.prompt)
        self.assertIn("Term: IA", engine.prompt)
        self.assertIn("Pergunta de teste", engine.prompt)
        self.assertEqual(engine.kwargs, {"max_new_tokens": 90})
        extractor_type.assert_called_once()
        vocabulary_type.assert_called_once_with("/tmp/lexicon.json")

    def test_gemini_client_reports_missing_key_as_unconfigured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiClient(config={"model_name": "gemini-test"})

        self.assertFalse(client.is_configured)

    def test_gemini_client_generates_content_with_configured_model(self) -> None:
        class FakeModels:
            def __init__(self) -> None:
                self.arguments: dict[str, object] = {}

            def generate_content(self, **kwargs: object) -> object:
                self.arguments = kwargs
                return type("Response", (), {"text": "Gemini generated answer"})()

        fake_models = FakeModels()
        fake_client = type("FakeClient", (), {"models": fake_models})()
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True),
            patch("google.genai.Client", return_value=fake_client),
        ):
            client = GeminiClient(
                config={
                    "model_name": "gemini-test",
                    "temperature": 0.2,
                    "max_output_tokens": 123,
                }
            )
            answer = client.generate("Explain this source.")

        self.assertEqual(answer, "Gemini generated answer")
        self.assertEqual(fake_models.arguments["model"], "gemini-test")
        self.assertEqual(fake_models.arguments["contents"], "Explain this source.")
        self.assertEqual(fake_models.arguments["config"].temperature, 0.2)
        self.assertEqual(fake_models.arguments["config"].max_output_tokens, 123)

    def test_gemini_embedding_provider_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                create_embedding_provider(
                    {"embedding_provider": "gemini", "embedding_model": "gemini-embedding-001"}
                )

    def test_gemini_embedding_provider_calls_query_and_document_tasks(self) -> None:
        class FakeGeminiModels:
            def __init__(self) -> None:
                self.task_types: list[str] = []

            def embed_content(self, model: str, contents: list[str], config: object) -> object:
                self.task_types.append(config.task_type)
                return type(
                    "EmbeddingResponse",
                    (),
                    {
                        "embeddings": [
                            type("Embedding", (), {"values": [0.1, 0.2]})()
                            for _ in contents
                        ]
                    },
                )()

        fake_models = FakeGeminiModels()
        fake_client = type("FakeClient", (), {"models": fake_models})()
        with (
            patch("google.genai.Client", return_value=fake_client, create=True),
            patch(
                "google.genai.types.EmbedContentConfig",
                side_effect=lambda task_type: type("Config", (), {"task_type": task_type})(),
                create=True,
            ),
        ):
            provider = GeminiEmbeddingProvider(api_key="test-key")
            query_embedding = provider.embed("what is this?")
            document_embeddings = provider.embed_many(["document one", "document two"])

        self.assertEqual(query_embedding, [0.1, 0.2])
        self.assertEqual(document_embeddings, [[0.1, 0.2], [0.1, 0.2]])
        self.assertEqual(fake_models.task_types, ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"])

    def test_gemini_embeddings_split_inputs_into_batches_of_at_most_100(self) -> None:
        class FakeGeminiModels:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def embed_content(self, model: str, contents: list[str], config: object) -> object:
                self.batch_sizes.append(len(contents))
                return type(
                    "EmbeddingResponse",
                    (),
                    {
                        "embeddings": [
                            type("Embedding", (), {"values": [float(index)]})()
                            for index, _ in enumerate(contents)
                        ]
                    },
                )()

        fake_models = FakeGeminiModels()
        fake_client = type("FakeClient", (), {"models": fake_models})()
        with patch("google.genai.Client", return_value=fake_client):
            provider = GeminiEmbeddingProvider(api_key="test-key")
            vectors = provider.embed_many([f"document {index}" for index in range(205)])

        self.assertEqual(fake_models.batch_sizes, [100, 100, 5])
        self.assertEqual(len(vectors), 205)
        self.assertEqual(vectors[0], [0.0])
        self.assertEqual(vectors[100], [0.0])
        self.assertEqual(vectors[200], [0.0])

    def test_gemini_embeddings_wait_for_quota_retry_delay(self) -> None:
        class QuotaError(Exception):
            code = 429

        class FakeGeminiModels:
            def __init__(self) -> None:
                self.calls = 0

            def embed_content(self, model: str, contents: list[str], config: object) -> object:
                self.calls += 1
                if self.calls == 1:
                    raise QuotaError("RESOURCE_EXHAUSTED; please retry in 15.2s")
                return type(
                    "EmbeddingResponse",
                    (),
                    {"embeddings": [type("Embedding", (), {"values": [0.1, 0.2]})()]},
                )()

        fake_models = FakeGeminiModels()
        fake_client = type("FakeClient", (), {"models": fake_models})()
        sleep_durations: list[float] = []
        with (
            patch("google.genai.Client", return_value=fake_client),
            patch("src.rag.embeddings.time.sleep", side_effect=sleep_durations.append),
        ):
            vectors = GeminiEmbeddingProvider(api_key="test-key").embed_many(["document"])

        self.assertEqual(vectors, [[0.1, 0.2]])
        self.assertEqual(fake_models.calls, 2)
        self.assertIn(15.2, sleep_durations)

    def test_factory_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            create_llm_client(config={"default_provider": "unknown"})

    def test_chunking_respects_size_and_overlap(self) -> None:
        chunks = chunk_text("one two three four five", chunk_size=3, overlap=1)

        self.assertEqual(chunks, ["one two three", "three four five"])

    def test_vector_store_searches_and_persists_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "vectors.sqlite3")
            embeddings = HashEmbeddingProvider()
            store = SQLiteVectorStore(path, embeddings)
            store.upsert(
                [
                    ("cats", "Cats are playful pets", "animals.txt", {"kind": "cat"}),
                    ("space", "Mars is a planet in space", "space.txt", {}),
                ]
            )

            results = store.search("playful cats", top_k=1)
            reopened_store = SQLiteVectorStore(path, embeddings)

            self.assertEqual(results[0].id, "cats")
            self.assertEqual(reopened_store.count(), 2)
            self.assertEqual(results[0].metadata["kind"], "cat")

    def test_vector_store_reuses_embeddings_for_unchanged_records(self) -> None:
        class CountingEmbeddingProvider(HashEmbeddingProvider):
            def __init__(self) -> None:
                super().__init__()
                self.embed_calls: list[list[str]] = []

            def embed_many(self, texts: list[str]) -> list[list[float]]:
                self.embed_calls.append(list(texts))
                return super().embed_many(texts)

        embedding_provider = CountingEmbeddingProvider()
        store = SQLiteVectorStore(":memory:", embedding_provider)
        records = [("same-id", "same text", "same-source", {"section": "intro"})]

        store.upsert(records, replace_source="same-source")
        store.upsert(records, replace_source="same-source")

        self.assertEqual(embedding_provider.embed_calls, [["same text"], []])
        self.assertEqual(store.count(), 1)

    def test_vector_store_supports_in_memory_database(self) -> None:
        store = SQLiteVectorStore(":memory:", HashEmbeddingProvider())
        store.upsert([("id", "memory database test", "memory.txt", {})])

        self.assertEqual(store.count(), 1)
        self.assertEqual(store.search("memory database", top_k=1)[0].id, "id")

    def test_rag_pipeline_retrieves_context_for_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = CountingClient()
            store = SQLiteVectorStore(
                os.path.join(directory, "vectors.sqlite3"),
                HashEmbeddingProvider(),
            )
            pipeline = RAGPipeline(store, client, chunk_size=20, chunk_overlap=2)
            pipeline.index_text("guide.txt", "RAG retrieves documents and provides context.")

            response = pipeline.ask("How does RAG provide context?")

            self.assertIn("generated:", response)
            self.assertIn("Source: guide.txt", response)

    def test_reindexing_source_replaces_old_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteVectorStore(
                os.path.join(directory, "vectors.sqlite3"),
                HashEmbeddingProvider(),
            )
            pipeline = RAGPipeline(store, CountingClient(), chunk_size=3, chunk_overlap=0)
            pipeline.index_text("guide.txt", "alpha beta gamma")
            pipeline.index_text("guide.txt", "delta epsilon")

            self.assertEqual(store.count(), 1)
            self.assertEqual(store.search("alpha", top_k=1)[0].text, "delta epsilon")

    def test_rag_pipeline_requires_indexed_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = RAGPipeline(
                SQLiteVectorStore(
                    os.path.join(directory, "vectors.sqlite3"),
                    HashEmbeddingProvider(),
                ),
                CountingClient(),
            )

            with self.assertRaisesRegex(LookupError, "Index source documents"):
                pipeline.ask("What is in the knowledge base?")

    def test_vector_store_filters_by_learning_level(self) -> None:
        store = SQLiteVectorStore(":memory:", HashEmbeddingProvider())
        store.upsert(
            [
                ("basic", "Artificial intelligence is a computer science field", "wiki", {"learning_level": "basico"}),
                ("advanced", "Artificial intelligence has theoretical limitations", "wiki", {"learning_level": "avancado"}),
            ]
        )

        results = store.search(
            "artificial intelligence",
            top_k=5,
            metadata_filter={"learning_level": "avancado"},
        )

        self.assertEqual([result.id for result in results], ["advanced"])

    def test_wikipedia_extractor_parses_levels_and_saves_attribution(self) -> None:
        api_response = {
            "parse": {
                "title": "Inteligência artificial",
                "text": {
                    "*": (
                        "<p>Resumo simples do conceito.</p>"
                        "<h2>História</h2><p>Como o tema evoluiu.</p>"
                        "<h2>Críticas</h2><p>Limitações e debates avançados.</p>"
                        "<sup>referência removida</sup>"
                    )
                },
            }
        }
        with patch(
            "src.utils.wiki_extractor.urlopen",
            return_value=BytesIO(json.dumps(api_response).encode("utf-8")),
        ):
            article = WikipediaExtractor().extract("Inteligência artificial")

        self.assertEqual(
            [section.learning_level for section in article.sections],
            ["basico", "intermediario", "avancado"],
        )
        self.assertNotIn("referência removida", " ".join(section.text for section in article.sections))
        with tempfile.TemporaryDirectory() as directory:
            json_path = WikipediaExtractor().save_json(article, directory)
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["license"], "CC BY-SA 4.0; attribute the original Wikipedia article")
            self.assertEqual(saved["sections"][2]["learning_level"], "avancado")

    def test_wikipedia_extractor_searches_official_api(self) -> None:
        response_body = {
            "query": {
                "search": [
                    {
                        "title": "Método científico",
                        "snippet": "Pesquisa sobre <span class=\"searchmatch\">ciência</span>.",
                    }
                ]
            }
        }
        with patch(
            "src.utils.wiki_extractor.urlopen",
            side_effect=lambda *args, **kwargs: BytesIO(
                json.dumps(response_body).encode("utf-8")
            ),
        ) as mocked_urlopen:
            extractor = WikipediaExtractor()
            title = extractor.search("como funciona o método científico")
            topics = extractor.search_related_topics("ciência")

        self.assertEqual(title, "Método científico")
        self.assertEqual(topics[0]["snippet"], "Pesquisa sobre ciência.")
        request = mocked_urlopen.call_args.args[0]
        self.assertIn("https://pt.wikipedia.org/w/api.php?", request.full_url)
        self.assertIn("action=query", request.full_url)
        self.assertIn("list=search", request.full_url)

    def test_wikipedia_extractor_fetches_thumbnail_and_image_license(self) -> None:
        responses = [
            {
                "query": {
                    "pages": {
                        "42": {
                            "pageimage": "Neural_network.svg",
                            "thumbnail": {
                                "source": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Neural_network.svg/960px-Neural_network.svg.png"
                            },
                        }
                    }
                }
            },
            {
                "query": {
                    "pages": {
                        "101": {
                            "imageinfo": [
                                {
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Neural_network.svg",
                                    "extmetadata": {
                                        "LicenseShortName": {
                                            "value": "<a>CC BY-SA 4.0</a>"
                                        },
                                        "Artist": {"value": "<a>Jane Doe</a>"},
                                    },
                                }
                            ]
                        }
                    }
                }
            },
        ]
        with patch(
            "src.utils.wiki_extractor.urlopen",
            side_effect=[
                BytesIO(json.dumps(response).encode("utf-8"))
                for response in responses
            ],
        ) as mocked_urlopen:
            image = WikipediaExtractor().fetch_article_image("Redes neurais")

        self.assertEqual(image["title"], "Neural_network.svg")
        self.assertTrue(image["url"].startswith("https://upload.wikimedia.org/"))
        self.assertEqual(
            image["source_url"],
            "https://commons.wikimedia.org/wiki/File:Neural_network.svg",
        )
        self.assertEqual(image["license"], "CC BY-SA 4.0")
        self.assertEqual(image["artist"], "Jane Doe")
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_wikipedia_extractor_rejects_non_wikimedia_thumbnail_urls(self) -> None:
        response = {
            "query": {
                "pages": {
                    "42": {
                        "pageimage": "external.jpg",
                        "thumbnail": {"source": "https://example.org/external.jpg"},
                    }
                }
            }
        }
        with patch(
            "src.utils.wiki_extractor.urlopen",
            return_value=BytesIO(json.dumps(response).encode("utf-8")),
        ):
            with self.assertRaisesRegex(ValueError, "outside upload.wikimedia.org"):
                WikipediaExtractor().fetch_article_image("Example")

    def test_semantic_wiki_discovery_reranks_related_topics_and_fetches_context(self) -> None:
        class FakeExtractor:
            def __init__(self) -> None:
                self.queries: list[str] = []
                self.article = WikipediaArticle(
                    title="Visão computacional",
                    language="pt",
                    source_url="https://pt.wikipedia.org/wiki/Vis%C3%A3o_computacional",
                    retrieved_at="2026-10-24T00:00:00Z",
                    sections=[
                        WikipediaSection(
                            "Introdução",
                            "Visão computacional permite às máquinas interpretar imagens.",
                            "basico",
                            0,
                        )
                    ],
                )

            def search_related_topics(
                self,
                query: str,
                limit: int = 5,
            ) -> list[dict[str, str]]:
                self.queries.append(query)
                if query.startswith("Gostaria"):
                    return [
                        {"title": "Fotografia", "snippet": "Registo de imagens por câmaras."},
                        {"title": "Câmara digital", "snippet": "Dispositivo para captar imagens."},
                    ][:limit]
                return [
                    {
                        "title": "Visão computacional",
                        "snippet": "Reconhecimento de padrões visuais em imagens.",
                    },
                    {
                        "title": "Reconhecimento facial",
                        "snippet": "Identificação de rostos em fotografias.",
                    },
                ][:limit]

            def extract(self, title: str) -> WikipediaArticle:
                self.extracted_title = title
                return self.article

        extractor = FakeExtractor()
        discoverer = SemanticWikiDiscovererSkill(extractor=extractor)
        query = (
            "Gostaria de entender como as máquinas aprendem a reconhecer "
            "padrões visuais e rostos humanos nas fotos."
        )

        topics = discoverer.search_related_topics(query, limit=3)
        context = discoverer.fetch_best_context(query)

        self.assertEqual(len(extractor.queries), 12)
        self.assertEqual(topics[0]["title"], "Visão computacional")
        self.assertIn(
            "Reconhecimento facial",
            [topic["title"] for topic in topics],
        )
        self.assertEqual(extractor.extracted_title, "Visão computacional")
        self.assertIn("interpretar imagens", context)
        self.assertIn("https://pt.wikipedia.org/wiki/", context)

    def test_semantic_discovery_finds_topic_with_typo_and_avoids_irrelevant_sun_article(self) -> None:
        class FakeExtractor:
            def search_related_topics(
                self,
                query: str,
                limit: int = 5,
            ) -> list[dict[str, str]]:
                return [
                    {
                        "title": "Sol",
                        "snippet": "Estrela central do Sistema Solar.",
                    },
                    {
                        "title": "Fotossíntese",
                        "snippet": "Processo pelo qual plantas usam luz solar para produzir energia.",
                    },
                ]

            def extract(self, title: str) -> WikipediaArticle:
                self.extracted_title = title
                return WikipediaArticle(
                    title=title,
                    language="pt",
                    source_url=f"https://pt.wikipedia.org/wiki/{title}",
                    retrieved_at="2026-10-24T00:00:00Z",
                    sections=[
                        WikipediaSection(
                            "Introdução",
                            "A fotossíntese converte luz em energia química nas plantas.",
                            "basico",
                            0,
                        )
                    ],
                )

        extractor = FakeExtractor()
        discoverer = SemanticWikiDiscovererSkill(extractor=extractor)
        question = "Como a luz solar influencia na fotossínte das plantas?"

        topics = discoverer.search_related_topics(question, limit=2)
        article = discoverer.fetch_best_article(question)

        self.assertEqual(topics[0]["title"], "Fotossíntese")
        self.assertEqual(article.title, "Fotossíntese")
        self.assertEqual(extractor.extracted_title, "Fotossíntese")

    def test_semantic_discovery_prefers_stainless_steel_over_movie_about_steel(self) -> None:
        class FakeExtractor:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def search_related_topics(
                self,
                query: str,
                limit: int = 5,
            ) -> list[dict[str, str]]:
                self.queries.append(query)
                return [
                    {
                        "title": "Steel - O Homem de Aço",
                        "snippet": "Filme de ação sobre o personagem Aço.",
                    },
                    {
                        "title": "Aço inoxidável",
                        "snippet": (
                            "O aço inoxidável resiste à corrosão devido à camada passiva "
                            "formada pelo crómio."
                        ),
                    },
                    {
                        "title": "Corrosão",
                        "snippet": "Degradação de metais por reações químicas com o ambiente.",
                    },
                ][:limit]

            def extract(self, title: str) -> WikipediaArticle:
                self.selected_title = title
                return WikipediaArticle(
                    title=title,
                    language="pt",
                    source_url=f"https://pt.wikipedia.org/wiki/{title}",
                    retrieved_at="2026-10-24T00:00:00Z",
                    sections=[
                        WikipediaSection("Introdução", "Conteúdo relevante.", "basico", 0)
                    ],
                )

        extractor = FakeExtractor()
        discoverer = SemanticWikiDiscovererSkill(extractor=extractor)
        question = "Como o processo de tornar o aço inox retarda a corrosão metálica?"

        topics = discoverer.search_related_topics(question, limit=3)
        article = discoverer.fetch_best_article(question)

        self.assertIn("aco inox", extractor.queries)
        self.assertIn("corrosao metalica", extractor.queries)
        self.assertEqual(topics[0]["title"], "Aço inoxidável")
        self.assertEqual(article.title, "Aço inoxidável")
        self.assertEqual(extractor.selected_title, "Aço inoxidável")

    def test_follow_up_detection_does_not_reuse_previous_topic_for_new_question(self) -> None:
        self.assertTrue(_is_context_dependent("Como isso funciona?"))
        self.assertFalse(
            _is_context_dependent("Como a luz solar influencia na fotossíntese das plantas?")
        )

    def test_wikipedia_chat_searches_indexes_and_answers_with_cited_excerpts(self) -> None:
        class FakeExtractor:
            def __init__(self) -> None:
                self.search_query = ""

            def search(self, query: str) -> str:
                self.search_query = query
                return "Método científico"

            def search_related_topics(
                self,
                query: str,
                limit: int = 5,
            ) -> list[dict[str, str]]:
                self.search_query = query
                return [{"title": self.search(query), "snippet": "Método científico."}]

            def extract(self, title: str) -> WikipediaArticle:
                self.asserted_title = title
                return WikipediaArticle(
                    title=title,
                    language="pt",
                    source_url="https://pt.wikipedia.org/wiki/M%C3%A9todo_cient%C3%ADfico",
                    retrieved_at="2026-10-24T00:00:00Z",
                    sections=[
                        WikipediaSection(
                            "Introdução",
                            "O método científico testa hipóteses por observação e experimentação. "
                            + ("Este conteúdo completo não deve ser cortado. " * 30),
                            "basico",
                            0,
                        ),
                        WikipediaSection(
                            "Funcionamento",
                            "Os resultados são analisados e podem levar a novas hipóteses.",
                            "intermediario",
                            1,
                        ),
                        WikipediaSection(
                            "Resultados",
                            "Os resultados podem ser reproduzidos e avaliados por outros investigadores.",
                            "avancado",
                            2,
                        ),
                    ],
                )

            def fetch_article_image(self, title: str) -> dict[str, str] | None:
                return {
                    "title": "Scientific_method.svg",
                    "url": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Scientific_method.svg",
                    "source_url": "https://commons.wikimedia.org/wiki/File:Scientific_method.svg",
                    "license": "CC BY-SA 4.0",
                    "artist": "Jane Doe",
                }

        extractor = FakeExtractor()
        client = CountingClient()
        with tempfile.TemporaryDirectory() as directory:
            pipeline = RAGPipeline(
                SQLiteVectorStore(
                    os.path.join(directory, "vectors.sqlite3"),
                    HashEmbeddingProvider(),
                ),
                client,
            )
            learner = CognitiveLearnerSkill(
                SQLiteCognitiveMemory(os.path.join(directory, "memory.sqlite3"))
            )
            chat = WikipediaChat(extractor, pipeline, learner)
            history = [{"role": "user", "content": "Método científico"}]

            answer = chat.answer(
                "Como isso funciona?",
                learning_level="intermediario",
                conversation_history=history,
            )
            memory_query = "Método científico Como isso funciona?"
            recalled = learner.recall(memory_query, level="intermediario")
            chat.answer(
                "Como isso funciona?",
                learning_level="intermediario",
                conversation_history=history,
            )
            recalled_again = learner.recall(memory_query, level="intermediario")

            self.assertEqual(extractor.asserted_title, "Método científico")
            self.assertIn("visão geral sobre Método científico", answer)
            self.assertIn("Os resultados são analisados", answer)
            self.assertIn("Este conteúdo completo não deve ser cortado. " * 30, answer)
            self.assertIn("## Resultados", answer)
            self.assertIn("https://pt.wikipedia.org/wiki/", answer)
            self.assertIn("CC BY-SA 4.0", answer)
            self.assertIn(
                "![Ilustração: Método científico](https://upload.wikimedia.org/",
                answer,
            )
            self.assertIn(
                "[detalhes do ficheiro](https://commons.wikimedia.org/wiki/File:",
                answer,
            )
            self.assertIn("Autor: Jane Doe", answer)
            self.assertEqual(pipeline.vector_store.count(), 4)
            self.assertEqual(client.calls, 0)
            self.assertIsNotNone(recalled)
            self.assertEqual(recalled["sources"], ["https://pt.wikipedia.org/wiki/M%C3%A9todo_cient%C3%ADfico"])
            self.assertFalse(recalled["assessment"]["has_knowledge_gap"])
            self.assertTrue(
                recalled_again["assessment"]["initial_assessment"]["prior_memory_found"]
            )

    def test_progressive_rag_filters_level_and_instructs_tutor(self) -> None:
        client = CountingClient()
        store = SQLiteVectorStore(":memory:", HashEmbeddingProvider())
        pipeline = RAGPipeline(store, client)
        pipeline.index_documents(
            [
                ("wiki", "AI is a field of computer science.", {"learning_level": "basico"}),
                ("wiki", "Advanced AI theory studies model limitations.", {"learning_level": "avancado"}),
            ]
        )

        pipeline.ask("What are AI limitations?", learning_level="avancado")

        self.assertIn("precise technical terminology", client.last_prompt)
        self.assertIn("Advanced AI theory", client.last_prompt)
        self.assertNotIn("AI is a field of computer science", client.last_prompt)

    def test_rag_pipeline_includes_recent_conversation_as_context(self) -> None:
        client = CountingClient()
        store = SQLiteVectorStore(":memory:", HashEmbeddingProvider())
        pipeline = RAGPipeline(store, client)
        pipeline.index_text("guide.txt", "The scientific method tests hypotheses.")

        pipeline.ask(
            "Why is that useful?",
            conversation_history=[
                {"role": "user", "content": "What does the scientific method test?"},
                {"role": "assistant", "content": "It tests hypotheses."},
            ],
        )

        self.assertIn("Recent conversation", client.last_prompt)
        self.assertIn("Assistant: It tests hypotheses.", client.last_prompt)

    def test_rag_generation_uses_only_relevant_personal_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vocabulary = PersonalVocabularyBuilder(Path(directory) / "lexicon.json")
            vocabulary.acquire_sign(
                term="Algoritmo",
                meaning="Sequência finita de instruções para resolver um problema.",
                context="Pesquisa sobre computação.",
                level="basico",
                source="https://pt.wikipedia.org/wiki/Algoritmo",
            )
            vocabulary.acquire_sign(
                term="Fotossíntese",
                meaning="Processo pelo qual plantas convertem energia luminosa.",
                context="Pesquisa sobre plantas.",
                level="basico",
            )
            client = CountingClient()
            pipeline = RAGPipeline(
                SQLiteVectorStore(":memory:", HashEmbeddingProvider()),
                client,
                vocabulary_builder=vocabulary,
            )
            pipeline.index_text(
                "computacao.txt",
                "Algoritmos descrevem passos para realizar tarefas computacionais.",
            )

            pipeline.ask("Como funciona um algoritmo?")

        self.assertIn("Personal lexicon", client.last_prompt)
        self.assertIn("Sequência finita de instruções", client.last_prompt)
        self.assertIn("https://pt.wikipedia.org/wiki/Algoritmo", client.last_prompt)
        self.assertNotIn("Processo pelo qual plantas", client.last_prompt)
        self.assertIn("cite those excerpts", client.last_prompt)

    def test_cli_ingests_wikipedia_before_answering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "vectors.sqlite3")
            article = WikipediaArticle(
                title="Artificial intelligence",
                language="en",
                source_url="https://en.wikipedia.org/wiki/Artificial_intelligence",
                retrieved_at="2026-10-24T00:00:00Z",
                sections=[
                    WikipediaSection(
                        title="Introduction",
                        text="Artificial intelligence is a field of computer science.",
                        learning_level="basico",
                        section_index=0,
                    )
                ],
            )
            configuration = {
                "default_provider": "openai",
                "openai": {"model_name": "test-model"},
                "rag": {
                    "embedding_provider": "hash",
                    "database_path": database_path,
                    "chunk_size": 160,
                    "chunk_overlap": 30,
                    "top_k": 2,
                },
                "wikipedia": {"output_dir": directory},
                "cognition": {"memory_path": os.path.join(directory, "memory.sqlite3")},
            }

            with (
                patch("main.load_model_config", return_value=configuration),
                patch("main.WikipediaExtractor") as extractor_type,
                patch("examples.basic_completion.create_llm_client", return_value=CountingClient()),
                redirect_stdout(StringIO()) as output,
            ):
                extractor = extractor_type.return_value
                extractor.search.return_value = article.title
                extractor.search_related_topics.return_value = [
                    {"title": article.title, "snippet": "Artificial intelligence overview."}
                ]
                extractor.extract.return_value = article
                extractor.save_json.return_value = os.path.join(directory, "article.json")
                app_main.main(
                    [
                        "--wiki-topic",
                        article.title,
                        "--ask",
                        "What is artificial intelligence?",
                        "--level",
                        "basico",
                    ]
                )

            self.assertIn("visão geral sobre Artificial intelligence", output.getvalue())
            self.assertIn(article.source_url, output.getvalue())
            store = SQLiteVectorStore(database_path, HashEmbeddingProvider())
            self.assertEqual(store.count(), 1)
            self.assertEqual(
                store.search(
                    "artificial intelligence",
                    metadata_filter={"learning_level": "basico"},
                )[0].metadata["article_title"],
                article.title,
            )

    def test_universal_curriculum_exports_levels_and_indexes_metadata(self) -> None:
        class FakeExtractor:
            def extract(self, title: str) -> WikipediaArticle:
                return WikipediaArticle(
                    title=title,
                    language="pt",
                    source_url=f"https://pt.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    retrieved_at="2026-10-24T00:00:00Z",
                    sections=[
                        WikipediaSection("Introdução", f"{title} é um conceito científico.", "basico", 0),
                        WikipediaSection("Funcionamento", f"Como funciona {title}.", "intermediario", 1),
                    ],
                )

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteVectorStore(
                os.path.join(directory, "vectors.sqlite3"),
                HashEmbeddingProvider(),
            )
            pipeline = RAGPipeline(store, CountingClient())
            report = ingest_universal_curriculum(
                FakeExtractor(),
                pipeline,
                output_dir=os.path.join(directory, "universal_curriculum"),
                topics=(
                    CurriculumTopic("Ciência", "ciência"),
                    CurriculumTopic("Redes neurais artificiais", "tecnologia", advanced=True),
                ),
            )
            basic_json = os.path.join(
                directory,
                "universal_curriculum",
                "nivel_1_basico",
                "ciência.json",
            )
            advanced_json = os.path.join(
                directory,
                "universal_curriculum",
                "nivel_3_avancado",
                "redes_neurais_artificiais.json",
            )
            basic_data = json.loads(Path(basic_json).read_text(encoding="utf-8"))
            advanced_data = json.loads(Path(advanced_json).read_text(encoding="utf-8"))

            self.assertEqual((report.articles, report.sections, report.chunks), (2, 4, 4))
            self.assertEqual(basic_data["sections"][0]["title"], "Introdução")
            self.assertEqual(advanced_data["learning_level"], "avancado")
            self.assertEqual(len(report.files), 6)
            self.assertEqual(
                store.search(
                    "redes neurais",
                    top_k=3,
                    metadata_filter={"learning_level": "avancado"},
                )[0].metadata["subject"],
                "tecnologia",
            )

    def test_local_chat_server_serves_ui_status_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = RAGPipeline(
                SQLiteVectorStore(
                    os.path.join(directory, "vectors.sqlite3"),
                    HashEmbeddingProvider(),
                ),
                CountingClient(),
            )
            pipeline.index_text(
                "guide.txt",
                "The scientific method tests hypotheses using observation and experiments.",
                {"learning_level": "basico", "section_title": "Introduction"},
            )
            server = create_chat_server(pipeline, port=0)
            self.assertEqual(server.server_address[0], "127.0.0.1")
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(f"{base_url}/", timeout=2) as response:
                    page = response.read().decode("utf-8")
                    self.assertIn("Assistente de conhecimento", page)
                    self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

                with urlopen(f"{base_url}/api/status", timeout=2) as response:
                    status = json.loads(response.read())
                    self.assertEqual(status["indexed_chunks"], 1)
                    self.assertEqual(status["generation_mode"], "llm")

                request = Request(
                    f"{base_url}/api/chat",
                    data=json.dumps(
                        {
                            "question": "Why is that useful?",
                            "level": "basico",
                            "history": [
                                {"role": "user", "content": "What does the method test?"},
                                {"role": "assistant", "content": "It tests hypotheses."},
                            ],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2) as response:
                    result = json.loads(response.read())
                    self.assertIn("generated:", result["answer"])
                    self.assertIn("Source: guide.txt", result["answer"])
                    self.assertEqual(result["generation_mode"], "llm")

                malicious_request = Request(
                    f"{base_url}/api/chat",
                    data=json.dumps({"question": "hello"}).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "https://attacker.example",
                    },
                    method="POST",
                )
                with self.assertRaises(HTTPError) as error:
                    urlopen(malicious_request, timeout=2)
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_chat_server_uses_live_wikipedia_answer_service(self) -> None:
        class FakeExtractor:
            def search(self, query: str) -> str:
                self.query = query
                return "Ciência"

            def search_related_topics(
                self,
                query: str,
                limit: int = 5,
            ) -> list[dict[str, str]]:
                return [{"title": self.search(query), "snippet": "Ciência e evidências."}]

            def extract(self, title: str) -> WikipediaArticle:
                return WikipediaArticle(
                    title=title,
                    language="pt",
                    source_url="https://pt.wikipedia.org/wiki/Ci%C3%AAncia",
                    retrieved_at="2026-10-24T00:00:00Z",
                    sections=[
                        WikipediaSection(
                            "Introdução",
                            "Ciência é um sistema de conhecimento baseado em evidências.",
                            "basico",
                            0,
                        )
                    ],
                )

            def fetch_article_image(self, title: str) -> dict[str, str] | None:
                return {
                    "title": "Science.svg",
                    "url": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Science.svg",
                    "source_url": "https://commons.wikimedia.org/wiki/File:Science.svg",
                    "license": "CC BY-SA 4.0",
                }

        client = CountingClient()
        with tempfile.TemporaryDirectory() as directory:
            pipeline = RAGPipeline(
                SQLiteVectorStore(
                    os.path.join(directory, "vectors.sqlite3"),
                    HashEmbeddingProvider(),
                ),
                client,
            )
            wikipedia_chat = WikipediaChat(FakeExtractor(), pipeline)
            server = create_chat_server(pipeline, port=0, wikipedia_chat=wikipedia_chat)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(f"{base_url}/api/status", timeout=2) as response:
                    status = json.loads(response.read())
                self.assertEqual(status["generation_mode"], "wikipedia")

                request = Request(
                    f"{base_url}/api/chat",
                    data=json.dumps({"question": "O que é ciência?"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2) as response:
                    result = json.loads(response.read())

                self.assertEqual(result["generation_mode"], "wikipedia")
                self.assertIn("Ciência é um sistema", result["answer"])
                self.assertIn("https://pt.wikipedia.org/wiki/Ci%C3%AAncia", result["answer"])
                self.assertIn(
                    "![Ilustração: Ciência](https://upload.wikimedia.org/",
                    result["answer"],
                )
                self.assertIn("CC BY-SA 4.0", result["answer"])
                self.assertEqual(client.calls, 0)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
