import argparse
from collections import Counter
from typing import Sequence

from examples.basic_completion import build_rag_pipeline, run_example
from src.cognition.cognitive_learner import CognitiveLearnerSkill, SQLiteCognitiveMemory
from src.llm.utils import load_model_config
from src.utils.logger import logger
from src.utils.secrets import load_local_secrets
from src.utils.universal_curriculum import ingest_universal_curriculum
from src.utils.wiki_extractor import LEARNING_LEVELS, WikipediaExtractor
from src.utils.wikipedia_chat import WikipediaChat
from src.web.chat_server import serve_chat


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Generative AI RAG MVP.")
    wiki_ingestion = parser.add_mutually_exclusive_group()
    wiki_ingestion.add_argument(
        "--wiki-topic",
        help="Fetch and index a Wikipedia article before answering.",
    )
    wiki_ingestion.add_argument(
        "--universal-curriculum",
        action="store_true",
        help="Fetch and index the curated universal knowledge curriculum.",
    )
    parser.add_argument(
        "--wiki-language",
        help="Wikipedia language code (defaults to config, then pt).",
    )
    parser.add_argument(
        "--curriculum-output",
        help="Curriculum JSON output directory (defaults to config).",
    )
    interaction = parser.add_mutually_exclusive_group()
    interaction.add_argument(
        "--ask",
        help="Ask a question against the indexed RAG knowledge base.",
    )
    interaction.add_argument(
        "--chat",
        action="store_true",
        help="Start the local browser chat interface.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Local chat interface port (default: 8000).",
    )
    parser.add_argument(
        "--level",
        choices=LEARNING_LEVELS,
        help="Filter retrieved sections and tune the answer for this learning level.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    load_local_secrets()
    config = load_model_config()
    wiki_config = config.get("wikipedia", {})
    topic = args.wiki_topic
    if topic is None and wiki_config.get("enabled", False):
        topic = wiki_config.get("topic")
        if not topic:
            raise ValueError("wikipedia.topic must be set when Wikipedia ingestion is enabled.")

    if args.level and not args.ask:
        raise ValueError("--level requires --ask.")
    if not topic and not args.universal_curriculum and not args.ask and not args.chat:
        logger.info("Starting the Generative AI MVP...")
        run_example()
        logger.info("Execution completed.")
        return

    pipeline = build_rag_pipeline(config)
    extractor = WikipediaExtractor(
        language=args.wiki_language or wiki_config.get("language", "pt"),
        timeout=wiki_config.get("timeout_seconds", 15),
    )
    if args.universal_curriculum:
        curriculum_config = wiki_config.get("universal_curriculum", {})
        report = ingest_universal_curriculum(
            extractor,
            pipeline,
            output_dir=(
                args.curriculum_output
                or curriculum_config.get("output_dir", "data/universal_curriculum")
            ),
        )
        logger.info(
            "Indexed universal curriculum: %d articles, %d sections, %d chunks; %d JSON files",
            report.articles,
            report.sections,
            report.chunks,
            len(report.files),
        )
    elif topic:
        article = extractor.extract(topic)
        output_path = extractor.save_json(
            article,
            output_dir=wiki_config.get("output_dir", "data/prompts"),
        )
        documents = [
            (
                article.source_url,
                section.text,
                {
                    "article_title": article.title,
                    "section_title": section.title,
                    "section_index": section.section_index,
                    "learning_level": section.learning_level,
                    "source_url": article.source_url,
                    "license": "CC BY-SA 4.0",
                },
            )
            for section in article.sections
        ]
        chunk_count = pipeline.index_documents(documents)
        level_counts = Counter(section.learning_level for section in article.sections)
        logger.info(
            "Indexed Wikipedia article %r: %d sections, %d chunks; levels=%s; JSON=%s",
            article.title,
            len(article.sections),
            chunk_count,
            dict(level_counts),
            output_path,
        )

    if args.ask or args.chat:
        cognition_config = config.get("cognition", {})
        cognitive_learner = CognitiveLearnerSkill(
            SQLiteCognitiveMemory(
                cognition_config.get("memory_path", "data/cognitive_memory.sqlite3")
            )
        )
        wikipedia_chat = WikipediaChat(
            extractor,
            pipeline,
            cognitive_learner,
            vocabulary_builder=pipeline.vocabulary_builder,
        )
    if args.ask:
        print(wikipedia_chat.answer(args.ask, learning_level=args.level))
    elif args.chat:
        rag_config = config.get("rag", {})
        serve_chat(
            pipeline,
            port=args.port,
            top_k=rag_config.get("top_k", 4),
            wikipedia_chat=wikipedia_chat,
        )


if __name__ == "__main__":
    main()
