import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from src.rag.pipeline import RAGPipeline
from src.utils.wiki_extractor import LEARNING_LEVELS, WikipediaArticle, WikipediaSection


@dataclass(frozen=True)
class CurriculumTopic:
    title: str
    subject: str
    advanced: bool = False


@dataclass(frozen=True)
class CurriculumIngestionReport:
    articles: int
    sections: int
    chunks: int
    files: tuple[Path, ...]


class ArticleExtractor(Protocol):
    def extract(self, title: str) -> WikipediaArticle: ...


CURRICULUM_TOPICS = (
    CurriculumTopic("Método científico", "ciência"),
    CurriculumTopic("Física", "ciência"),
    CurriculumTopic("Biologia celular", "ciência"),
    CurriculumTopic("História do mundo", "história e sociedade"),
    CurriculumTopic("Revolução Industrial", "história e sociedade"),
    CurriculumTopic("Democracia", "história e sociedade"),
    CurriculumTopic("Ciência da computação", "tecnologia e lógica"),
    CurriculumTopic("Algoritmo", "tecnologia e lógica"),
    CurriculumTopic("Inteligência artificial", "tecnologia e lógica"),
    CurriculumTopic("Filosofia", "humanidades"),
    CurriculumTopic("Linguística", "humanidades"),
    CurriculumTopic("Aprendizado de máquina", "tecnologia e lógica", advanced=True),
    CurriculumTopic("Redes neurais artificiais", "tecnologia e lógica", advanced=True),
    CurriculumTopic("Mecânica quântica", "ciência", advanced=True),
    CurriculumTopic("Teoria dos jogos", "matemática e sociedade", advanced=True),
)

_LEVEL_DIRECTORIES = {
    "basico": "nivel_1_basico",
    "intermediario": "nivel_2_intermediario",
    "avancado": "nivel_3_avancado",
}


def ingest_universal_curriculum(
    extractor: ArticleExtractor,
    pipeline: RAGPipeline,
    output_dir: str | Path = "data/universal_curriculum",
    topics: Sequence[CurriculumTopic] = CURRICULUM_TOPICS,
) -> CurriculumIngestionReport:
    """Fetch a curated set of Wikipedia articles, export by level, and index them."""
    if not topics:
        raise ValueError("At least one curriculum topic is required.")

    root = Path(output_dir)
    files: list[Path] = []
    total_sections = 0
    total_chunks = 0

    for topic in topics:
        article = extractor.extract(topic.title)
        sections = _apply_curriculum_override(article.sections, topic.advanced)
        if not sections:
            raise ValueError(f"Wikipedia article {topic.title!r} has no sections to index.")

        sections_by_level = {level: [] for level in LEARNING_LEVELS}
        documents = []
        for section in sections:
            sections_by_level[section.learning_level].append(section)
            documents.append(
                (
                    article.source_url,
                    section.text,
                    {
                        "article_title": article.title,
                        "subject": topic.subject,
                        "section_title": section.title,
                        "section_index": section.section_index,
                        "learning_level": section.learning_level,
                        "source_url": article.source_url,
                        "license": "CC BY-SA 4.0",
                    },
                )
            )

        total_chunks += pipeline.index_documents(documents)
        total_sections += len(sections)
        files.extend(_save_level_files(article, topic, sections_by_level, root))

    return CurriculumIngestionReport(
        articles=len(topics),
        sections=total_sections,
        chunks=total_chunks,
        files=tuple(files),
    )


def _apply_curriculum_override(
    sections: list[WikipediaSection],
    advanced: bool,
) -> list[WikipediaSection]:
    if not advanced:
        return sections
    return [
        WikipediaSection(
            title=section.title,
            text=section.text,
            learning_level="avancado",
            section_index=section.section_index,
        )
        for section in sections
    ]


def _save_level_files(
    article: WikipediaArticle,
    topic: CurriculumTopic,
    sections_by_level: dict[str, list[WikipediaSection]],
    root: Path,
) -> list[Path]:
    slug = re.sub(
        r"[^\w.-]+",
        "_",
        topic.title.casefold(),
        flags=re.UNICODE,
    ).strip("_.") or "wikipedia_article"
    saved_files = []

    for level, sections in sections_by_level.items():
        directory = root / _LEVEL_DIRECTORIES[level]
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{slug}.json"
        payload = {
            "title": article.title,
            "subject": topic.subject,
            "language": article.language,
            "source_url": article.source_url,
            "retrieved_at": article.retrieved_at,
            "license": "CC BY-SA 4.0; attribute the original Wikipedia article",
            "learning_level": level,
            "sections": [
                {
                    "title": section.title,
                    "text": section.text,
                    "section_index": section.section_index,
                }
                for section in sections
            ],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        saved_files.append(output_path)

    return saved_files
