import re
import unicodedata
from difflib import SequenceMatcher

from src.utils.wiki_extractor import WikipediaArticle, WikipediaExtractor


_STOP_WORDS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "sobre", "entre",
    "que", "qual", "quais", "como", "quando", "onde", "porque", "porquê", "e", "ou",
    "mas", "se", "me", "te", "lhe", "isso", "isto", "aquele", "aquela", "ser", "estar",
    "ter", "fazer", "poder", "gostaria", "entender", "explicar", "explique", "quero",
    "preciso", "ajude", "mais", "muito", "pouco", "influencia", "influenciam",
    "influenciar", "papel", "efeito", "impacto", "funciona",
    "fale", "falar", "fala", "sobre", "das", "dos", "pelas", "pelos", "planta",
    "plantas", "humano", "humanos",
    "processo", "processos", "tornar", "retarda", "retardar",
}
_TOKEN_PATTERN = re.compile(r"[^\W_]{3,}", flags=re.UNICODE)
_MAX_QUERY_LENGTH = 500
_MAX_SEARCH_QUERIES = 6


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _tokens(value: str) -> list[str]:
    return _TOKEN_PATTERN.findall(_normalize(value))


class SemanticWikiDiscovererSkill:
    """Discover relevant Wikipedia articles from question concepts and context."""

    def __init__(
        self,
        language: str = "pt",
        extractor: WikipediaExtractor | None = None,
    ) -> None:
        self.extractor = extractor or WikipediaExtractor(language=language)

    def extract_concepts(self, context: str) -> list[str]:
        """Select informative unique terms without requiring an article title."""
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must not be empty.")
        concepts: list[str] = []
        for token in _tokens(context):
            if token not in _STOP_WORDS and token not in concepts:
                concepts.append(token)
        return concepts

    def search_related_topics(
        self,
        context: str,
        limit: int = 5,
    ) -> list[dict[str, str]]:
        """Use MediaWiki full-text search and locally rerank matches by concepts."""
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must not be empty.")
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10.")

        query = " ".join(context.split())[:_MAX_QUERY_LENGTH]
        concepts = self.extract_concepts(query)
        queries = [query]
        compact_query = " ".join(concepts[:12])
        if compact_query and compact_query.casefold() != query.casefold():
            queries.append(compact_query)
        phrases = [
            f"{left} {right}"
            for left, right in zip(concepts, concepts[1:])
            if len(left) > 2 and len(right) > 2
        ]
        queries.extend(phrases[: _MAX_SEARCH_QUERIES - len(queries)])
        queries = list(dict.fromkeys(item for item in queries if item))[:_MAX_SEARCH_QUERIES]

        best_by_title: dict[str, tuple[float, int, dict[str, str]]] = {}
        for query_rank, candidate_query in enumerate(queries):
            try:
                candidates = self.extractor.search_related_topics(candidate_query, limit=8)
            except LookupError:
                continue
            for rank, candidate in enumerate(candidates):
                title = candidate["title"]
                snippet = candidate.get("snippet", "")
                normalized_title = _normalize(title)
                normalized_snippet = _normalize(snippet)
                title_tokens = set(_tokens(title))
                snippet_tokens = set(_tokens(snippet))

                def term_score(concept: str, tokens: set[str]) -> float:
                    if concept in tokens:
                        return 1.0
                    if any(
                        min(len(concept), len(token)) >= 4
                        and (concept.startswith(token) or token.startswith(concept))
                        for token in tokens
                    ):
                        return 0.85
                    return max(
                        (
                            SequenceMatcher(None, concept, token).ratio()
                            for token in tokens
                        ),
                        default=0.0,
                    )

                title_scores = [term_score(concept, title_tokens) for concept in concepts]
                snippet_scores = [term_score(concept, snippet_tokens) for concept in concepts]
                title_exact = sum(score == 1.0 for score in title_scores)
                title_close = sum(score >= 0.82 for score in title_scores)
                snippet_exact = sum(score == 1.0 for score in snippet_scores)
                snippet_close = sum(score >= 0.88 for score in snippet_scores)
                phrase_matches = sum(
                    phrase in normalized_title
                    or phrase in normalized_snippet
                    for phrase in phrases
                )
                query_terms = set(_tokens(candidate_query))
                query_overlap = sum(
                    max(
                        (term_score(term, title_tokens) for term in query_terms),
                        default=0.0,
                    )
                    for term in query_terms
                )
                score = (
                    title_exact * 4
                    + title_close * 2
                    + snippet_exact
                    + snippet_close * 0.25
                    + phrase_matches * 5
                    + query_overlap * 0.5
                    + 0.1 / (rank + 1)
                    - query_rank * 0.02
                )
                current = best_by_title.get(normalized_title)
                if current is None or score > current[0]:
                    best_by_title[normalized_title] = (
                        score,
                        rank + query_rank * 10,
                        candidate,
                    )

        ranked = sorted(
            best_by_title.values(),
            key=lambda item: (-item[0], item[1], item[2]["title"].casefold()),
        )
        topics = [candidate for _, _, candidate in ranked[:limit]]
        if not topics:
            raise LookupError(f"Wikipedia did not find an article for {query!r}.")
        return topics

    def fetch_best_context(self, context: str) -> str:
        """Fetch and format the best matching article with section and source labels."""
        article = self.fetch_best_article(context)
        sections = article.sections[:5]
        if not sections:
            raise LookupError(f"Wikipedia article {article.title!r} has no readable context.")

        lines = [
            f"Artigo: {article.title}",
            f"Fonte: {article.source_url}",
            "Licença: CC BY-SA 4.0",
        ]
        lines.extend(f"{section.title}: {section.text}" for section in sections)
        return "\n\n".join(lines)

    def fetch_best_article(self, context: str) -> WikipediaArticle:
        """Fetch the best matching article for RAG indexing."""
        topics = self.search_related_topics(context, limit=1)
        return self.extractor.extract(topics[0]["title"])

    def fetch_article_image(self, title: str) -> dict[str, str] | None:
        """Fetch an article thumbnail and its Wikimedia file attribution metadata."""
        return self.extractor.fetch_article_image(title)
