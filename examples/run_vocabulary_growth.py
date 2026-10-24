from src.cognition.vocabulary_builder import PersonalVocabularyBuilder
from src.cognition.semantic_wiki_discoverer import SemanticWikiDiscovererSkill


def run_vocabulary_demo(term: str = "Algoritmo") -> dict[str, object]:
    """Fetch a Wikipedia article and persist its sourced lead in the local lexicon."""
    vocab_builder = PersonalVocabularyBuilder()
    wiki_discoverer = SemanticWikiDiscovererSkill(language="pt")

    article = wiki_discoverer.fetch_best_article(term)
    introduction = next(
        (section.text for section in article.sections if section.title == "Introdução"),
        article.sections[0].text if article.sections else "",
    )
    if not introduction:
        raise LookupError(f"Wikipedia article {article.title!r} has no readable introduction.")

    vocab_builder.acquire_sign(
        term=article.title,
        meaning=introduction,
        context=f"Pesquisa: {term}",
        level="basico",
        source=article.source_url,
    )

    record = vocab_builder.consult_lexicon(article.title)
    if record is None:
        raise RuntimeError(f"Could not read vocabulary record for {article.title!r}.")
    print(f"Termo: {record['term']}")
    print(f"Nível registado: {record['maturity_level']}")
    print(f"Exposições: {record['exposure_count']}")
    print(f"Significado: {record['meanings'][0]}")
    print(f"Fonte: {record['sources'][0]}")
    return record


if __name__ == "__main__":
    run_vocabulary_demo()