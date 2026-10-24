from src.rag.pipeline import RAGPipeline
from src.cognition.cognitive_learner import CognitiveLearnerSkill
from src.cognition.semantic_wiki_discoverer import SemanticWikiDiscovererSkill
from src.cognition.vocabulary_builder import PersonalVocabularyBuilder
from src.utils.logger import logger
from src.utils.wiki_extractor import LEARNING_LEVELS, WikipediaArticle, WikipediaExtractor


INITIAL_RESPONSE = (
    "Com base no nosso repositório de conhecimento geral, aqui está a visão geral sobre "
    "[Tema]. Este resumo foca-se nos pontos essenciais extraídos das fontes primárias. "
    "Podemos detalhar os processos ou casos específicos conforme o seu interesse."
)


class WikipediaChat:
    """Search Wikipedia for every question, index the article, and return cited excerpts."""

    def __init__(
        self,
        extractor: WikipediaExtractor,
        pipeline: RAGPipeline,
        cognitive_learner: CognitiveLearnerSkill | None = None,
        wiki_discoverer: SemanticWikiDiscovererSkill | None = None,
        vocabulary_builder: PersonalVocabularyBuilder | None = None,
    ) -> None:
        self.extractor = extractor
        self.pipeline = pipeline
        self.cognitive_learner = cognitive_learner
        self.wiki_discoverer = wiki_discoverer or SemanticWikiDiscovererSkill(
            extractor=extractor
        )
        self.vocabulary_builder = vocabulary_builder

    def answer(
        self,
        question: str,
        learning_level: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        if not question.strip():
            raise ValueError("question must not be empty.")
        if learning_level is not None and learning_level not in LEARNING_LEVELS:
            raise ValueError(
                f"Unsupported learning level {learning_level!r}; choose one of: "
                f"{', '.join(LEARNING_LEVELS)}."
            )
        search_query = question
        if conversation_history and _is_context_dependent(question):
            previous_question = next(
                (
                    message["content"]
                    for message in reversed(conversation_history)
                    if message["role"] == "user"
                ),
                None,
            )
            if previous_question:
                search_query = f"{previous_question} {question}"

        perception = (
            self.cognitive_learner.perceive(question)
            if self.cognitive_learner is not None
            else None
        )
        effective_level = learning_level or (
            perception["detected_complexity"] if perception is not None else None
        )
        prior_memory = (
            self.cognitive_learner.recall(search_query, level=effective_level)
            if self.cognitive_learner is not None
            else None
        )
        initial_assessment = (
            self.cognitive_learner.evaluate_cognition(
                perception,
                retrieved_context=(
                    prior_memory["synthesized_answer"] if prior_memory is not None else None
                ),
                prior_memory=prior_memory,
            )
            if self.cognitive_learner is not None and perception is not None
            else None
        )

        article = self.wiki_discoverer.fetch_best_article(search_query)
        self._index_article(article)
        sections = article.sections
        if not sections:
            raise LookupError(f"Wikipedia article {article.title!r} has no readable text.")

        assessment = None
        if self.cognitive_learner is not None and perception is not None:
            assessment = self.cognitive_learner.evaluate_cognition(
                perception,
                retrieved_context="\n".join(section.text for section in sections),
                prior_memory=prior_memory,
            )
            assessment["initial_assessment"] = initial_assessment

        opening = INITIAL_RESPONSE.replace("[Tema]", article.title)
        answer = (
            f"{opening}\n\n"
            + "\n\n".join(
                f"## {section.title}\n\n{section.text}" for section in sections
            )
            + f"\n\nFonte: {article.source_url}"
            + "\n\nConteúdo da Wikipédia, disponível sob a licença CC BY-SA 4.0."
        )
        try:
            image = self.wiki_discoverer.fetch_article_image(article.title)
        except (RuntimeError, ValueError) as error:
            logger.warning(
                "Could not fetch the Wikipedia article image for %r: %s",
                article.title,
                error,
            )
            image = None
        if image is not None:
            alt_text = article.title.replace("[", "").replace("]", "").replace("\n", " ")
            source_url = image["source_url"].replace("(", "%28").replace(")", "%29")
            image_url = image["url"].replace("(", "%28").replace(")", "%29")
            image_attribution = (
                f"Imagem e licença: [detalhes do ficheiro]({source_url})"
                + (f" · Autor: {image['artist']}" if image.get("artist") else "")
                + (f" · Licença: {image['license']}" if image.get("license") else "")
            )
            answer += (
                f"\n\n![Ilustração: {alt_text}]({image_url})"
                f"\n\n{image_attribution}"
            )
        if self.cognitive_learner is not None and perception is not None:
            self.cognitive_learner.learn_and_store(
                search_query,
                answer,
                level=effective_level or perception["detected_complexity"],
                sources=[article.source_url],
                assessment=assessment,
            )
        if self.vocabulary_builder is not None:
            introduction = next(
                (section.text for section in sections if section.title == "Introdução"),
                sections[0].text,
            )
            self.vocabulary_builder.acquire_sign(
                term=article.title,
                meaning=introduction,
                context=search_query,
                level=effective_level or "basico",
                source=article.source_url,
            )
        return answer

    def _index_article(self, article: WikipediaArticle) -> None:
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
        self.pipeline.index_documents(documents)


def _is_context_dependent(question: str) -> bool:
    """Only attach past questions for short anaphoric follow-ups."""
    tokens = question.casefold().strip(" .?!,").split()
    if len(tokens) > 8:
        return False
    references = {
        "isso", "isto", "ele", "ela", "eles", "elas", "esse", "essa", "esses",
        "essas", "dessa", "desse", "disto", "disso", "dela", "dele", "deles",
        "delas", "aquela", "aquele", "aquilo",
    }
    return any(token.strip(".,?!") in references for token in tokens)
