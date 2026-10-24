from src.cognition.semantic_wiki_discoverer import SemanticWikiDiscovererSkill
from src.cognition.vocabulary_builder import PersonalVocabularyBuilder
from src.llm.factory import create_llm_client
from src.llm.utils import load_model_config
from src.utils.wiki_extractor import WikipediaExtractor


def run_local_pipeline(
    question: str = "O que é inteligência artificial e como ela aprende?",
) -> str:
    config = load_model_config()
    wikipedia_config = config.get("wikipedia", {})
    extractor = WikipediaExtractor(
        language=wikipedia_config.get("language", "pt"),
        timeout=wikipedia_config.get("timeout_seconds", 15),
    )
    wiki_discoverer = SemanticWikiDiscovererSkill(extractor=extractor)
    context = wiki_discoverer.fetch_best_context(question)

    cognition_config = config.get("cognition", {})
    vocabulary = PersonalVocabularyBuilder(
        cognition_config.get("vocabulary_path", "data/personal_lexicon.json")
    )
    vocabulary_context = vocabulary.format_relevant_context(question)
    local_config = config.get("local", {})
    ai_engine = create_llm_client(
        config={"default_provider": "local", "local": local_config}
    )
    prompt = (
        "Responda à pergunta usando apenas o contexto abaixo. "
        "Use o vocabulário pessoal apenas como apoio para esclarecer termos; "
        "não o trate como fonte factual superior ao artigo. "
        "Se o artigo não contiver a resposta, diga isso claramente.\n\n"
        f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
    )
    if vocabulary_context:
        prompt = (
            "Vocabulário pessoal recuperado como memória auxiliar:\n"
            f"{vocabulary_context}\n\n{prompt}"
        )
    return ai_engine.generate(
        prompt,
        max_new_tokens=local_config.get("max_new_tokens", 150),
    )


if __name__ == "__main__":
    print(run_local_pipeline())