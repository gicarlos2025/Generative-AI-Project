from src.llm.factory import create_llm_client
from src.llm.utils import load_model_config
from src.cognition.vocabulary_builder import PersonalVocabularyBuilder
from src.rag.embeddings import create_embedding_provider
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import SQLiteVectorStore
from src.utils.secrets import load_local_secrets


def run_example() -> str:
    pipeline = build_rag_pipeline()
    pipeline.index_text(
        source="mvp-introduction",
        text=(
            "Generative artificial intelligence creates new content such as text, images, "
            "audio, and code by learning patterns from large datasets. "
            "Retrieval-augmented generation (RAG) retrieves relevant source passages from "
            "a knowledge base and supplies them as context to a language model. "
            "A vector database stores embedding vectors and ranks passages by similarity "
            "to the user's question."
        ),
    )
    result = pipeline.ask("What does retrieval-augmented generation do?")

    print(f"Result: {result}")
    return result


def build_rag_pipeline(config: dict | None = None) -> RAGPipeline:
    load_local_secrets()
    config = config if config is not None else load_model_config()
    rag_config = config.get("rag", {})
    cognition_config = config.get("cognition", {})
    vector_store = SQLiteVectorStore(
        db_path=rag_config.get("database_path", "data/vectors.sqlite3"),
        embedding_provider=create_embedding_provider(rag_config),
    )
    return RAGPipeline(
        vector_store=vector_store,
        llm_client=create_llm_client(config=config),
        chunk_size=rag_config.get("chunk_size", 160),
        chunk_overlap=rag_config.get("chunk_overlap", 30),
        vocabulary_builder=PersonalVocabularyBuilder(
            cognition_config.get("vocabulary_path", "data/personal_lexicon.json")
        ),
    )


if __name__ == "__main__":
    run_example()
