from src.cognition.semantic_wiki_discoverer import SemanticWikiDiscovererSkill

def run_discovery_test():
    discoverer = SemanticWikiDiscovererSkill(language="pt")

    # Pergunta vaga ou baseada num contexto complexo (sem nome de artigo exato)
    pergunta_utilizador = "Gostaria de entender como as máquinas aprendem a reconhecer padrões visuais e rostos humanos nas fotos."

    print(f"-> Pergunta do Utilizador: \"{pergunta_utilizador}\"")

    # 1. O agente descobre os temas relacionados de forma autónoma
    topicos_encontrados = discoverer.search_related_topics(pergunta_utilizador, limit=3)
    print("\n-> Tópicos/Artigos descobertos pela Wikipédia com base no contexto:")
    for i, t in enumerate(topicos_encontrados, 1):
        print(f"   {i}. **{t['title']}** -> Trecho: {t['snippet'][:100]}...")

    # 2. O agente puxa o melhor contexto automaticamente para o RAG
    contexto_ideal = discoverer.fetch_best_context(pergunta_utilizador)
    if contexto_ideal:
        print(f"\n-> Contexto Semântico Extraído com Sucesso ({len(contexto_ideal)} caracteres):\n")
        print(contexto_ideal[:350] + "...\n[Conteúdo pronto para alimentar o prompt do LLM]")

if __name__ == "__main__":
    run_discovery_test()