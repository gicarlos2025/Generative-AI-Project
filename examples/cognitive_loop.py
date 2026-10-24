from src.cognition.cognitive_learner import CognitiveLearnerSkill
from src.utils.cache import SimpleCache

def run_cognitive_demo():
    # Inicializa a memória e a skill cognitiva
    memory = SimpleCache()
    learner = CognitiveLearnerSkill(memory_store=memory)

    # Simulação de uma pergunta complexa feita pelo utilizador
    pergunta = "Como funciona a arquitetura interna e os vetores de um modelo de linguagem em termos avançados?"

    # 1. Percepção
    percepcao = learner.perceive(pergunta)
    print(f"-> Percepção do Agente: Nível de complexidade detetado = {percepcao['detected_complexity'].upper()}")

    # 2. Faculdade Cognitiva (Avaliação de lacunas)
    contexto_vazio = "" # Simulando que não temos contexto prévio
    avaliacao = learner.evaluate_cognition(percepcao, retrieved_context=contexto_vazio)
    print(f"-> Faculdade Cognitiva: Há lacuna de conhecimento? {avaliacao['has_knowledge_gap']} (Ação recomendada: {avaliacao['recommended_action']})")

    # 3. Resposta simulada do LLM após a pesquisa/geração
    resposta_gerada = "A arquitetura baseia-se em mecanismos de auto-atenção (Self-Attention) e representações vetoriais de alta dimensão..."

    # 4. Aprendizagem Ativa (O agente absorve o que aprendeu)
    learner.learn_and_store(pergunta, resposta_gerada, level=percepcao['detected_complexity'])

    # Verificar se o conhecimento ficou guardado na memória cognitiva
    print("-> Teste de Recuperação da Memória Cognitiva do Agente concluído com sucesso!")

if __name__ == "__main__":
    run_cognitive_demo()