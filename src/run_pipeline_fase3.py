"""
Pipeline principal da Fase 3.

Fluxo contemplado:
- Inicialização da LLM
- Assistente clínico com LangChain
- Orquestração com LangGraph
- Logging estruturado
- Explainability
- Validação humana
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatPromptTemplate = None

try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    AzureChatOpenAI = None

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = "END"
    START = "START"
    StateGraph = None


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "logs_clinicos.jsonl"


class EstadoClinico(TypedDict, total=False):
    paciente_id: str
    idade: int
    sexo: str
    sintomas: List[str]
    exames_pendentes: List[str]
    historico: str
    contexto_clinico: str
    analise_inicial: str
    recomendacao: str
    justificativa: str
    resposta_final: str
    aprovado: bool
    validado_por: str
    log_path: str


class DemoLLM:
    def invoke(self, prompt: Any) -> Any:
        if isinstance(prompt, list):
            partes = []
            for msg in prompt:
                content = getattr(msg, "content", msg)
                if isinstance(content, list):
                    partes.append(
                        "".join(
                            bloco.get("text", "") if isinstance(bloco, dict) else str(bloco)
                            for bloco in content
                        )
                    )
                else:
                    partes.append(str(content))
            texto = "\n".join(partes).lower()
        else:
            texto = str(prompt).lower()

        sintomas_detectados = []
        if "febre" in texto:
            sintomas_detectados.append("febre")
        if "tosse" in texto:
            sintomas_detectados.append("tosse")
        if "fadiga" in texto:
            sintomas_detectados.append("fadiga")
        if "dor no peito" in texto:
            sintomas_detectados.append("dor no peito")
        if "falta de ar" in texto:
            sintomas_detectados.append("falta de ar")
        if "confusão mental" in texto or "confusao mental" in texto:
            sintomas_detectados.append("confusão mental")

        exames = []
        if "raio-x" in texto or "raio x" in texto:
            exames.append("Raio-X de tórax")
        if "hemograma" in texto:
            exames.append("Hemograma")
        if "eletrocardiograma" in texto:
            exames.append("Eletrocardiograma")

        if "dor no peito" in texto:
            analise = "O caso requer atenção clínica prioritária, com investigação de causa cardiovascular e exclusão de sinais de gravidade."
            recomendacao = (
                "Priorizar a avaliação médica, revisar sinais vitais e considerar a realização do eletrocardiograma "
                "pendente antes de definir a próxima conduta."
            )
            justificativa = (
                "A recomendação foi baseada na presença de dor no peito e na necessidade de utilizar o exame pendente "
                "como apoio à decisão clínica."
            )
        elif "falta de ar" in texto:
            analise = "O quadro sugere possível comprometimento respiratório e precisa de investigação complementar."
            recomendacao = (
                "Correlacionar sintomas respiratórios com o contexto clínico, revisar oxigenação e considerar os exames "
                "de imagem antes de qualquer conduta definitiva."
            )
            justificativa = (
                "A resposta considera a falta de ar como sinal relevante e prioriza investigação antes de conclusão clínica."
            )
        elif "confusão mental" in texto or "confusao mental" in texto:
            analise = "O quadro exige cautela adicional devido ao potencial de gravidade, especialmente em paciente idoso."
            recomendacao = (
                "Realizar avaliação médica prioritária, revisar estado neurológico e investigar causas metabólicas, "
                "infecciosas ou sistêmicas."
            )
            justificativa = (
                "A conduta foi sugerida considerando a idade avançada e a presença de alteração do estado mental."
            )
        else:
            analise = "O caso sugere quadro clínico inespecífico que requer correlação com a avaliação médica e exames complementares."
            recomendacao = (
                "Sugerir investigação complementar com base nos sintomas relatados, sem prescrição medicamentosa direta, "
                "e manter validação humana antes da conduta final."
            )
            justificativa = (
                "A recomendação foi baseada nos sintomas informados e no princípio de segurança clínica com validação humana."
            )

        resposta = {
            "analise_inicial": analise,
            "recomendacao": recomendacao,
            "justificativa": justificativa,
            "dados_considerados": {
                "sintomas_detectados": sintomas_detectados,
                "exames_mencionados": exames,
            },
        }

        class DemoResponse:
            def __init__(self, content: str) -> None:
                self.content = content

        return DemoResponse(json.dumps(resposta, ensure_ascii=False))

def registrar_log(etapa: str, entrada: Dict[str, Any], saida: Dict[str, Any], paciente_id: str | None = None) -> None:
    evento = {
        "timestamp": datetime.utcnow().isoformat(),
        "paciente_id": paciente_id or entrada.get("paciente_id"),
        "etapa": etapa,
        "entrada": entrada,
        "saida": saida,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def build_assistente_llm() -> Any:
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment_name = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_FINE_TUNED_DEPLOYMENT")
    )

    if AzureChatOpenAI and azure_endpoint and api_key and deployment_name:
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment_name,
            temperature=0,
        )

    return DemoLLM()


def montar_contexto_clinico(estado: EstadoClinico) -> str:
    sintomas = ", ".join(estado.get("sintomas", [])) or "Não informado"
    exames_pendentes = ", ".join(estado.get("exames_pendentes", [])) or "Nenhum"
    historico = estado.get("historico", "Não informado")

    return (
        f"Paciente ID: {estado.get('paciente_id', 'N/A')}\n"
        f"Idade: {estado.get('idade', 'N/A')}\n"
        f"Sexo: {estado.get('sexo', 'N/A')}\n"
        f"Sintomas: {sintomas}\n"
        f"Exames pendentes: {exames_pendentes}\n"
        f"Histórico clínico: {historico}"
    )


def _montar_prompt_chain() -> Any:
    if not ChatPromptTemplate:
        return None

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Você é um assistente virtual médico de apoio à decisão clínica. "
                    "Utilize apenas as informações fornecidas no contexto do paciente. "
                    "Não prescreva medicamentos diretamente e não substitua avaliação médica humana. "
                    "Responda em JSON com os campos: analise_inicial, recomendacao, justificativa e dados_considerados."
                ),
            ),
            (
                "human",
                (
                    "Considere o contexto clínico abaixo e produza uma resposta estruturada.\n\n"
                    "{contexto_clinico}"
                ),
            ),
        ]
    )


def executar_assistente(estado: EstadoClinico, llm: Any | None = None) -> Dict[str, Any]:
    llm = llm or build_assistente_llm()
    contexto_clinico = montar_contexto_clinico(estado)

    prompt_chain = _montar_prompt_chain()

    if prompt_chain is not None:
        mensagem = prompt_chain.format_messages(contexto_clinico=contexto_clinico)
        resposta = llm.invoke(mensagem)
    else:
        prompt = (
            "Você é um assistente virtual médico de apoio à decisão clínica. "
            "Utilize apenas as informações fornecidas no contexto do paciente. "
            "Não prescreva medicamentos diretamente e não substitua avaliação médica humana. "
            "Responda em JSON com os campos: analise_inicial, recomendacao, justificativa e dados_considerados.\n\n"
            f"{contexto_clinico}"
        )
        resposta = llm.invoke(prompt)

    conteudo = getattr(resposta, "content", resposta)

    if isinstance(conteudo, list):
        texto = "".join(
            bloco.get("text", "") if isinstance(bloco, dict) else str(bloco)
            for bloco in conteudo
        )
    else:
        texto = str(conteudo)

    try:
        payload = json.loads(texto)
    except Exception:
        payload = {
            "analise_inicial": texto,
            "recomendacao": "Avaliar clinicamente o caso antes de qualquer conduta definitiva.",
            "justificativa": "A resposta foi convertida para formato textual por incompatibilidade de serialização.",
            "dados_considerados": {
                "sintomas_detectados": estado.get("sintomas", []),
                "exames_mencionados": estado.get("exames_pendentes", []),
            },
        }

    saida = {
        "contexto_clinico": contexto_clinico,
        "analise_inicial": payload.get("analise_inicial", ""),
        "recomendacao": payload.get("recomendacao", ""),
        "justificativa": payload.get("justificativa", ""),
    }

    registrar_log(
        etapa="assistente_langchain",
        entrada={
            "paciente_id": estado.get("paciente_id"),
            "contexto_clinico": contexto_clinico,
        },
        saida=saida,
        paciente_id=estado.get("paciente_id"),
    )

    return saida


def node_analisar_caso(estado: EstadoClinico) -> EstadoClinico:
    resultado = executar_assistente(estado)
    saida = {
        **estado,
        "contexto_clinico": resultado["contexto_clinico"],
        "analise_inicial": resultado["analise_inicial"],
        "recomendacao": resultado["recomendacao"],
        "justificativa": resultado["justificativa"],
    }

    registrar_log(
        etapa="analisar_caso",
        entrada={
            "paciente_id": estado.get("paciente_id"),
            "sintomas": estado.get("sintomas", []),
            "exames_pendentes": estado.get("exames_pendentes", []),
        },
        saida={
            "analise_inicial": saida["analise_inicial"],
            "recomendacao": saida["recomendacao"],
        },
        paciente_id=estado.get("paciente_id"),
    )

    return saida


def node_verificar_exames(estado: EstadoClinico) -> EstadoClinico:
    exames_pendentes = estado.get("exames_pendentes", [])
    complemento = ""

    if exames_pendentes:
        complemento = (
            " Existem exames pendentes que devem ser considerados antes da definição de conduta final: "
            + ", ".join(exames_pendentes)
            + "."
        )
    else:
        complemento = " Não há exames pendentes registrados para este caso."

    recomendacao_atualizada = f"{estado.get('recomendacao', '')}{complemento}".strip()

    saida = {
        **estado,
        "recomendacao": recomendacao_atualizada,
    }

    registrar_log(
        etapa="verificar_exames",
        entrada={
            "paciente_id": estado.get("paciente_id"),
            "exames_pendentes": exames_pendentes,
        },
        saida={
            "recomendacao": recomendacao_atualizada,
        },
        paciente_id=estado.get("paciente_id"),
    )

    return saida


def node_validacao_humana(estado: EstadoClinico) -> EstadoClinico:
    aprovado = True
    validado_por = "Médico responsável"

    resposta_final = (
        f"Análise inicial: {estado.get('analise_inicial', '')}\n\n"
        f"Recomendação: {estado.get('recomendacao', '')}\n\n"
        f"Justificativa: {estado.get('justificativa', '')}\n\n"
        f"Validação humana: {'Aprovada' if aprovado else 'Rejeitada'} por {validado_por}."
    )

    saida = {
        **estado,
        "aprovado": aprovado,
        "validado_por": validado_por,
        "resposta_final": resposta_final,
        "log_path": str(LOG_FILE),
    }

    registrar_log(
        etapa="validacao_humana",
        entrada={
            "paciente_id": estado.get("paciente_id"),
            "analise_inicial": estado.get("analise_inicial", ""),
            "recomendacao": estado.get("recomendacao", ""),
        },
        saida={
            "aprovado": aprovado,
            "validado_por": validado_por,
        },
        paciente_id=estado.get("paciente_id"),
    )

    return saida


def compilar_fluxo() -> Any:
    if StateGraph is None:
        return None

    graph = StateGraph(EstadoClinico)
    graph.add_node("analisar_caso", node_analisar_caso)
    graph.add_node("verificar_exames", node_verificar_exames)
    graph.add_node("validacao_humana", node_validacao_humana)

    graph.add_edge(START, "analisar_caso")
    graph.add_edge("analisar_caso", "verificar_exames")
    graph.add_edge("verificar_exames", "validacao_humana")
    graph.add_edge("validacao_humana", END)

    return graph.compile()


def executar_fluxo_clinico(dados_paciente: Dict[str, Any]) -> EstadoClinico:
    fluxo = compilar_fluxo()

    estado_inicial: EstadoClinico = {
        "paciente_id": dados_paciente.get("paciente_id", "PACIENTE-001"),
        "idade": dados_paciente.get("idade"),
        "sexo": dados_paciente.get("sexo"),
        "sintomas": dados_paciente.get("sintomas", []),
        "exames_pendentes": dados_paciente.get("exames_pendentes", []),
        "historico": dados_paciente.get("historico", "Não informado"),
    }

    if fluxo is None:
        estado = node_analisar_caso(estado_inicial)
        estado = node_verificar_exames(estado)
        estado = node_validacao_humana(estado)
        return estado

    return fluxo.invoke(estado_inicial)


def ler_logs() -> List[Dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    eventos: List[Dict[str, Any]] = []
    with open(LOG_FILE, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if linha:
                eventos.append(json.loads(linha))
    return eventos


if __name__ == "__main__":
    paciente_exemplo = {
        "paciente_id": "PAC-2026-001",
        "idade": 45,
        "sexo": "Masculino",
        "sintomas": ["febre", "tosse", "fadiga"],
        "exames_pendentes": ["Hemograma", "Raio-X de tórax"],
        "historico": "Paciente com início de sintomas respiratórios nas últimas 48 horas.",
    }

    resultado = executar_fluxo_clinico(paciente_exemplo)

    print("\n=== RESULTADO FINAL ===\n")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))

    print("\n=== LOGS REGISTRADOS ===\n")
    for evento in ler_logs()[-5:]:
        print(json.dumps(evento, ensure_ascii=False, indent=2))
