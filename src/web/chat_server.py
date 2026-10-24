import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.rag.pipeline import RAGPipeline
from src.utils.wiki_extractor import LEARNING_LEVELS
from src.utils.wikipedia_chat import WikipediaChat


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"
_MAX_REQUEST_BYTES = 32_768
_LOG = logging.getLogger("app.web")


def _is_loopback_address(host: str) -> bool:
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return host.casefold() == "localhost"


def _is_valid_browser_host(host: str) -> bool:
    hostname = host.casefold().rstrip(".")
    if _is_loopback_address(hostname):
        return True
    return hostname.endswith(".app.github.dev") or hostname.endswith(".github.dev")


def create_chat_server(
    pipeline: RAGPipeline,
    port: int = 8000,
    top_k: int = 4,
    wikipedia_chat: WikipediaChat | None = None,
) -> ThreadingHTTPServer:
    """Create a threaded chat server bound to IPv4 loopback only."""
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    class ChatRequestHandler(BaseHTTPRequestHandler):
        server_version = "GenerativeAIChat/1.0"
        sys_version = ""

        def do_GET(self) -> None:
            if not self._request_is_local():
                self._send_error(403, "A interface só aceita pedidos locais.")
                return

            if self.path == "/":
                try:
                    content = _INDEX_FILE.read_bytes()
                except OSError:
                    _LOG.exception("Could not load the chat interface.")
                    self._send_error(500, "Não foi possível carregar a interface.")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return

            if self.path == "/api/status":
                try:
                    indexed_chunks = pipeline.vector_store.count()
                except Exception:
                    _LOG.exception("Could not read vector store status.")
                    self._send_error(500, "Não foi possível consultar a biblioteca.")
                    return
                self._send_json(
                    200,
                    {
                        "status": "ready",
                        "indexed_chunks": indexed_chunks,
                        "levels": list(LEARNING_LEVELS),
                        "generation_mode": (
                            "wikipedia"
                            if wikipedia_chat is not None
                            else "llm" if pipeline.llm_client.is_configured else "extractive"
                        ),
                    },
                )
                return

            self._send_error(404, "Rota não encontrada.")

        def do_POST(self) -> None:
            if not self._request_is_local():
                self._send_error(403, "A interface só aceita pedidos locais.")
                return
            if self.path != "/api/chat":
                self._send_error(404, "Rota não encontrada.")
                return
            if self.headers.get_content_type() != "application/json":
                self._send_error(415, "Envie o pedido em formato JSON.")
                return

            content_length = self.headers.get("Content-Length")
            try:
                body_size = int(content_length) if content_length is not None else -1
            except ValueError:
                body_size = -1
            if body_size < 0 or body_size > _MAX_REQUEST_BYTES:
                self._send_error(413, "O pedido está vazio ou excede o limite permitido.")
                return

            try:
                payload: Any = json.loads(self.rfile.read(body_size))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_error(400, "O corpo do pedido não contém JSON válido.")
                return
            if not isinstance(payload, dict):
                self._send_error(400, "O corpo do pedido deve ser um objeto JSON.")
                return

            question = payload.get("question")
            level = payload.get("level")
            history = payload.get("history", [])
            if not isinstance(question, str) or not question.strip():
                self._send_error(400, "Escreva uma pergunta antes de enviar.")
                return
            if len(question) > 8_000:
                self._send_error(413, "A pergunta excede o limite de 8000 caracteres.")
                return
            if level is not None and level not in LEARNING_LEVELS:
                self._send_error(400, "Selecione um nível de aprendizagem válido.")
                return
            if not isinstance(history, list) or len(history) > 12:
                self._send_error(400, "O histórico deve conter no máximo 12 mensagens.")
                return
            if any(
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                or len(message["content"]) > 1_000
                for message in history
            ):
                self._send_error(400, "O histórico da conversa contém uma mensagem inválida.")
                return

            try:
                if wikipedia_chat is not None:
                    answer = wikipedia_chat.answer(
                        question,
                        learning_level=level,
                        conversation_history=history,
                    )
                else:
                    answer = pipeline.ask(
                        question,
                        top_k=top_k,
                        learning_level=level,
                        conversation_history=history,
                    )
            except LookupError as error:
                self._send_error(404, str(error))
                return
            except ValueError as error:
                self._send_error(400, str(error))
                return
            except Exception:
                _LOG.exception("RAG chat request failed.")
                self._send_error(502, "O agente não conseguiu responder. Tente novamente.")
                return

            self._send_json(
                200,
                {
                    "answer": answer,
                    "generation_mode": (
                        "wikipedia"
                        if wikipedia_chat is not None
                        else "llm" if pipeline.llm_client.is_configured else "extractive"
                    ),
                },
            )

        def _request_is_local(self) -> bool:
            host_header = self.headers.get("Host", "")
            parsed_host = urlsplit(f"//{host_header}")
            if not parsed_host.hostname or not _is_valid_browser_host(parsed_host.hostname):
                return False

            origin = self.headers.get("Origin")
            if origin:
                parsed_origin = urlsplit(origin)
                if parsed_origin.scheme not in {"http", "https"}:
                    return False
                if not parsed_origin.hostname or not _is_valid_browser_host(parsed_origin.hostname):
                    return False
                if parsed_origin.hostname.casefold() != parsed_host.hostname.casefold():
                    return False
            return True

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _send_error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": message})

        def log_message(self, format_string: str, *args: Any) -> None:
            _LOG.info("%s - %s", self.address_string(), format_string % args)

    server = ThreadingHTTPServer(("127.0.0.1", port), ChatRequestHandler)
    server.daemon_threads = True
    return server


def serve_chat(
    pipeline: RAGPipeline,
    port: int = 8000,
    top_k: int = 4,
    wikipedia_chat: WikipediaChat | None = None,
) -> None:
    """Run the local browser chat interface until interrupted."""
    server = create_chat_server(pipeline, port, top_k, wikipedia_chat)
    _LOG.info("Local chat interface available at http://127.0.0.1:%d", server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _LOG.info("Stopping local chat interface.")
    finally:
        server.server_close()
