import json
import os
import re
import tempfile
import threading
import unicodedata
from pathlib import Path
from typing import Any

from src.utils.logger import logger
from src.utils.wiki_extractor import LEARNING_LEVELS


_WORD_PATTERN = re.compile(r"[^\W_]{3,}", flags=re.UNICODE)


class PersonalVocabularyBuilder:
    """Persist concepts encountered in research as a local personal lexicon."""

    def __init__(self, storage_path: str | Path = "data/personal_lexicon.json") -> None:
        self.storage_path = Path(storage_path)
        self.lexicon: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load_lexicon()

    def _load_lexicon(self) -> None:
        if not self.storage_path.exists():
            logger.info("No personal lexicon found at %s; starting an empty lexicon.", self.storage_path)
            return

        try:
            with self.storage_path.open("r", encoding="utf-8") as lexicon_file:
                loaded = json.load(lexicon_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not load personal lexicon from {self.storage_path}.") from error
        if not isinstance(loaded, dict):
            raise ValueError(f"Personal lexicon must be a JSON object: {self.storage_path}")

        normalized_lexicon: dict[str, dict[str, Any]] = {}
        for key, entry in loaded.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(entry, dict):
                raise ValueError(f"Invalid entry in personal lexicon: {self.storage_path}")
            self._validate_entry(key, entry)
            entry.setdefault("sources", [])
            normalized_key = self._normalize_term(key)
            if normalized_key in normalized_lexicon:
                raise ValueError(f"Duplicate normalized term in personal lexicon: {key!r}.")
            normalized_lexicon[normalized_key] = entry
        self.lexicon = normalized_lexicon
        logger.info("Loaded %d terms from the personal lexicon.", len(self.lexicon))

    @staticmethod
    def _normalize_term(term: str) -> str:
        normalized = unicodedata.normalize("NFKD", term.casefold())
        without_accents = "".join(
            character for character in normalized if not unicodedata.combining(character)
        )
        return " ".join(without_accents.split())

    @staticmethod
    def _validate_entry(key: str, entry: dict[str, Any]) -> None:
        if (
            not isinstance(entry.get("term"), str)
            or not entry["term"].strip()
            or not isinstance(entry.get("meanings"), list)
            or any(not isinstance(value, str) or not value.strip() for value in entry["meanings"])
            or not isinstance(entry.get("contexts"), list)
            or any(not isinstance(value, str) or not value.strip() for value in entry["contexts"])
            or entry.get("maturity_level") not in LEARNING_LEVELS
            or not isinstance(entry.get("exposure_count"), int)
            or isinstance(entry.get("exposure_count"), bool)
            or entry["exposure_count"] < 1
        ):
            raise ValueError(f"Invalid vocabulary record for {key!r}.")
        sources = entry.get("sources", [])
        if not isinstance(sources, list) or any(
            not isinstance(source, str) or not source.strip() for source in sources
        ):
            raise ValueError(f"Invalid sources for vocabulary record {key!r}.")

    def _save_lexicon(self, lexicon: dict[str, dict[str, Any]]) -> None:
        parent = self.storage_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent,
                prefix=f".{self.storage_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as lexicon_file:
                temporary_path = lexicon_file.name
                json.dump(lexicon, lexicon_file, ensure_ascii=False, indent=2)
                lexicon_file.write("\n")
            os.replace(temporary_path, self.storage_path)
        except OSError as error:
            raise RuntimeError(f"Could not save personal lexicon to {self.storage_path}.") from error
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def acquire_sign(
        self,
        term: str,
        meaning: str,
        context: str,
        level: str = "basico",
        source: str | None = None,
    ) -> None:
        """Add a researched concept or consolidate a new exposure to it."""
        for field_name, value in (
            ("term", term),
            ("meaning", meaning),
            ("context", context),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
        if level not in LEARNING_LEVELS:
            raise ValueError(f"level must be one of: {', '.join(LEARNING_LEVELS)}.")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError("source must be a non-empty string when provided.")

        normalized_term = self._normalize_term(term)
        with self._lock:
            updated = {
                key: {
                    **entry,
                    "meanings": list(entry["meanings"]),
                    "contexts": list(entry["contexts"]),
                    "sources": list(entry.get("sources", [])),
                }
                for key, entry in self.lexicon.items()
            }
            entry = updated.get(normalized_term)
            if entry is None:
                entry = {
                    "term": term.strip(),
                    "meanings": [],
                    "contexts": [],
                    "maturity_level": level,
                    "exposure_count": 0,
                    "sources": [],
                }
                updated[normalized_term] = entry

            entry["exposure_count"] += 1
            if meaning.strip() not in entry["meanings"]:
                entry["meanings"].append(meaning.strip())
            if context.strip() not in entry["contexts"]:
                entry["contexts"].append(context.strip())
            if source is not None and source.strip() not in entry["sources"]:
                entry["sources"].append(source.strip())
            if LEARNING_LEVELS.index(level) > LEARNING_LEVELS.index(entry["maturity_level"]):
                entry["maturity_level"] = level

            self._save_lexicon(updated)
            self.lexicon = updated
        logger.info(
            "Recorded vocabulary exposure for %r at %s level (exposure %d).",
            entry["term"],
            entry["maturity_level"],
            entry["exposure_count"],
        )

    def consult_lexicon(self, term: str) -> dict[str, Any] | None:
        """Return a copy of the locally stored record for a term, if known."""
        if not isinstance(term, str) or not term.strip():
            raise ValueError("term must not be empty.")
        with self._lock:
            entry = self.lexicon.get(self._normalize_term(term))
            if entry is None:
                return None
            return {
                **entry,
                "meanings": list(entry["meanings"]),
                "contexts": list(entry["contexts"]),
                "sources": list(entry.get("sources", [])),
            }

    def find_relevant(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find lexicon records whose terms occur in the current question."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty.")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20.")

        query_tokens = set(_WORD_PATTERN.findall(self._normalize_term(query)))
        normalized_query = self._normalize_term(query)
        ranked: list[tuple[float, dict[str, Any]]] = []
        with self._lock:
            entries = list(self.lexicon.items())
        for normalized_term, entry in entries:
            term_tokens = set(_WORD_PATTERN.findall(normalized_term))
            if not term_tokens:
                continue
            overlap = len(term_tokens & query_tokens)
            if overlap == 0:
                continue
            phrase_match = normalized_term in normalized_query
            score = (2.0 if phrase_match else 0.0) + overlap / len(term_tokens)
            result = {
                **entry,
                "meanings": list(entry["meanings"]),
                "contexts": list(entry["contexts"]),
                "sources": list(entry.get("sources", [])),
            }
            ranked.append((score, result))
        ranked.sort(
            key=lambda item: (
                -item[0],
                -LEARNING_LEVELS.index(item[1]["maturity_level"]),
                -item[1]["exposure_count"],
                item[1]["term"].casefold(),
            )
        )
        return [entry for _, entry in ranked[:limit]]

    def format_relevant_context(self, query: str, limit: int = 5) -> str:
        """Format relevant memory as supplemental, explicitly sourced prompt context."""
        entries = self.find_relevant(query, limit=limit)
        if not entries:
            return ""

        formatted_entries: list[str] = []
        for entry in entries:
            meanings = "\n".join(f"- Meaning: {meaning}" for meaning in entry["meanings"])
            contexts = "\n".join(
                f"- Encountered in: {context}" for context in entry["contexts"][:2]
            )
            sources = "\n".join(f"- Source: {source}" for source in entry["sources"])
            formatted_entries.append(
                f"Term: {entry['term']} (maturity: {entry['maturity_level']}, "
                f"exposures: {entry['exposure_count']})\n"
                f"{meanings}\n{contexts}\n{sources}"
            )
        return "\n\n".join(formatted_entries)
