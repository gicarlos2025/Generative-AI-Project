import html
import json
import re
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


LEARNING_LEVELS = ("basico", "intermediario", "avancado")

_ADVANCED_TERMS = (
    "crítica", "críticas", "debate", "debates", "limitação", "limitações",
    "teoria", "teorias", "matemática", "matemático", "pesquisa", "investigação",
    "avanços", "estado da arte", "controvérsia", "controvérsias",
)
_INTERMEDIATE_TERMS = (
    "história", "funcionamento", "mecanismo", "característica", "características",
    "estrutura", "aplicação", "aplicações", "tipo", "tipos", "classificação",
    "desenvolvimento", "método", "métodos",
)
_IGNORED_TAGS = {"figure", "nav", "script", "style", "sup", "table"}


@dataclass(frozen=True)
class WikipediaSection:
    title: str
    text: str
    learning_level: str
    section_index: int


@dataclass(frozen=True)
class WikipediaArticle:
    title: str
    language: str
    source_url: str
    retrieved_at: str
    sections: list[WikipediaSection]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["license"] = "CC BY-SA 4.0; attribute the original Wikipedia article"
        return result


class _ArticleHTMLParser(HTMLParser):
    """Extract the lead and section text from MediaWiki-rendered article HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[tuple[str, str]] = []
        self._title = "Introdução"
        self._body: list[str] = []
        self._heading: list[str] = []
        self._in_heading = False
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._ignored_depth:
            self._ignored_depth += 1
            return
        if tag in _IGNORED_TAGS:
            self._ignored_depth = 1
            return
        if tag in {"h2", "h3"}:
            self._finish_section()
            self._heading = []
            self._in_heading = True

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_depth:
            self._ignored_depth -= 1
            return
        if tag in {"h2", "h3"} and self._in_heading:
            title = " ".join(" ".join(self._heading).split())
            if title:
                self._title = title
            self._heading = []
            self._in_heading = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth or not data.strip():
            return
        if self._in_heading:
            self._heading.append(data.strip())
        else:
            self._body.append(data.strip())

    def finish(self) -> list[tuple[str, str]]:
        self._finish_section()
        return self.sections

    def _finish_section(self) -> None:
        text = " ".join(" ".join(self._body).split())
        if text:
            self.sections.append((self._title, text))
        self._body = []


class WikipediaExtractor:
    """Fetch and group Portuguese Wikipedia content using the official MediaWiki API."""

    def __init__(self, language: str = "pt", timeout: float = 15.0) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,11}", language):
            raise ValueError("language must be a valid Wikipedia language code.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        self.language = language
        self.timeout = timeout
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"

    def search(self, query: str) -> str:
        """Return the most relevant article title from the official MediaWiki API."""
        topics = self.search_related_topics(query, limit=1)
        return topics[0]["title"]

    def search_related_topics(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Search MediaWiki for relevant titles and return readable snippets."""
        if not query.strip():
            raise ValueError("Wikipedia search query must not be empty.")
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10.")

        payload = self._request(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
            }
        )
        query_result = payload.get("query")
        results = query_result.get("search") if isinstance(query_result, dict) else None
        if not isinstance(results, list) or not results:
            raise LookupError(f"Wikipedia did not find an article for {query!r}.")
        topics: list[dict[str, str]] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            title = result.get("title")
            snippet = result.get("snippet", "")
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(snippet, str):
                snippet = ""
            topics.append(
                {
                    "title": title,
                    "snippet": _strip_html(snippet),
                }
            )
        if not topics:
            raise ValueError("Wikipedia search returned no valid article titles.")
        return topics

    def extract(self, title: str) -> WikipediaArticle:
        """Retrieve an article, assigning its lead and sections to learning levels."""
        if not title.strip():
            raise ValueError("Wikipedia article title must not be empty.")

        payload = self._request(
            {
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "text",
                "disableeditsection": "1",
                "disabletoc": "1",
            }
        )
        parsed = payload.get("parse")
        if not isinstance(parsed, dict):
            raise ValueError(f"Wikipedia did not return article content for {title!r}.")
        html = parsed.get("text", {}).get("*")
        resolved_title = parsed.get("title")
        if not isinstance(html, str) or not isinstance(resolved_title, str):
            raise ValueError(f"Wikipedia returned an invalid article response for {title!r}.")

        parser = _ArticleHTMLParser()
        parser.feed(html)
        raw_sections = parser.finish()
        if not raw_sections:
            raise ValueError(f"Wikipedia article {resolved_title!r} contains no readable text.")

        article_sections = self._classify_sections(raw_sections)
        page_name = quote(resolved_title.replace(" ", "_"), safe="()!,._-")
        return WikipediaArticle(
            title=resolved_title,
            language=self.language,
            source_url=f"https://{self.language}.wikipedia.org/wiki/{page_name}",
            retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sections=article_sections,
        )

    def fetch_article_image(self, title: str) -> dict[str, str] | None:
        """Fetch a page thumbnail and its file-description metadata from MediaWiki."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Wikipedia article title must not be empty.")

        payload = self._request(
            {
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": "960",
                "titles": title,
            }
        )
        query_result = payload.get("query")
        pages = query_result.get("pages") if isinstance(query_result, dict) else None
        if not isinstance(pages, dict):
            return None
        page = next(
            (
                page_data
                for page_id, page_data in pages.items()
                if page_id != "-1" and isinstance(page_data, dict)
            ),
            None,
        )
        if page is None:
            return None
        page_image = page.get("pageimage")
        thumbnail = page.get("thumbnail")
        image_url = thumbnail.get("source") if isinstance(thumbnail, dict) else None
        if not isinstance(page_image, str) or not isinstance(image_url, str):
            return None
        if not self._is_upload_url(image_url):
            raise ValueError("Wikipedia returned an image URL outside upload.wikimedia.org.")

        image_payload = self._request(
            {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "titles": f"File:{page_image}",
            }
        )
        image_query = image_payload.get("query")
        image_pages = image_query.get("pages") if isinstance(image_query, dict) else None
        image_page = (
            next(
                (
                    page_data
                    for page_id, page_data in image_pages.items()
                    if page_id != "-1" and isinstance(page_data, dict)
                ),
                None,
            )
            if isinstance(image_pages, dict)
            else None
        )
        image_info = image_page.get("imageinfo") if image_page else None
        metadata = image_info[0] if isinstance(image_info, list) and image_info else {}
        if not isinstance(metadata, dict):
            metadata = {}

        description_url = metadata.get("descriptionurl")
        if not isinstance(description_url, str) or not self._is_file_page_url(description_url):
            description_url = (
                f"https://{self.language}.wikipedia.org/wiki/"
                f"{quote('File:' + page_image.replace(' ', '_'), safe='()!,._-:')}"
            )
        extmetadata = metadata.get("extmetadata")
        license_metadata = (
            extmetadata.get("LicenseShortName")
            if isinstance(extmetadata, dict)
            else None
        )
        license_name = (
            license_metadata.get("value")
            if isinstance(license_metadata, dict)
            else None
        )
        artist_metadata = extmetadata.get("Artist") if isinstance(extmetadata, dict) else None
        artist_name = artist_metadata.get("value") if isinstance(artist_metadata, dict) else None

        result = {
            "title": page_image,
            "url": image_url,
            "source_url": description_url,
        }
        if isinstance(license_name, str) and license_name.strip():
            result["license"] = _strip_html(license_name)
        if isinstance(artist_name, str) and artist_name.strip():
            result["artist"] = _strip_html(artist_name)
        return result

    def save_json(self, article: WikipediaArticle, output_dir: str | Path = "data/prompts") -> Path:
        """Save extracted content and attribution metadata as UTF-8 JSON."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^\w.-]+", "_", article.title.casefold(), flags=re.UNICODE).strip("_.")
        if not slug:
            slug = "wikipedia_article"
        output_path = directory / f"{slug}.json"
        output_path.write_text(
            json.dumps(article.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

    def _request(self, parameters: dict[str, str]) -> dict[str, Any]:
        query = urlencode(parameters)
        request = Request(
            f"{self.api_url}?{query}",
            headers={
                "User-Agent": (
                    "GenerativeAIProjectMVP/1.0 "
                    "(https://github.com/gicarlos2025/Generative-AI-Project)"
                )
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"Wikipedia API returned HTTP {error.code}.") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach the Wikipedia API: {error.reason}") from error
        except (TimeoutError, OSError) as error:
            raise RuntimeError(f"Wikipedia API request failed: {error}") from error

        if not isinstance(payload, dict):
            raise ValueError("Wikipedia API returned an invalid JSON response.")
        api_error = payload.get("error")
        if api_error:
            message = api_error.get("info", "Unknown MediaWiki API error")
            raise RuntimeError(f"Wikipedia API error: {message}")
        return payload

    @staticmethod
    def _is_upload_url(value: str) -> bool:
        parsed = urlparse(value)
        return (
            parsed.scheme == "https"
            and parsed.hostname == "upload.wikimedia.org"
            and parsed.username is None
            and parsed.password is None
        )

    def _is_file_page_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return (
            parsed.scheme == "https"
            and parsed.hostname in {"commons.wikimedia.org", f"{self.language}.wikipedia.org"}
            and parsed.path.startswith("/wiki/File:")
            and parsed.username is None
            and parsed.password is None
        )

    @staticmethod
    def _classify_sections(raw_sections: list[tuple[str, str]]) -> list[WikipediaSection]:
        lead_title, lead_text = raw_sections[0]
        sections = [
            WikipediaSection(
                title=lead_title,
                text=lead_text,
                learning_level="basico",
                section_index=0,
            )
        ]
        detailed = raw_sections[1:]
        section_count = len(detailed)
        for index, (title, text) in enumerate(detailed, start=1):
            normalized_title = title.casefold()
            if any(term in normalized_title for term in _ADVANCED_TERMS):
                level = "avancado"
            elif any(term in normalized_title for term in _INTERMEDIATE_TERMS):
                level = "intermediario"
            elif section_count and index > section_count * 2 / 3:
                level = "avancado"
            else:
                level = "intermediario"
            sections.append(
                WikipediaSection(
                    title=title,
                    text=text,
                    learning_level=level,
                    section_index=index,
                )
            )
        return sections


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]*>", " ", value)
    text = re.sub(r"\s+([.,!?;:])", r"\1", html.unescape(text))
    return " ".join(text.split())
