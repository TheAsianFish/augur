"""Turn rendered HTML into the plain prose a model reads, with the stdlib parser only.

Built for MediaWiki's parser output: tables (infoboxes, navboxes), reference
markers, hidden elements and maintenance boxes are dropped, paragraph and list
text is kept, and everything from the first back-matter heading ("References",
"See also", "External links" and the like) onwards is cut.
"""

from __future__ import annotations

from html.parser import HTMLParser

_SKIP_TAGS = frozenset(
    {"table", "style", "script", "noscript", "figure", "math", "audio", "video", "template"}
)
_SKIP_CLASSES = frozenset(
    {
        "reference",
        "references",
        "reflist",
        "mw-references-wrap",
        "mw-editsection",
        "navbox",
        "navbox-styles",
        "hatnote",
        "shortdescription",
        "noprint",
        "metadata",
        "ambox",
        "infobox",
        "sidebar",
        "thumb",
        "toc",
        "mw-empty-elt",
        "redirectMsg",
        "catlinks",
        "mw-cite-backlink",
        "sistersitebox",
        "rcat",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }
)
_BLOCK_TAGS = frozenset(
    {"p", "div", "li", "ul", "ol", "dl", "dd", "dt", "blockquote", "pre", "section", "br"}
)
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
STOP_HEADINGS = frozenset(
    {
        "references",
        "notes",
        "citations",
        "sources",
        "footnotes",
        "bibliography",
        "see also",
        "external links",
        "further reading",
        "notes and references",
    }
)


class _ProseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._heading: list[str] | None = None
        self.stopped = False

    def _should_skip(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in _SKIP_TAGS:
            return True
        for name, value in attrs:
            if value is None:
                continue
            if name == "class" and _SKIP_CLASSES.intersection(value.split()):
                return True
            if name == "style" and "display:none" in value.replace(" ", "").lower():
                return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.stopped:
            return
        if tag in _VOID_TAGS:
            if tag == "br" and not self._skip_depth:
                self._emit("\n")
            return
        if self._skip_depth:
            self._skip_depth += 1
            return
        if self._should_skip(tag, attrs):
            self._skip_depth = 1
            return
        if tag in _HEADINGS:
            self._heading = []
        elif tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self.stopped or tag in _VOID_TAGS:
            return
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in _HEADINGS and self._heading is not None:
            text = " ".join("".join(self._heading).split())
            self._heading = None
            if text.lower() in STOP_HEADINGS:
                self.stopped = True
                return
            if text:
                self.parts.append(f"\n{text}\n")
        elif tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self.stopped or self._skip_depth:
            return
        self._emit(data)

    def _emit(self, text: str) -> None:
        if self._heading is not None:
            self._heading.append(text)
        else:
            self.parts.append(text)


def html_to_text(html: str) -> str:
    """Plain prose from MediaWiki HTML: one line per block, whitespace collapsed."""
    parser = _ProseParser()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line)


def truncate_words(text: str, max_chars: int) -> str:
    """``text`` cut to at most ``max_chars`` characters at a word boundary, marked with "…"."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - 1]
    cut = max(head.rfind(" "), head.rfind("\n"))
    if cut > 0:
        head = head[:cut]
    return head.rstrip() + "…"
