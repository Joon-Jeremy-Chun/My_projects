"""Read-only structure, documentation-link, and source checks for this archive."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import ast
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".spyproject", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "venv"}


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"a", "img"}:
            value = attrs.get("href" if tag == "a" else "src")
            if value:
                self.targets.append(value)
        if "id" in attrs:
            self.anchors.add(attrs["id"])
        if tag == "a" and "name" in attrs:
            self.anchors.add(attrs["name"])


def inspect_markdown(path):
    text = path.read_text(encoding="utf-8-sig")
    issues = []
    parser = References()
    pieces = []
    inside = False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif not inside:
            pieces.append(line)
    if inside:
        issues.append("unclosed code fence")
    if "\ufffd" in text:
        issues.append("replacement character in UTF-8 text")
    body = "\n".join(pieces)
    parser.feed(body)
    targets = parser.targets + [m.group(1).strip("<>") for m in re.finditer(r'!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\s*\)', body)]
    anchors = set(parser.anchors)
    duplicates = Counter()
    for line in pieces:
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        title = re.sub(r"<[^>]*>", "", match.group(1)).lower()
        title = re.sub(r"[^\w\-\s]", "", title)
        slug = re.sub(r"\s", "-", title)
        count = duplicates[slug]
        duplicates[slug] += 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return targets, anchors, issues


def main():
    errors = []
    paths = [p for p in ROOT.rglob("*") if p.is_file() and not SKIP.intersection(p.relative_to(ROOT).parts)]
    python_files = [p for p in paths if p.suffix == ".py"]
    notebooks = [p for p in paths if p.suffix == ".ipynb"]
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path.relative_to(ROOT)))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(notebook.get("cells"), list):
                raise ValueError("missing cell list")
        except (ValueError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    documents = {p: inspect_markdown(p) for p in paths if p.suffix.lower() == ".md"}
    local_refs = 0
    for path,(targets,_,issues) in documents.items():
        errors.extend(f"{path.relative_to(ROOT)}: {issue}" for issue in issues)
        for target in targets:
            split = urlsplit(target)
            if split.scheme or split.netloc:
                continue
            local_refs += 1
            destination = (path.parent/unquote(split.path)).resolve() if split.path else path
            if not destination.is_relative_to(ROOT):
                errors.append(f"{path.relative_to(ROOT)}: reference leaves repository: {target}")
            elif not destination.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing target: {target}")
            elif split.fragment and destination in documents and unquote(split.fragment) not in documents[destination][1]:
                errors.append(f"{path.relative_to(ROOT)}: unknown anchor: {target}")
    svgs = [p for p in paths if p.suffix.lower() == ".svg"]
    for path in svgs:
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    report = {"python_parsed":len(python_files), "notebook_json_parsed":len(notebooks),
              "markdown_files":len(documents), "local_references":local_refs,
              "svg_parsed":len(svgs), "errors":errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
