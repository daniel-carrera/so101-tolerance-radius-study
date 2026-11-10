"""Shared by the figure tests: what a PDF says about itself, read from its bytes with the standard library.

The figure tests check the files the figure code writes and not the code's own account of them: the page's size from
its `MediaBox`, the size of every piece of text from the page's drawing instructions, and what the bytes hold that an
anonymous paper must not.
"""

from __future__ import annotations

import getpass
import importlib.util
import math
import re
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MM_PER_PT = 25.4 / 72.0

_STREAM = re.compile(rb"<<(?P<dictionary>(?:[^<>]|<<[^<>]*>>)*)>>\s*stream\r?\n(?P<body>.*?)\r?\nendstream", re.S)
_MEDIA_BOX = re.compile(rb"/MediaBox\s*\[\s*([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*\]")
_NUMBER = re.compile(rb"[-+]?(?:\d+\.?\d*|\.\d+)")


def page_size_mm(data: bytes) -> tuple[float, float]:
    """The width and height of the page of a one-page PDF, in millimetres."""
    boxes = _MEDIA_BOX.findall(data)
    if len(boxes) != 1:
        raise AssertionError(f"expected one MediaBox, found {len(boxes)}")
    x0, y0, x1, y1 = (float(value) for value in boxes[0])
    return (x1 - x0) * MM_PER_PT, (y1 - y0) * MM_PER_PT


def streams(data: bytes) -> list[bytes]:
    """The contents of every stream of the file, decompressed where the file says it is compressed."""
    found = []
    for match in _STREAM.finditer(data):
        body = match.group("body")
        found.append(zlib.decompress(body) if b"/FlateDecode" in match.group("dictionary") else body)
    return found


def _tokens(content: bytes) -> list[bytes]:
    """A page's drawing instructions as operands and operators. A string of text is one token."""
    tokens, at, end = [], 0, len(content)
    while at < end:
        byte = content[at : at + 1]
        if byte.isspace():
            at += 1
        elif byte == b"(":  # a string: ends at the bracket that closes it; a backslash protects the next byte
            depth, start = 1, at
            at += 1
            while at < end and depth:
                if content[at : at + 1] == b"\\":
                    at += 1
                elif content[at : at + 1] == b"(":
                    depth += 1
                elif content[at : at + 1] == b")":
                    depth -= 1
                at += 1
            tokens.append(content[start:at])
        elif byte in b"[]":
            tokens.append(byte)
            at += 1
        elif byte == b"<" and content[at : at + 2] != b"<<":
            start = at
            at = content.index(b">", at) + 1
            tokens.append(content[start:at])
        else:
            start = at
            while at < end and not content[at : at + 1].isspace() and content[at : at + 1] not in b"()[]":
                at += 1
            tokens.append(content[start:at])
    return tokens


def text_sizes_pt(data: bytes) -> list[float]:
    """The size in points at which every run of text of the file is drawn on the page.

    A size is what the page's instructions select (`Tf`) times the scale of the transformations in force at that
    point (`cm`, and `Tm` inside a text object), so a small font enlarged, or a large one shrunk, is seen for what it
    prints as.
    """
    sizes: list[float] = []
    for content in streams(data):
        if b" Tf" not in content and b"\nTf" not in content:
            continue
        scale, stack, text_scale, font_size = 1.0, [], 1.0, None
        operands: list[bytes] = []
        for token in _tokens(content):
            if _NUMBER.fullmatch(token) or token[:1] in (b"/", b"(", b"<", b"[", b"]"):
                operands.append(token)
                continue
            if token == b"q":
                stack.append(scale)
            elif token == b"Q" and stack:
                scale = stack.pop()
            elif token == b"cm":
                a, b, c, d = (float(value) for value in operands[-6:-2])
                scale *= math.sqrt(abs(a * d - b * c))
            elif token == b"BT":
                text_scale = 1.0
            elif token == b"Tm":
                a, b, c, d = (float(value) for value in operands[-6:-2])
                text_scale = math.sqrt(abs(a * d - b * c))
            elif token == b"Tf":
                font_size = float(operands[-1])
            elif token in (b"Tj", b"TJ", b"'", b'"'):
                if font_size is None:
                    raise AssertionError("text is drawn before a font is chosen")
                sizes.append(font_size * text_scale * scale)
            operands = []
    return sizes


def font_kinds(data: bytes) -> set[bytes]:
    """The kinds of font the file declares (`/Subtype` of every font object)."""
    return set(re.findall(rb"/Type\s*/Font\b[^>]*?/Subtype\s*/(\w+)", data)) | set(
        re.findall(rb"/Subtype\s*/(\w+)[^>]*?/Type\s*/Font\b", data)
    )


def _check_paper():
    """The paper's own checker, loaded from its file: its list of forbidden strings is read the way it reads it."""
    name = "check_paper_for_figure_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "paper" / "check_paper.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def forbidden_hits(data: bytes) -> list[str]:
    """Every entry of `paper/forbidden.txt` that the bytes of a file match, found as `check_paper.py` looks for them:
    line by line, in the bytes read as text.
    """
    entries = _check_paper().read_forbidden(REPO / "paper" / "forbidden.txt")
    lines = data.decode("latin-1").split("\n")
    return [entry.shown for entry in entries if any(entry.pattern.search(line) for line in lines)]


def names_this_machine(data: bytes) -> list[str]:
    """Which of this machine's user name and home directory appear in the bytes. Both are read here, when asked, and
    are named in the answer without being repeated.

    A user name of fewer than four letters is not looked for: it would be found by chance in any compressed file.
    """
    found = []
    user = getpass.getuser()
    if len(user) >= 4 and user.encode() in data:
        found.append("the user's name")
    if str(Path.home()).encode() in data:
        found.append("the home directory")
    return found
