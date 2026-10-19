#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal pure-Python PDF text generator (no third-party libraries)."""

import unicodedata


def strip_diacritics(t):
    """Remove Vietnamese/Unicode diacritics and keep ASCII-printable newlines."""
    t = unicodedata.normalize("NFD", t or "")
    s = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return "".join(c for c in s if c == "\n" or 32 <= ord(c) < 127)


def wrap(text, width=90):
    lines = []
    for para in text.split("\n"):
        words = para.split()
        cur = ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > width:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
    return lines


def _esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(text, title="Tài liệu"):
    """Return PDF bytes for a simple one-page text document."""
    title = strip_diacritics(title)
    text = strip_diacritics(text)
    body_lines = wrap(text)[:48]  # keep to one page (~48 lines)

    ops = ["BT", "/F1 16 Tf", "50 760 Td (%s) Tj" % _esc(title),
           "/F1 12 Tf", "50 740 Td () Tj"]
    for ln in body_lines:
        ops.append("0 -15 Td (%s) Tj" % _esc(ln))
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1", "replace") + b"\n"

    def obj(n):
        return b"%d 0 obj\n" % n

    catalog = obj(1) + b"<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pages = obj(2) + b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    page = obj(3) + (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                     b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
    contents = obj(4) + (b"<< /Length %d >>\nstream\n" % len(stream)) + \
        stream + b"endstream\nendobj\n"
    font = obj(5) + b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {0: 0}
    pieces = {1: catalog, 2: pages, 3: page, 4: contents, 5: font}
    for i in range(1, 6):
        offsets[i] = len(out)
        out += pieces[i]

    xref_pos = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    for i in range(1, 6):
        out += b"%010d 00000 n \n" % offsets[i]
    size = 6
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_pos)
    return bytes(out)
