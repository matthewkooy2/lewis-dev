"""Tokenizer for JavaScript, TypeScript, JSX, and TSX sources.

A real lexer (not regex-over-text): it understands comments, string and
template literals with nested ``${}`` expressions, regex-vs-division
disambiguation, multi-character operators, and JSX elements with embedded
expression islands. The token stream is what the structural analyzer in
``js_analysis`` consumes; neither module executes any analyzed code.

JSX handling: in JSX-enabled files a ``<`` in expression position opens an
element scan that emits ``jsx_open``/``jsx_attr``/``jsx_close`` tokens and
re-lexes ``{...}`` islands as ordinary code between ``jsx_expr_open`` and
``jsx_expr_close`` markers. A ``<`` that does not scan as a plausible tag is
backtracked and emitted as a plain punctuator, so TS generics in ``.tsx``
files degrade safely.
"""

from __future__ import annotations

from dataclasses import dataclass


MAX_TOKENS = 300_000
MAX_STRING_VALUE_CHARS = 200

KEYWORDS = frozenset(
    {
        "abstract", "any", "as", "async", "await", "boolean", "break", "case",
        "catch", "class", "const", "continue", "debugger", "declare", "default",
        "delete", "do", "else", "enum", "export", "extends", "false", "finally",
        "for", "from", "function", "get", "if", "implements", "import", "in",
        "instanceof", "interface", "is", "keyof", "let", "namespace", "new",
        "null", "number", "of", "override", "private", "protected", "public",
        "readonly", "return", "satisfies", "set", "static", "string", "super",
        "switch", "this", "throw", "true", "try", "type", "typeof", "undefined",
        "var", "void", "while", "with", "yield",
    }
)

# Longest-match punctuator table.
_PUNCTUATORS = (
    ">>>=",
    "...", "===", "!==", "**=", "<<=", ">>=", ">>>", "&&=", "||=", "??=",
    "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--", "+=",
    "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "**",
    "{", "}", "(", ")", "[", "]", ";", ",", "<", ">", "+", "-", "*", "/",
    "%", "&", "|", "^", "!", "~", "?", ":", "=", ".", "@", "#",
)

# Previous significant tokens after which a `/` is a regex and a `<` may open
# JSX (expression position).
_EXPRESSION_POSITION_PUNCTS = frozenset(
    {
        "(", "[", "{", ",", ";", ":", "=>", "=", "+", "-", "*", "/", "%",
        "!", "?", "&&", "||", "??", "==", "===", "!=", "!==", "<", ">",
        "<=", ">=", "+=", "-=", "*=", "/=", "%=", "&", "|", "^", "~",
        "<<", ">>", ">>>", "...", "jsx_expr_open", "template_expr_open",
    }
)
_EXPRESSION_POSITION_KEYWORDS = frozenset(
    {
        "return", "typeof", "case", "delete", "void", "throw", "do", "else",
        "yield", "await", "in", "of", "new", "instanceof", "default",
    }
)

_ID_START = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$"
)
_ID_CONTINUE = _ID_START | frozenset("0123456789")


@dataclass(frozen=True)
class Token:
    type: str  # id | keyword | punct | string | template | regex | number |
    #            jsx_open | jsx_attr | jsx_close | jsx_expr_open |
    #            jsx_expr_close | template_expr_open | template_expr_close
    value: str
    line: int


@dataclass
class LexResult:
    tokens: list[Token]
    ok: bool = True
    error: str = ""


class _Frame:
    """Lexing mode frame: what the lexer returns to when a region closes."""

    NORMAL = "normal"
    TEMPLATE = "template"
    JSX_TAG = "jsx_tag"
    JSX_CHILDREN = "jsx_children"

    def __init__(self, kind: str, detail: str = "", brace_depth: int = 0):
        self.kind = kind
        self.detail = detail
        self.brace_depth = brace_depth


def lex(text: str, *, jsx_enabled: bool) -> LexResult:
    return _Lexer(text, jsx_enabled=jsx_enabled).run()


class _Lexer:
    def __init__(self, text: str, *, jsx_enabled: bool):
        self.text = text.lstrip("﻿")
        self.length = len(text)
        self.pos = 0
        self.line = 1
        self.jsx_enabled = jsx_enabled
        self.tokens: list[Token] = []
        self.frames: list[_Frame] = [_Frame(_Frame.NORMAL)]
        self.error = ""

    # ---- driving loop ----------------------------------------------------

    def run(self) -> LexResult:
        while self.pos < self.length:
            if len(self.tokens) >= MAX_TOKENS:
                return LexResult(self.tokens, ok=False, error="token limit reached")
            frame = self.frames[-1]
            if frame.kind == _Frame.TEMPLATE:
                self._lex_template_chunk()
            elif frame.kind == _Frame.JSX_TAG:
                self._lex_jsx_tag()
            elif frame.kind == _Frame.JSX_CHILDREN:
                self._lex_jsx_children()
            else:
                self._lex_normal()
        ok = len(self.frames) == 1 and not self.error
        if not ok and not self.error:
            self.error = f"unclosed {self.frames[-1].kind} region"
        return LexResult(self.tokens, ok=ok, error=self.error)

    # ---- normal code -----------------------------------------------------

    def _lex_normal(self) -> None:
        if not self._skip_trivia():
            return
        char = self.text[self.pos]
        frame = self.frames[-1]

        if char == "`":
            self.pos += 1
            self.frames.append(_Frame(_Frame.TEMPLATE))
            return
        if char in "'\"":
            self._lex_string(char)
            return
        if char.isdigit() or (
            char == "." and self._peek(1).isdigit()
        ):
            self._lex_number()
            return
        if char in _ID_START:
            self._lex_identifier()
            return
        if char == "/" and self._regex_allowed():
            if self._try_lex_regex():
                return
        if char == "<" and self.jsx_enabled and self._jsx_allowed():
            if self._try_enter_jsx():
                return

        if char == "{":
            frame.brace_depth += 1
            self._emit("punct", "{")
            self.pos += 1
            return
        if char == "}":
            if frame.brace_depth == 0 and frame.detail in {"template", "jsx"}:
                self.frames.pop()
                if frame.detail == "template":
                    self._emit("template_expr_close", "}")
                else:
                    self._emit("jsx_expr_close", "}")
                self.pos += 1
                return
            frame.brace_depth = max(0, frame.brace_depth - 1)
            self._emit("punct", "}")
            self.pos += 1
            return

        for punct in _PUNCTUATORS:
            if self.text.startswith(punct, self.pos):
                self._emit("punct", punct)
                self.pos += len(punct)
                return
        # Unknown character (unicode identifier, stray byte): skip it.
        self.pos += 1

    def _skip_trivia(self) -> bool:
        """Skip whitespace/comments; False when input ended or state changed."""
        while self.pos < self.length:
            char = self.text[self.pos]
            if char == "\n":
                self.line += 1
                self.pos += 1
            elif char in " \t\r\f\v":
                self.pos += 1
            elif char == "/" and self._peek(1) == "/":
                index = self.text.find("\n", self.pos)
                self.pos = self.length if index == -1 else index
            elif char == "/" and self._peek(1) == "*":
                end = self.text.find("*/", self.pos + 2)
                if end == -1:
                    self.pos = self.length
                else:
                    self.line += self.text.count("\n", self.pos, end)
                    self.pos = end + 2
            else:
                return True
        return False

    def _lex_string(self, quote: str) -> None:
        start_line = self.line
        self.pos += 1
        chunks: list[str] = []
        while self.pos < self.length:
            char = self.text[self.pos]
            if char == "\\":
                if self._peek(1) == "\n":
                    self.line += 1
                self.pos += 2
                continue
            if char == quote:
                self.pos += 1
                break
            if char == "\n":
                # Unterminated string; recover at end of line.
                break
            chunks.append(char)
            self.pos += 1
        self._emit_at("string", "".join(chunks)[:MAX_STRING_VALUE_CHARS], start_line)

    def _lex_number(self) -> None:
        start = self.pos
        self.pos += 1
        while self.pos < self.length:
            char = self.text[self.pos]
            if char in "0123456789abcdefoxABCDEFOX_n.":
                self.pos += 1
            elif char in "+-" and self.text[self.pos - 1] in "eE":
                self.pos += 1
            else:
                break
        self._emit("number", self.text[start : self.pos])

    def _lex_identifier(self) -> None:
        start = self.pos
        while self.pos < self.length and self.text[self.pos] in _ID_CONTINUE:
            self.pos += 1
        value = self.text[start : self.pos]
        self._emit("keyword" if value in KEYWORDS else "id", value)

    # ---- regex -----------------------------------------------------------

    def _regex_allowed(self) -> bool:
        previous = self._previous_significant()
        if previous is None:
            return True
        if previous.type == "punct":
            return previous.value in _EXPRESSION_POSITION_PUNCTS
        if previous.type in {"jsx_expr_open", "template_expr_open"}:
            return True
        if previous.type == "keyword":
            return previous.value in _EXPRESSION_POSITION_KEYWORDS
        return False

    def _try_lex_regex(self) -> bool:
        start = self.pos
        start_line = self.line
        self.pos += 1
        in_class = False
        while self.pos < self.length:
            char = self.text[self.pos]
            if char == "\\":
                self.pos += 2
                continue
            if char == "\n":
                # Not a regex after all; treat the slash as division.
                self.pos = start
                self.line = start_line
                self._emit("punct", "/")
                self.pos += 1
                return True
            if char == "[":
                in_class = True
            elif char == "]":
                in_class = False
            elif char == "/" and not in_class:
                self.pos += 1
                while self.pos < self.length and self.text[self.pos].isalpha():
                    self.pos += 1
                self._emit_at("regex", self.text[start : self.pos][:80], start_line)
                return True
            self.pos += 1
        self.pos = start
        self.line = start_line
        self._emit("punct", "/")
        self.pos += 1
        return True

    # ---- JSX -------------------------------------------------------------

    def _jsx_allowed(self) -> bool:
        previous = self._previous_significant()
        if previous is None:
            return True
        if previous.type == "punct":
            return previous.value in _EXPRESSION_POSITION_PUNCTS
        if previous.type in {"jsx_expr_open", "template_expr_open"}:
            return True
        if previous.type == "keyword":
            return previous.value in _EXPRESSION_POSITION_KEYWORDS
        return False

    def _try_enter_jsx(self) -> bool:
        """Attempt to open a JSX element at ``<``; backtrack when implausible."""
        save_pos, save_line = self.pos, self.line
        self.pos += 1
        name = self._scan_jsx_name()
        if name is None:
            self.pos, self.line = save_pos, save_line
            self._emit("punct", "<")
            self.pos += 1
            return True
        self._skip_jsx_space()
        if self.pos >= self.length:
            self.pos, self.line = save_pos, save_line
            self._emit("punct", "<")
            self.pos += 1
            return True
        char = self.text[self.pos]
        if name == "" and char != ">":
            self.pos, self.line = save_pos, save_line
            self._emit("punct", "<")
            self.pos += 1
            return True
        if name != "" and not (
            char in _ID_START or char in ">/{"
        ):
            self.pos, self.line = save_pos, save_line
            self._emit("punct", "<")
            self.pos += 1
            return True
        self._emit("jsx_open", name)
        self.frames.append(_Frame(_Frame.JSX_TAG, detail=name))
        return True

    def _scan_jsx_name(self) -> str | None:
        """Tag name at pos: identifier with ., :, - allowed; '' for fragment."""
        if self.pos < self.length and self.text[self.pos] == ">":
            return ""
        if self.pos >= self.length or self.text[self.pos] not in _ID_START:
            return None
        start = self.pos
        while self.pos < self.length and (
            self.text[self.pos] in _ID_CONTINUE or self.text[self.pos] in ".:-"
        ):
            self.pos += 1
        return self.text[start : self.pos]

    def _skip_jsx_space(self) -> None:
        while self.pos < self.length and self.text[self.pos] in " \t\r\n\f\v":
            if self.text[self.pos] == "\n":
                self.line += 1
            self.pos += 1

    def _lex_jsx_tag(self) -> None:
        self._skip_jsx_space()
        if self.pos >= self.length:
            self.error = "unclosed JSX tag"
            self.frames.pop()
            return
        frame = self.frames[-1]
        char = self.text[self.pos]
        if char == "/" and self._peek(1) == ">":
            self._emit("jsx_close", frame.detail)
            self.pos += 2
            self.frames.pop()
            return
        if char == ">":
            self.pos += 1
            self.frames.pop()
            self.frames.append(_Frame(_Frame.JSX_CHILDREN, detail=frame.detail))
            return
        if char == "{":
            self._emit("jsx_expr_open", "{")
            self.pos += 1
            self.frames.append(_Frame(_Frame.NORMAL, detail="jsx"))
            return
        if char in _ID_START:
            start = self.pos
            while self.pos < self.length and (
                self.text[self.pos] in _ID_CONTINUE or self.text[self.pos] in "-:"
            ):
                self.pos += 1
            self._emit("jsx_attr", self.text[start : self.pos])
            self._skip_jsx_space()
            if self.pos < self.length and self.text[self.pos] == "=":
                self.pos += 1
                self._skip_jsx_space()
                if self.pos < self.length:
                    value_char = self.text[self.pos]
                    if value_char in "'\"":
                        self._lex_string(value_char)
                    elif value_char == "{":
                        self._emit("jsx_expr_open", "{")
                        self.pos += 1
                        self.frames.append(_Frame(_Frame.NORMAL, detail="jsx"))
            return
        # Unexpected character inside a tag: abandon the element honestly.
        self.error = f"unexpected character {char!r} in JSX tag"
        self.frames.pop()
        self.pos += 1

    def _lex_jsx_children(self) -> None:
        frame = self.frames[-1]
        while self.pos < self.length:
            char = self.text[self.pos]
            if char == "\n":
                self.line += 1
                self.pos += 1
                continue
            if char == "{":
                self._emit("jsx_expr_open", "{")
                self.pos += 1
                self.frames.append(_Frame(_Frame.NORMAL, detail="jsx"))
                return
            if char == "<":
                if self._peek(1) == "/":
                    self.pos += 2
                    self._scan_jsx_name()
                    self._skip_jsx_space()
                    if self.pos < self.length and self.text[self.pos] == ">":
                        self.pos += 1
                    self._emit("jsx_close", frame.detail)
                    self.frames.pop()
                    return
                save_pos, save_line = self.pos, self.line
                self.pos += 1
                name = self._scan_jsx_name()
                if name is None:
                    # Literal '<' in text; consume it.
                    self.pos = save_pos + 1
                    self.line = save_line
                    continue
                self._emit("jsx_open", name)
                self.frames.append(_Frame(_Frame.JSX_TAG, detail=name))
                return
            self.pos += 1
        self.error = "unclosed JSX children"
        self.frames.pop()

    # ---- template literals -------------------------------------------------

    def _lex_template_chunk(self) -> None:
        start_line = self.line
        chunks: list[str] = []
        while self.pos < self.length:
            char = self.text[self.pos]
            if char == "\\":
                if self._peek(1) == "\n":
                    self.line += 1
                self.pos += 2
                continue
            if char == "\n":
                self.line += 1
                chunks.append(char)
                self.pos += 1
                continue
            if char == "`":
                self.pos += 1
                self.frames.pop()
                self._emit_at(
                    "template", "".join(chunks)[:MAX_STRING_VALUE_CHARS], start_line
                )
                return
            if char == "$" and self._peek(1) == "{":
                self._emit_at(
                    "template", "".join(chunks)[:MAX_STRING_VALUE_CHARS], start_line
                )
                self._emit("template_expr_open", "${")
                self.pos += 2
                self.frames.append(_Frame(_Frame.NORMAL, detail="template"))
                return
            chunks.append(char)
            self.pos += 1
        self.error = "unclosed template literal"
        self.frames.pop()

    # ---- utilities ---------------------------------------------------------

    def _previous_significant(self) -> Token | None:
        for token in reversed(self.tokens):
            if token.type in {"jsx_open", "jsx_attr", "jsx_close"}:
                return None
            return token
        return None

    def _peek(self, offset: int) -> str:
        index = self.pos + offset
        return self.text[index] if index < self.length else ""

    def _emit(self, token_type: str, value: str) -> None:
        self.tokens.append(Token(token_type, value, self.line))

    def _emit_at(self, token_type: str, value: str, line: int) -> None:
        self.tokens.append(Token(token_type, value, line))
