"""Stream one string field out of a JSON object that arrives in arbitrary chunks.

The model streams `{"kind": "answer", "answer": "Within 72 hours [1]..."}`. To show the answer
as it is written, the text of `answer` is decoded (JSON escapes included) while the object is
still incomplete. The full object is parsed and validated at the end, as for non-streamed calls.
"""

import json

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class JsonFieldStream:
    def __init__(self, field: str) -> None:
        self._opening = f'"{field}"'
        self._buffer = ""  # everything received so far
        self._pos = 0  # next character of the buffer to examine
        self._state = "seek"  # seek -> colon -> quote -> value -> done

    @property
    def text(self) -> str:
        return self._buffer

    def feed(self, chunk: str) -> str:
        """Add a chunk; return the newly decoded characters of the field (possibly empty)."""
        self._buffer += chunk
        out: list[str] = []
        while self._pos < len(self._buffer) and self._state != "done":
            if self._state == "seek":
                i = self._buffer.find(self._opening, self._pos)
                if i < 0:  # keep a tail in case the key is split across chunks
                    self._pos = max(self._pos, len(self._buffer) - len(self._opening))
                    break
                self._pos = i + len(self._opening)
                self._state = "colon"
            elif self._state in ("colon", "quote"):
                ch = self._buffer[self._pos]
                self._pos += 1
                if ch.isspace():
                    continue
                expected = ":" if self._state == "colon" else '"'
                if ch != expected:  # `"answer"` was a value or a prefix, not our key
                    self._state = "seek"
                    continue
                self._state = "quote" if self._state == "colon" else "value"
            else:  # value
                decoded = self._decode()
                if decoded is None:
                    break  # incomplete escape: wait for the next chunk
                out.append(decoded)
        return "".join(out)

    def _decode(self) -> str | None:
        ch = self._buffer[self._pos]
        if ch == '"':
            self._pos += 1
            self._state = "done"
            return ""
        if ch != "\\":
            self._pos += 1
            return ch
        rest = self._buffer[self._pos + 1 :]
        if not rest:
            return None
        if rest[0] == "u":
            if len(rest) < 5:
                return None
            self._pos += 6
            return str(json.loads(f'"\\u{rest[1:5]}"'))
        self._pos += 2
        return _ESCAPES.get(rest[0], rest[0])
