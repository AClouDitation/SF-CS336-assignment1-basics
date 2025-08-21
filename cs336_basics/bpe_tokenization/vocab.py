class Vocab:
    def __init__(self, special_tokens: list[str]):
        self.special_tokens = special_tokens
        self._token_by_id = [i.to_bytes() for i in range(256)] + [
            token.encode() for token in special_tokens
        ]

    def token(self, id: int) -> bytes | None:
        return self._token_by_id[id]  

    def token_id(self, token: bytes) -> int:
        return self._token_by_id.index(token)

    def add(self, token: bytes) -> int:
        self._token_by_id.append(token)
        return len(self._token_by_id) - 1

    def __len__(self) -> int:
        return len(self._token_by_id)

    def __repr__(self) -> str:
        return f"Vocab({self._token_by_id})"
