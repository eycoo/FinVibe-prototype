from typing import TypedDict


class TransactionResult(TypedDict):
    intent: str
    amount: int
    category: str
    description: str


class ImageResult(TypedDict):
    status: str
    message: str
