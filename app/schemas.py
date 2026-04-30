from typing import TypedDict


class TransactionResult(TypedDict):
    intent: str
    amount: int
    category: str
    description: str


class PaymentReceiptResult(TypedDict):
    is_receipt: bool
    status: str
    amount: int
    sender_name: str
    recipient_name: str
    bank_or_app: str
    timestamp: str
    ref_no: str
