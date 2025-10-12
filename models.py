from typing import Optional
from pydantic import BaseModel


class UserAccountContext(BaseModel):

    customer_id: int
    name: str
    tier: str = "basic"  # premium enterprise
    email: Optional[str] = None


class InputGuardrailOutput(BaseModel):

    is_off_topic: bool
    reason: str
