"""Generic durable inbox items and execution receipts."""

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue


class InboxItem(BaseModel):
    """A stable identity with JSON content owned by the input producer."""

    model_config = ConfigDict(extra="allow", frozen=True, allow_inf_nan=False, revalidate_instances="always")

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    id: str = Field(min_length=1, max_length=128)


class InboxReceipt(BaseModel):
    """The durable commit point for one processed input batch."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    item_ids: list[str] = Field(min_length=1)
    consume_ts: AwareDatetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
