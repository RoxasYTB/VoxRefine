"""Small typed contracts shared by local audio engines."""

from dataclasses import dataclass
from typing import Literal, Protocol

from ..audio import AudioDomain


@dataclass(frozen=True)
class BackendSpec:
    id: str
    display_name: str
    backend_version: str
    mode: Literal["file", "stream"]
    input_domain: AudioDomain
    output_domain: AudioDomain
    required_extra: str | None
    supports_cpu: bool
    supports_gpu: bool
    preserves_channels: bool
    preserves_sample_rate: bool


class Backend(Protocol):
    name: str

    def spec(self) -> BackendSpec: ...

    def identity(self) -> dict[str, str]: ...
