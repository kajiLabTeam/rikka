"""アプリから利用する解析・成果物の共通窓口。

役割: 方式契約の公開、明示登録された実行処理への振り分け、成果物復元。
依存元: analysis_contracts の契約と analysis_backend の既存方式アダプター。
利用先: Nozomi等の実行基盤。HTTPやストレージには依存しない。
処理フロー: 契約取得、方式ID指定の実行、軌跡返却、遅延成果物の保存・描画。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from typing import Any, Protocol

from rikka.analysis_backend import (
    AnalysisInputError as AnalysisInputError,
)
from rikka.analysis_backend import (
    ParticleAnimation,
    execute_builtin,
)
from rikka.analysis_contracts import (
    ExecutionOptions,
    MethodDefinition,
    builtin_definitions,
)

Executor = Callable[
    [dict[str, bytes], dict[str, Any], ExecutionOptions], dict[str, Any]
]


class DeferredArtifact(Protocol):
    """保存素材と描画結果を返す、方式に依存しない成果物契約。"""

    @property
    def source_kind(self) -> str: ...

    def serialize(self) -> bytes: ...
    def render(self) -> dict[str, dict[str, bytes]]: ...


@dataclass(frozen=True)
class Method:
    definition: MethodDefinition
    execute: Executor


# 任意モジュールの探索は行わず、対応する処理を明示する。
_DEFINITIONS = {
    definition.method_id: definition for definition in builtin_definitions()
}
METHODS: dict[str, Method] = {
    "pdr": Method(
        _DEFINITIONS["pdr"], partial(execute_builtin, "pdr", particle=False, ble=False)
    ),
    "pdr-particle-filter": Method(
        _DEFINITIONS["pdr-particle-filter"],
        partial(execute_builtin, "pdr-particle-filter", particle=True, ble=False),
    ),
    "pdr-ble": Method(
        _DEFINITIONS["pdr-ble"],
        partial(execute_builtin, "pdr-ble", particle=False, ble=True),
    ),
    "pdr-particle-filter-ble": Method(
        _DEFINITIONS["pdr-particle-filter-ble"],
        partial(execute_builtin, "pdr-particle-filter-ble", particle=True, ble=True),
    ),
}


def list_methods() -> tuple[MethodDefinition, ...]:
    """呼び出し元の変更が登録内容に影響しない契約一覧を返す。"""
    return tuple(deepcopy(method.definition) for method in METHODS.values())


_MAGIC = b"rikka-artifact-v1\n"


@dataclass
class _VersionedArtifact:
    artifact: DeferredArtifact
    method_id: str
    definition_version: str
    source_kind: str

    def serialize(self) -> bytes:
        header = json.dumps(
            {
                "kind": self.source_kind,
                "method_id": self.method_id,
                "definition_version": self.definition_version,
                "rikka_version": version("rikka"),
                "source_version": 1,
            }
        ).encode()
        return _MAGIC + header + b"\n" + self.artifact.serialize()

    def render(self) -> dict[str, dict[str, bytes]]:
        return self.artifact.render()


# 新たな保存形式はここへ明示登録する。既存形式は後方互換を保つ。
ARTIFACT_RENDERERS: dict[str, Callable[[bytes], DeferredArtifact]] = {
    "particle_animation": ParticleAnimation.deserialize,
    "particle-animation-npz-v1": ParticleAnimation.deserialize,
}


def render_artifact(source: bytes) -> dict[str, dict[str, bytes]]:
    """保存した不透明な素材を復元し、成果物を生成する。旧NPZも受け付ける。"""
    if not source.startswith(_MAGIC):
        return ParticleAnimation.deserialize(source).render()
    try:
        header_bytes, payload = source[len(_MAGIC) :].split(b"\n", 1)
        header = json.loads(header_bytes)
        if header["source_version"] != 1:
            raise ValueError("unsupported artifact source version")
        restore = ARTIFACT_RENDERERS[header["kind"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise AnalysisInputError("unsupported artifact source") from exc
    return restore(payload).render()


def execute_method(
    method_id: str,
    inputs: dict[str, bytes],
    parameters: dict[str, Any],
    *,
    options: ExecutionOptions | None = None,
) -> dict[str, Any]:
    """方式IDを指定して実行し、共通の軌跡・成果物形式で返す。"""
    method = METHODS.get(method_id)
    if method is None:
        raise AnalysisInputError(f"unsupported Rikka method: {method_id}")
    if not all(isinstance(body, bytes) for body in inputs.values()):
        raise AnalysisInputError("Rikka inputs must be downloaded bytes")
    required = {
        slot["slot_id"]
        for slot in method.definition.input_slots
        if slot.get("required", True)
    }
    if not required.issubset(inputs):
        raise AnalysisInputError("missing required Rikka inputs")
    validate_parameters(method.definition.parameters_schema, parameters)
    result = method.execute(inputs, parameters, options or ExecutionOptions())
    trajectory = result.get("trajectory", {})
    for name, artifact in trajectory.get("_deferred_artifacts", {}).items():
        trajectory["_deferred_artifacts"][name] = _VersionedArtifact(
            artifact,
            method_id,
            method.definition.definition_version,
            artifact.source_kind,
        )
    return result


def validate_parameters(schema: dict[str, Any], parameters: dict[str, Any]) -> None:
    """公開契約で対応する単純なJSON Schemaの型と数値制約を検証する。"""
    properties = schema.get("properties", {})
    if (
        schema.get("additionalProperties") is False
        and set(parameters) - set(properties)
        or not set(schema.get("required", ())).issubset(parameters)
    ):
        raise AnalysisInputError("parameters do not match schema")
    for key, value in parameters.items():
        constraints = properties.get(key, {})
        expected = constraints.get("type")
        types = {expected} if isinstance(expected, str) else set(expected or ())
        matches = {
            "null": value is None,
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "string": isinstance(value, str),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }
        if types and not any(matches.get(kind, False) for kind in types):
            raise AnalysisInputError(f"parameter {key} has invalid type")
        if "enum" in constraints and value not in constraints["enum"]:
            raise AnalysisInputError(f"parameter {key} is not in enum")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise AnalysisInputError(f"parameter {key} must be finite")
            if (
                "minimum" in constraints
                and value < constraints["minimum"]
                or "maximum" in constraints
                and value > constraints["maximum"]
                or "exclusiveMinimum" in constraints
                and value <= constraints["exclusiveMinimum"]
            ):
                raise AnalysisInputError(f"parameter {key} is outside allowed range")
