"""解析方式の公開契約を定義する。

役割: 入出力とパラメータを通信層に依存せず表す。
依存元: 標準ライブラリのみ。利用先: analysis_api と呼び出しアプリ。
処理フロー: 方式を定義し、一覧取得時に独立した契約を返す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionOptions:
    """通信先を含まない成果物要求と任意の評価入力。"""

    requested_artifacts: tuple[str, ...] = ()
    ground_truth_csv: bytes | None = None


@dataclass(frozen=True)
class ArtifactDefinition:
    name: str
    format: str
    content_type: str
    deferred: bool = False


_EVALUATION_ARTIFACTS = (
    ArtifactDefinition("ground_truth", "csv", "text/csv"),
    ArtifactDefinition("evaluation_csv", "csv", "text/csv"),
    ArtifactDefinition("evaluation_summary", "json", "application/json"),
)
_DIAGNOSTICS = ArtifactDefinition("diagnostics", "json", "application/json")
_ANIMATION_ARTIFACTS = (
    ArtifactDefinition("particle_animation", "webm", "video/webm", deferred=True),
    ArtifactDefinition("particle_animation_mp4", "mp4", "video/mp4", deferred=True),
)


@dataclass(frozen=True)
class MethodDefinition:
    method_id: str
    display_name: str
    definition_version: str
    input_slots: tuple[dict[str, Any], ...]
    parameters_schema: dict[str, Any]
    output_slots: tuple[dict[str, Any], ...] = (
        {
            "slot_id": "trajectory",
            "accepted_contracts": (
                {"kind": "internal_value", "value_type": "trajectory"},
            ),
        },
    )

    artifacts: tuple[ArtifactDefinition, ...] = _EVALUATION_ARTIFACTS

    @property
    def component_id(self) -> str:
        return "rikka-" + self.method_id


def _slot(slot_id: str, data_type: str, format: str) -> dict[str, Any]:
    return {
        "slot_id": slot_id,
        "accepted_contracts": (
            {
                "kind": "asset",
                "data_type": data_type,
                "schema_version": "1",
                "format": format,
            },
        ),
    }


def _parameters_schema(
    *, map_required: bool, particle: bool, ble: bool
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "initial_direction": {"type": "number", "minimum": 0, "maximum": 360},
        "user_height_m": {"type": "number", "exclusiveMinimum": 0},
    }
    required: list[str] = []
    properties.update(
        {
            "origin_x": {"type": "integer", "minimum": 0},
            "origin_y": {"type": "integer", "minimum": 0},
            "floor_scale": {"type": "number", "exclusiveMinimum": 0},
        }
    )
    if map_required:
        required.extend(("origin_x", "origin_y", "floor_scale"))
    if particle:
        properties.update(
            {
                "particle_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100_000,
                },
                "particle_seed": {"type": ["integer", "null"]},
            }
        )
    if ble:
        properties["rssi_threshold_dbm"] = {
            "type": "number",
            "minimum": -150,
            "maximum": 0,
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def builtin_definitions() -> tuple[MethodDefinition, ...]:
    sensors = (
        _slot("acce", "acce", "csv"),
        _slot("gyro", "gyro", "csv"),
        _slot("metadata", "metadata", "json"),
    )
    floor = _slot("floor_map", "resource.floor_map", "png")
    ble_inputs = (
        _slot("ble", "ble", "csv"),
        floor,
        _slot("beacon_layout", "resource.beacon_layout", "json"),
    )
    return (
        MethodDefinition(
            "pdr",
            "PDR",
            "1.1.0",
            sensors,
            _parameters_schema(map_required=False, particle=False, ble=False),
        ),
        MethodDefinition(
            "pdr-particle-filter",
            "PDR + Particle Filter",
            "1.1.0",
            (*sensors, floor),
            _parameters_schema(map_required=True, particle=True, ble=False),
            artifacts=(*_EVALUATION_ARTIFACTS, *_ANIMATION_ARTIFACTS),
        ),
        MethodDefinition(
            "pdr-ble",
            "PDR + BLE",
            "1.2.0",
            (*sensors, *ble_inputs),
            _parameters_schema(map_required=True, particle=False, ble=True),
            artifacts=(*_EVALUATION_ARTIFACTS, _DIAGNOSTICS),
        ),
        MethodDefinition(
            "pdr-particle-filter-ble",
            "PDR + Particle Filter + BLE",
            "1.2.0",
            (*sensors, *ble_inputs),
            _parameters_schema(map_required=True, particle=True, ble=True),
            artifacts=(*_EVALUATION_ARTIFACTS, _DIAGNOSTICS, *_ANIMATION_ARTIFACTS),
        ),
    )
