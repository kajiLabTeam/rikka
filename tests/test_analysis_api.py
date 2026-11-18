"""公開方式契約と保存成果物の互換性を検証する。"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from rikka import analysis_api
from rikka.analysis_api import (
    AnalysisInputError,
    execute_method,
    list_methods,
    render_artifact,
)
from rikka.analysis_backend import ParticleAnimation
from rikka.analysis_contracts import ExecutionOptions


def test_catalog_contract_is_independent_of_caller_mutations() -> None:
    definitions = list_methods()
    assert [method.method_id for method in definitions] == [
        "pdr",
        "pdr-particle-filter",
        "pdr-ble",
        "pdr-particle-filter-ble",
    ]
    definitions[0].parameters_schema["properties"].clear()
    assert "initial_direction" in list_methods()[0].parameters_schema["properties"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"particle_count": 1},
        {"initial_direction": float("nan")},
        {"initial_direction": True},
        {"initial_direction": 361},
        {"user_height_m": 0},
        {"unknown": 0},
    ],
)
def test_public_api_rejects_invalid_parameters(parameters: dict[str, Any]) -> None:
    with pytest.raises(AnalysisInputError):
        execute_method("pdr", {"acce": b"", "gyro": b"", "metadata": b""}, parameters)


def test_versioned_artifact_restores_after_execution_and_reads_legacy_sources(
    monkeypatch: Any,
) -> None:
    animation = ParticleAnimation(
        particles=np.array([[[1.0, 2.0, 3.0]]]),
        trajectory=[[0.0, 0.0]],
        gx_mean=0.0,
        gz_mean=9.8,
        landmark=None,
        floor_map_bytes=b"map",
        origin_px=(1, 2),
        scale=0.05,
    )
    definition = replace(list_methods()[0], method_id="test-render")

    def execute(
        _inputs: dict[str, bytes],
        _parameters: dict[str, Any],
        _options: ExecutionOptions,
    ) -> dict[str, Any]:
        return {
            "trajectory": {"_deferred_artifacts": {"particle_animation": animation}}
        }

    monkeypatch.setitem(
        analysis_api.METHODS,
        definition.method_id,
        analysis_api.Method(definition, execute),
    )
    artifact = execute_method(
        "test-render", {"acce": b"", "gyro": b"", "metadata": b""}, {}
    )["trajectory"]["_deferred_artifacts"]["particle_animation"]
    source = artifact.serialize()
    assert b'"method_id": "test-render"' in source
    assert b'"definition_version": "1.1.0"' in source

    def render(self: ParticleAnimation) -> dict[str, dict[str, bytes]]:
        np.testing.assert_array_equal(self.particles, animation.particles)
        assert self.floor_map_bytes == b"map"
        return {"particle_animation": {"body": b"video"}}

    monkeypatch.setattr(ParticleAnimation, "render", render)
    assert render_artifact(source) == {"particle_animation": {"body": b"video"}}
    assert render_artifact(animation.serialize()) == render_artifact(source)
    with pytest.raises(AnalysisInputError, match="unsupported artifact"):
        render_artifact(b'rikka-artifact-v1\n{"source_version": 2}\nignored')
