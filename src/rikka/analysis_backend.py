"""組み込み解析方式の入力変換、実行、成果物生成。

役割:
    既存方式の差異を公開窓口の背後で吸収し、解析と動画生成を提供する。

依存元:
    common の共有型・設定、PDR/PF/BLE の既存パイプライン、plot の動画描画を使う。
利用先:
    analysis_api が方式登録に対応する処理として呼び出す。
処理フロー:
    バイト入力を正規化して既存パイプラインを実行し、共通の軌跡・評価・成果物を
    返す。動画は計算済み状態を保存し、解析終了後や別プロセスで復元して描画する。
"""

from __future__ import annotations

import json
import math
import struct
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from rikka.analysis_contracts import ExecutionOptions
from rikka.ble.pipeline import run_ble_ranging_input
from rikka.common.lib.models import (
    FloorMap,
    Landmark,
    LandmarkCorrection,
    LandmarkCorrectionResult,
    LandmarkDetection,
    LandmarkRange,
    PathLossModel,
    RangingConsistency,
    StepHeading,
    TrajectoryResult,
)
from rikka.common.lib.trajectory_evaluation import (
    evaluate_trajectory_csv,
    write_evaluation_failure,
    write_evaluation_outputs,
)
from rikka.common.settings import (
    BleLandmarkSettings,
    HeadingSettings,
    ParticleSettings,
    PdrSettings,
    StepSettings,
)
from rikka.particle.pipeline import run_particle
from rikka.pdr.pipeline import run_pdr
from rikka.plot.lib.animation import save_particle_animation


class AnalysisInputError(ValueError):
    """入力の修正なしでは再試行できない解析エラー。"""


SENSOR_COLUMN_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "acce": {
        "t": ("t", "Time (s)", "time_s", "time_seconds"),
        "x": ("x", "Acceleration x (m/s^2)", "X (m/s^2)", "x(m/s^2)"),
        "y": ("y", "Acceleration y (m/s^2)", "Y (m/s^2)", "y(m/s^2)"),
        "z": ("z", "Acceleration z (m/s^2)", "Z (m/s^2)", "z(m/s^2)"),
    },
    "gyro": {
        "t": ("t", "Time (s)", "time_s", "time_seconds"),
        "x": ("x", "Gyroscope x (rad/s)", "X (rad/s)", "x(rad/s)"),
        "y": ("y", "Gyroscope y (rad/s)", "Y (rad/s)", "y(rad/s)"),
        "z": ("z", "Gyroscope z (rad/s)", "Z (rad/s)", "z(rad/s)"),
    },
}


def _find_column(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in dataframe), None)


def _sensor_dataframe(
    content: bytes,
    sensor_kind: str,
    recording_start_wall_time_ms: float | None = None,
) -> pd.DataFrame:
    try:
        source = pd.read_csv(BytesIO(content)).rename(
            columns=lambda column: str(column).strip()
        )
    except Exception as exc:
        raise AnalysisInputError(f"{sensor_kind} input is not a readable CSV") from exc

    normalized = pd.DataFrame()
    time_column = _find_column(source, SENSOR_COLUMN_ALIASES[sensor_kind]["t"])
    if time_column is not None:
        normalized["t"] = pd.to_numeric(source[time_column], errors="coerce")
    elif "timestamp_ns" in source:
        timestamps = pd.to_numeric(source["timestamp_ns"], errors="coerce")
        start_offset_s = 0.0
        if recording_start_wall_time_ms is not None and "wall_time_ms" in source:
            first_wall_time_ms = pd.to_numeric(
                source["wall_time_ms"], errors="coerce"
            ).iloc[0]
            start_offset_s = (
                float(first_wall_time_ms) - recording_start_wall_time_ms
            ) / 1_000.0
        normalized["t"] = (
            timestamps - timestamps.iloc[0]
        ) / 1_000_000_000 + start_offset_s
    elif "wall_time_ms" in source:
        timestamps = pd.to_numeric(source["wall_time_ms"], errors="coerce")
        normalized["t"] = (timestamps - timestamps.iloc[0]) / 1_000

    for axis in ("x", "y", "z"):
        column = _find_column(source, SENSOR_COLUMN_ALIASES[sensor_kind][axis])
        if column is not None:
            normalized[axis] = pd.to_numeric(source[column], errors="coerce")

    missing = {"x", "y", "z"} - set(normalized)
    if missing:
        raise AnalysisInputError(
            f"{sensor_kind} input is missing columns: {', '.join(sorted(missing))}"
        )
    return normalized


def _png_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
        raise AnalysisInputError("floor_map input is not a valid PNG")
    width, height = struct.unpack(">II", content[16:24])
    if width == 0 or height == 0:
        raise AnalysisInputError("floor_map dimensions must be positive")
    return width, height


def _recording_start_wall_time_ms(content: bytes) -> float:
    try:
        metadata = json.loads(content)
        value = metadata["recording_start_wall_time_ms"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError
        return float(value)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(
            "metadata input does not contain a valid recording_start_wall_time_ms"
        ) from exc


def _landmarks(content: bytes, map_dimensions: tuple[int, int]) -> tuple[Landmark, ...]:
    try:
        document = json.loads(content)
        rows = document["beacons"] if isinstance(document, dict) else document
        if not isinstance(rows, list) or not rows:
            raise ValueError("beacons must be a non-empty array")
        width, height = map_dimensions
        landmarks: list[Landmark] = []
        beacon_ids: set[str] = set()
        for row in rows:
            beacon_id = str(row["beacon_id"])
            pixel_x = float(row["pixel_x"])
            pixel_y = float(row["pixel_y"])
            position_sigma = (
                None
                if row.get("position_sigma_m") is None
                else float(row["position_sigma_m"])
            )
            heading = (
                None if row.get("heading_deg") is None else float(row["heading_deg"])
            )
            if not beacon_id or beacon_id in beacon_ids:
                raise ValueError("beacon_id must be non-empty and unique")
            if not all(math.isfinite(value) for value in (pixel_x, pixel_y)):
                raise ValueError("beacon coordinates must be finite")
            if not (0 <= pixel_x < width and 0 <= pixel_y < height):
                raise ValueError("beacon coordinates must be inside the floor map")
            if position_sigma is not None and (
                not math.isfinite(position_sigma) or position_sigma <= 0
            ):
                raise ValueError("position_sigma_m must be a finite positive number")
            if heading is not None and (
                not math.isfinite(heading) or not 0 <= heading <= 360
            ):
                raise ValueError("heading_deg must be between 0 and 360")
            beacon_ids.add(beacon_id)
            landmarks.append(
                Landmark(
                    beacon_id=beacon_id,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    position_sigma_m=position_sigma,
                    heading_deg=heading,
                )
            )
        return tuple(landmarks)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AnalysisInputError("beacon_layout input is invalid") from exc


def _trajectory_payload(
    result: TrajectoryResult,
    component_id: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    has_origin = all(
        key in parameters for key in ("origin_x", "origin_y", "floor_scale")
    )
    if has_origin:
        origin_x = float(parameters["origin_x"])
        origin_y = float(parameters["origin_y"])
        floor_scale = float(parameters["floor_scale"])
        if not all(math.isfinite(value) for value in (origin_x, origin_y)):
            raise AnalysisInputError("trajectory origin must be finite")
        if not math.isfinite(floor_scale) or floor_scale <= 0:
            raise AnalysisInputError("floor_scale must be finite and positive")
    elif any(key in parameters for key in ("origin_x", "origin_y", "floor_scale")):
        raise AnalysisInputError(
            "origin_x, origin_y, and floor_scale must be provided together"
        )

    timestamps: list[float | None] = [None, *map(float, result.t_at_steps)]
    if len(timestamps) != len(result.trajectory):
        timestamps = [None] * len(result.trajectory)
    points = [
        {
            "step_index": index,
            "timestamp_s": timestamps[index],
            "x": (
                origin_x + float(point[0]) / floor_scale
                if has_origin
                else float(point[0])
            ),
            "y": (
                origin_y + float(point[1]) / floor_scale
                if has_origin
                else float(point[1])
            ),
        }
        for index, point in enumerate(result.trajectory)
    ]
    return {
        "schema_version": "1",
        "coordinate_system": "pixel" if has_origin else "local_meter",
        "component_id": component_id,
        "points": points,
        "step_count": max(0, len(points) - 1),
        "landmark_correction_count": (
            0 if result.landmark is None else len(result.landmark.corrections)
        ),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))  # type: ignore[arg-type]
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_trajectory_csv(trajectory: dict[str, Any], path: Path) -> None:
    dataframe = pd.DataFrame(trajectory["points"])
    dataframe.to_csv(path, index=False)


def _restore_tuple(kind: Any, value: Any) -> Any:
    """旧NPZが持つ辞書形式と配列形式の両方を復元する。"""
    return kind(**value) if isinstance(value, dict) else kind(*value)


def _restore_landmark_result(value: dict[str, Any]) -> LandmarkCorrectionResult:
    """描画用の共有型を追加依存なしで復元する。"""
    fields = dict(value)
    fields["corrections"] = tuple(
        _restore_tuple(LandmarkCorrection, row) for row in value["corrections"]
    )
    detections = []
    for row in value.get("detections", ()):
        ranging = "distance_m" in row if isinstance(row, dict) else len(row) > 3
        detections.append(
            _restore_tuple(LandmarkRange if ranging else LandmarkDetection, row)
        )
    fields["detections"] = tuple(detections)
    landmarks = []
    for row in value.get("landmarks", ()):
        item = dict(row)
        if item.get("path_loss_model") is not None:
            item["path_loss_model"] = PathLossModel(**item["path_loss_model"])
        landmarks.append(Landmark(**item))
    fields["landmarks"] = tuple(landmarks)
    fields["ranging_consistency"] = tuple(
        RangingConsistency(**row) for row in value.get("ranging_consistency", ())
    )
    if value.get("step_headings") is not None:
        fields["step_headings"] = [
            _restore_tuple(StepHeading, row) for row in value["step_headings"]
        ]
    return LandmarkCorrectionResult(**fields)


@dataclass
class ParticleAnimation:
    """一時地図のパスに依存せず、再描画に必要な計算結果を保持する。"""

    source_kind: ClassVar[str] = "particle-animation-npz-v1"

    particles: np.ndarray[Any, Any]
    trajectory: list[list[float]]
    gx_mean: float
    gz_mean: float
    landmark: LandmarkCorrectionResult | None
    floor_map_bytes: bytes
    origin_px: tuple[int, int]
    scale: float

    def serialize(self) -> bytes:
        metadata = {
            "trajectory": self.trajectory,
            "gx_mean": self.gx_mean,
            "gz_mean": self.gz_mean,
            "landmark": _jsonable(self.landmark),
            "origin_px": self.origin_px,
            "scale": self.scale,
        }
        buffer = BytesIO()
        np.savez_compressed(
            buffer,
            particles=self.particles,
            floor_map=np.frombuffer(self.floor_map_bytes, dtype=np.uint8),
            metadata=np.asarray(json.dumps(_jsonable(metadata))),
        )
        return buffer.getvalue()

    @classmethod
    def deserialize(cls, body: bytes) -> ParticleAnimation:
        with np.load(BytesIO(body), allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata"]))
            landmark = metadata["landmark"]
            return cls(
                particles=source["particles"],
                trajectory=metadata["trajectory"],
                gx_mean=metadata["gx_mean"],
                gz_mean=metadata["gz_mean"],
                landmark=(
                    None if landmark is None else _restore_landmark_result(landmark)
                ),
                floor_map_bytes=source["floor_map"].tobytes(),
                origin_px=tuple(metadata["origin_px"]),
                scale=metadata["scale"],
            )

    def render(self) -> dict[str, dict[str, bytes]]:
        with tempfile.TemporaryDirectory(prefix="rikka-animation-") as directory:
            temporary = Path(directory)
            floor_path = temporary / "floor-map.png"
            floor_path.write_bytes(self.floor_map_bytes)
            generated = cast(
                Path | None,
                save_particle_animation(
                    self.particles,
                    self.trajectory,
                    gx_mean=self.gx_mean,
                    gz_mean=self.gz_mean,
                    floormap_path=str(floor_path),
                    origin_px=self.origin_px,
                    scale=self.scale,
                    output_path=temporary / "particle-filter.webm",
                    landmark=self.landmark,
                ),
            )
            candidates = (
                [Path(generated)]
                if generated is not None
                else list(temporary.glob("particle-filter.*"))
            )
            path = next(
                (path for path in candidates if path.suffix in {".webm", ".mp4"}),
                None,
            )
            if path is None:
                raise AnalysisInputError(
                    "particle animation WebM/MP4 could not be generated"
                )
            key = (
                "particle_animation"
                if path.suffix == ".webm"
                else "particle_animation_mp4"
            )
            return {key: {"body": path.read_bytes()}}


def execute_builtin(
    method_id: str,
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    options: ExecutionOptions,
    *,
    particle: bool,
    ble: bool,
) -> dict[str, Any]:
    """登録時に明示された構成で既存方式を実行する。"""
    component_id = "rikka-" + method_id
    uses_particle = particle
    uses_ble = ble
    if not all(isinstance(value, bytes) for value in inputs.values()):
        raise AnalysisInputError("Rikka inputs must be downloaded bytes")

    recording_start_wall_time_ms = _recording_start_wall_time_ms(inputs["metadata"])
    acce = _sensor_dataframe(
        inputs["acce"],
        "acce",
        recording_start_wall_time_ms,
    )
    gyro = _sensor_dataframe(
        inputs["gyro"],
        "gyro",
        recording_start_wall_time_ms,
    )

    with tempfile.TemporaryDirectory(prefix="rikka-analysis-") as directory:
        temporary = Path(directory)
        floor_map: FloorMap | None = None
        animation: ParticleAnimation | None = None
        if uses_particle or uses_ble:
            floor_path = temporary / "floor-map.png"
            floor_path.write_bytes(inputs["floor_map"])
            map_dimensions = _png_dimensions(inputs["floor_map"])
            origin = (
                int(parameters["origin_x"]),
                int(parameters["origin_y"]),
            )
            if not (
                0 <= origin[0] < map_dimensions[0]
                and 0 <= origin[1] < map_dimensions[1]
            ):
                raise AnalysisInputError("floor map origin must be inside the image")
            floor_map = FloorMap(
                path=str(floor_path),
                origin_px=origin,
                scale=float(parameters["floor_scale"]),
            )

        landmark_settings = BleLandmarkSettings(enabled=False)
        ranging = None
        if uses_ble:
            ble_path = temporary / "ble.csv"
            ble_path.write_bytes(inputs["ble"])
            # 入力metadataをBLE CSVと同じ一時ディレクトリへ配置する。
            (temporary / "metadata.json").write_bytes(inputs["metadata"])
            landmark_settings = BleLandmarkSettings(
                enabled=True,
                data_path=ble_path,
                rssi_threshold_dbm=float(parameters.get("rssi_threshold_dbm", -70.0)),
                landmarks=_landmarks(inputs["beacon_layout"], map_dimensions),
            )
            ranging = run_ble_ranging_input(landmark_settings)

        pdr_settings = PdrSettings(
            step=StepSettings(height_m=float(parameters.get("user_height_m", 1.68))),
            heading=HeadingSettings(
                initial_direction=float(parameters.get("initial_direction", 90.0))
            ),
            landmark=(
                landmark_settings
                if uses_ble and not uses_particle
                else BleLandmarkSettings(enabled=False)
            ),
        )
        pdr_result = run_pdr(
            pdr_settings,
            df_acc=acce,
            df_gyro=gyro,
            floormap=(floor_map if uses_ble and not uses_particle else None),
        )

        result = pdr_result
        if uses_particle:
            assert floor_map is not None
            particle_settings = ParticleSettings(
                floormap_path=floor_map.path,
                origin_px=floor_map.origin_px,
                scale=floor_map.scale,
                seed=(
                    None
                    if parameters.get("particle_seed") is None
                    else int(parameters["particle_seed"])
                ),
                count=int(parameters.get("particle_count", 500)),
                landmark_mode="ranging" if uses_ble else "hybrid",
            )
            result = run_particle(
                pdr_result.prepared,
                floor_map,
                particle_settings,
                detections=(None if ranging is None else ranging.detections),
                ranging_observations=(
                    None if ranging is None else ranging.observations
                ),
                landmark_settings=(landmark_settings if uses_ble else None),
            )

        if (
            uses_particle
            and result.particle is not None
            and "particle_animation" in options.requested_artifacts
        ):
            assert floor_map is not None
            animation = ParticleAnimation(
                particles=np.asarray(result.particle.all_particles),
                trajectory=result.trajectory,
                gx_mean=result.prepared.gx_mean,
                gz_mean=result.prepared.gz_mean,
                landmark=result.landmark,
                floor_map_bytes=inputs["floor_map"],
                origin_px=floor_map.origin_px,
                scale=floor_map.scale,
            )

    trajectory = _trajectory_payload(result, component_id, parameters)
    artifact_targets = options.requested_artifacts
    payloads: dict[str, dict[str, bytes]] = {}
    ground_truth_csv = options.ground_truth_csv
    if ground_truth_csv is not None:
        if not isinstance(ground_truth_csv, bytes):
            raise AnalysisInputError("ground truth input must be bytes")
        evaluation: dict[str, Any]
        payloads["ground_truth"] = {"body": ground_truth_csv}
        with tempfile.TemporaryDirectory(
            prefix="rikka-analysis-evaluation-"
        ) as directory:
            evaluation_dir = Path(directory)
            estimated_path = evaluation_dir / "estimated.csv"
            truth_path = evaluation_dir / "ground-truth.csv"
            _write_trajectory_csv(trajectory, estimated_path)
            truth_path.write_bytes(ground_truth_csv)
            try:
                evaluation_dataframe, summary = evaluate_trajectory_csv(
                    estimated_path,
                    truth_path,
                    coordinate_system=(
                        "pixel"
                        if trajectory["coordinate_system"] == "pixel"
                        else "meter"
                    ),
                    floor_scale=(
                        float(parameters["floor_scale"])
                        if trajectory["coordinate_system"] == "pixel"
                        else None
                    ),
                )
                evaluation_csv, evaluation_json = write_evaluation_outputs(
                    evaluation_dataframe,
                    summary,
                    evaluation_dir / "outputs",
                )
                payloads["evaluation_csv"] = {"body": evaluation_csv.read_bytes()}
                payloads["evaluation_summary"] = {"body": evaluation_json.read_bytes()}
                evaluation = {"status": "completed", "summary": _jsonable(summary)}
            except Exception as exc:
                # 評価失敗でも軌跡生成結果は返す。
                error_message = str(exc)
                failure_path = write_evaluation_failure(
                    evaluation_dir / "outputs", error_message
                )
                payloads["evaluation_summary"] = {"body": failure_path.read_bytes()}
                evaluation = {"status": "failed", "error": error_message}
        trajectory["evaluation"] = evaluation
    if uses_ble and result.landmark is not None and "diagnostics" in artifact_targets:
        diagnostics = {
            "schema_version": "1",
            "kind": "positioning_diagnostics",
            "landmark_corrections": _jsonable(result.landmark.corrections),
            "landmark": _jsonable(result.landmark),
            "particle": (
                None
                if result.particle is None
                else {"steps": _jsonable(result.particle.diagnostics)}
            ),
        }
        payloads["diagnostics"] = {
            "body": json.dumps(diagnostics, ensure_ascii=False).encode()
        }
    if animation is not None:
        trajectory["_deferred_artifacts"] = {"particle_animation": animation}
    if payloads:
        trajectory["_artifact_payloads"] = payloads
    return {"trajectory": trajectory}
