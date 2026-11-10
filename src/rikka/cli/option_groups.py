"""各 CLI コマンドの Click オプション群を定義する。

役割:
    各コマンドに同じ既定値と並び順のオプションを付与する。
依存元:
    Click、common の既定値・選択肢、validators の callback を使用する。
利用先:
    options の各コマンドから使用する。
処理フロー:
    デコレータを所定の順序で適用し、オプション付き関数を返す。
"""

from pathlib import Path

import click

from ..common.config import (
    BLE_DATA_PATH,
    BLE_LANDMARK_ENABLED,
    BLE_MAX_CORRECTION_M,
    BLE_MAX_WARP_SPAN_M,
    BLE_PDR_CORRECTION_MODE,
    BLE_PREFLIGHT_MODE,
    BLE_RETROFIT_DAMP_FACTORS,
    BLE_RETROFIT_FORWARD_MODE,
    BLE_RETROFIT_MAP_CHECK,
    BLE_RETROFIT_MAX_HEADING_DEG,
    BLE_RETROFIT_MIN_SPAN_M,
    BLE_RETROFIT_STRIDE_SCALE_MAX,
    BLE_RETROFIT_STRIDE_SCALE_MIN,
    BLE_RSSI_RELEASE_MARGIN_DB,
    BLE_RSSI_RELEASE_STREAK,
    BLE_RSSI_THRESHOLD_DBM,
    BLE_SAMPLE_MODE,
    BLE_SAMPLE_SEED,
    BLE_SAMPLE_TRUTH_PATH,
    BLE_SYNC_WINDOW_S,
    DATA_DIR,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MOTION_ESTIMATION,
    PF_CANDIDATE_COUNT,
    PF_CONVERGENCE_CELL_M,
    PF_CONVERGENCE_RESTORE_NORMAL,
    PF_CONVERGENCE_SINGLE_STEPS,
    PF_CONVERGENCE_SPREAD_M,
    PF_CONVERGENCE_TOP_K,
    PF_CONVERGENCE_WEIGHT_RATIO,
    PF_INIT_MODE,
    PF_LANDMARK_MODE,
    PF_LANDMARK_RETROFIT,
    PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    PF_NUM_PARTICLES,
    PF_PATH_SELECTION,
    PF_PROPOSAL_METHOD,
    PF_STEP_FRAMES_ARROWS,
    PF_STEP_FRAMES_DPI,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    SMOOTHING_MODE,
    STEP_DETECTION_METHOD,
    STEP_LENGTH_METHOD,
    USER_HEIGHT_M,
)
from ..common.lib.validation import (
    BLE_PDR_CORRECTION_MODES,
    BLE_PREFLIGHT_MODES,
    BLE_RETROFIT_FORWARD_MODES,
    BLE_RETROFIT_MAP_CHECK_MODES,
    BLE_SAMPLE_MODES,
    BLE_SAMPLE_SOURCES,
    FORWARD_HEADING_SOURCES,
    GYRO_BIAS_METHODS,
    HEADING_METHODS,
    MOTION_ESTIMATION_METHODS,
    MOTION_HEADING_CORRECTION_METHODS,
    PF_LANDMARK_MODES,
    PF_PATH_SELECTION_METHODS,
    SIDESTEP_HEADING_SOURCES,
    SIDESTEP_SMOOTHING_METHODS,
    SIDESTEP_SUSPECT_MODES,
    SMOOTHING_MODES,
    STEP_DETECTION_METHODS,
    STEP_LENGTH_METHODS,
)
from .validators import (
    _validate_cli_finite_float,
    _validate_cli_non_negative_float,
    _validate_cli_optional_positive_float,
    _validate_cli_positive_float,
    _validate_cli_scale,
    _validate_cli_step_frames_range,
)

_DATA_DIR_DEFAULT = DATA_DIR
_BLE_DATA_DEFAULT = BLE_DATA_PATH
_BLE_SAMPLE_SEED_DEFAULT = BLE_SAMPLE_SEED
_BLE_SAMPLE_MODE_DEFAULT = BLE_SAMPLE_MODE
_BLE_SAMPLE_TRUTH_DEFAULT = BLE_SAMPLE_TRUTH_PATH
_FLOORMAP_DEFAULT = FLOORMAP_PATH
_SCALE_DEFAULT = FLOORMAP_SCALE
_STEP_DETECTION_DEFAULT = STEP_DETECTION_METHOD
_STEP_DETECTION_CHOICES = STEP_DETECTION_METHODS
_GYRO_BIAS_METHOD_DEFAULT = GYRO_BIAS_METHOD
_GYRO_BIAS_METHOD_CHOICES = GYRO_BIAS_METHODS
_BLE_LANDMARK_DEFAULT = BLE_LANDMARK_ENABLED
_BLE_RSSI_THRESHOLD_DEFAULT = BLE_RSSI_THRESHOLD_DBM
_BLE_CORRECTION_DEFAULT = BLE_PDR_CORRECTION_MODE
_BLE_CORRECTION_CHOICES = BLE_PDR_CORRECTION_MODES
_BLE_PREFLIGHT_DEFAULT = BLE_PREFLIGHT_MODE
_BLE_PREFLIGHT_CHOICES = BLE_PREFLIGHT_MODES
_BLE_MAX_CORRECTION_DEFAULT = BLE_MAX_CORRECTION_M
_BLE_MAX_WARP_SPAN_DEFAULT = BLE_MAX_WARP_SPAN_M
_BLE_RETROFIT_FORWARD_DEFAULT = BLE_RETROFIT_FORWARD_MODE
_BLE_RETROFIT_MAX_HEADING_DEFAULT = BLE_RETROFIT_MAX_HEADING_DEG
_BLE_RETROFIT_SCALE_MIN_DEFAULT = BLE_RETROFIT_STRIDE_SCALE_MIN
_BLE_RETROFIT_SCALE_MAX_DEFAULT = BLE_RETROFIT_STRIDE_SCALE_MAX
_BLE_RETROFIT_MIN_SPAN_DEFAULT = BLE_RETROFIT_MIN_SPAN_M
_BLE_RETROFIT_MAP_CHECK_DEFAULT = BLE_RETROFIT_MAP_CHECK
_BLE_RETROFIT_DAMP_FACTORS_DEFAULT = BLE_RETROFIT_DAMP_FACTORS
_BLE_RELEASE_MARGIN_DEFAULT = BLE_RSSI_RELEASE_MARGIN_DB
_BLE_RELEASE_STREAK_DEFAULT = BLE_RSSI_RELEASE_STREAK
_BLE_SYNC_WINDOW_DEFAULT = BLE_SYNC_WINDOW_S
_ORIGIN_DEFAULT = FLOORMAP_ORIGIN_PX
_DIRECTION_DEFAULT = INITIAL_DIRECTION
_HEIGHT_DEFAULT = USER_HEIGHT_M
_STEP_LENGTH_DEFAULT = STEP_LENGTH_METHOD
_STEP_LENGTH_CHOICES = STEP_LENGTH_METHODS
_HEADING_METHOD_DEFAULT = HEADING_METHOD
_HEADING_METHOD_CHOICES = HEADING_METHODS
_SIDESTEP_LATERAL_RATIO_DEFAULT = SIDESTEP_LATERAL_RATIO
_SIDESTEP_MIN_LATERAL_DISPLACEMENT_DEFAULT = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M
_MOTION_HEADING_CORRECTION_DEFAULT = "auto"
_MOTION_HEADING_CORRECTION_CHOICES = MOTION_HEADING_CORRECTION_METHODS
_SIDESTEP_SMOOTHING_DEFAULT = SIDESTEP_SMOOTHING_METHOD
_SIDESTEP_SMOOTHING_CHOICES = SIDESTEP_SMOOTHING_METHODS
_FORWARD_HEADING_SOURCE_DEFAULT = FORWARD_HEADING_SOURCE
_FORWARD_HEADING_SOURCE_CHOICES = FORWARD_HEADING_SOURCES
_SIDESTEP_HEADING_SOURCE_DEFAULT = "motion"
_SIDESTEP_HEADING_SOURCE_CHOICES = SIDESTEP_HEADING_SOURCES
_SIDESTEP_SUSPECT_MODE_DEFAULT = SIDESTEP_SUSPECT_MODE
_SIDESTEP_SUSPECT_MODE_CHOICES = SIDESTEP_SUSPECT_MODES
_MOTION_ESTIMATION_DEFAULT = MOTION_ESTIMATION
_MOTION_ESTIMATION_CHOICES = MOTION_ESTIMATION_METHODS
_SMOOTHING_MODE_DEFAULT = SMOOTHING_MODE
_SMOOTHING_MODE_CHOICES = SMOOTHING_MODES
_PF_MOTION_PREDICTIVE_WEIGHT_POWER_DEFAULT = PF_MOTION_PREDICTIVE_WEIGHT_POWER
_PF_LANDMARK_MODE_DEFAULT = PF_LANDMARK_MODE
_PF_LANDMARK_MODE_CHOICES = PF_LANDMARK_MODES
_PF_LANDMARK_RETROFIT_DEFAULT = PF_LANDMARK_RETROFIT
_PF_NUM_PARTICLES_DEFAULT = PF_NUM_PARTICLES
_PF_PATH_SELECTION_DEFAULT = PF_PATH_SELECTION
_PF_PATH_SELECTION_CHOICES = PF_PATH_SELECTION_METHODS
_PF_STEP_FRAMES_ARROWS_DEFAULT = PF_STEP_FRAMES_ARROWS
_PF_STEP_FRAMES_DPI_DEFAULT = PF_STEP_FRAMES_DPI


def _common_options(f: click.decorators.FC) -> click.decorators.FC:
    """run / particle コマンド共通オプションをまとめたデコレータ。"""
    f = click.option(
        "--ble-retrofit-damp-factor",
        "ble_retrofit_damp_factors",
        type=float,
        multiple=True,
        default=_BLE_RETROFIT_DAMP_FACTORS_DEFAULT,
        show_default=True,
        callback=lambda ctx, param, value: tuple(
            _validate_cli_positive_float(ctx, param, item) for item in value
        ),
        help="地図違反時に試す相似補正の減衰係数（複数指定可）",
    )(f)
    f = click.option(
        "--ble-retrofit-map-check",
        type=click.Choice(BLE_RETROFIT_MAP_CHECK_MODES),
        default=_BLE_RETROFIT_MAP_CHECK_DEFAULT,
        show_default=True,
        help="相似補正後の壁交差の扱い",
    )(f)
    f = click.option(
        "--ble-retrofit-min-span",
        type=float,
        default=_BLE_RETROFIT_MIN_SPAN_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="相似補正を解く最小アンカー間距離 [m]",
    )(f)
    f = click.option(
        "--ble-retrofit-stride-scale-max",
        type=float,
        default=_BLE_RETROFIT_SCALE_MAX_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="相似補正で許容する歩幅倍率の上限",
    )(f)
    f = click.option(
        "--ble-retrofit-stride-scale-min",
        type=float,
        default=_BLE_RETROFIT_SCALE_MIN_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="相似補正で許容する歩幅倍率の下限",
    )(f)
    f = click.option(
        "--ble-retrofit-max-heading",
        type=float,
        default=_BLE_RETROFIT_MAX_HEADING_DEFAULT,
        show_default=True,
        callback=_validate_cli_non_negative_float,
        help="相似補正で許容する絶対回転角の上限 [deg]",
    )(f)
    f = click.option(
        "--ble-retrofit-forward",
        type=click.Choice(BLE_RETROFIT_FORWARD_MODES),
        default=_BLE_RETROFIT_FORWARD_DEFAULT,
        show_default=True,
        help="相似補正を後続歩へ保持するか固定するか",
    )(f)
    f = click.option(
        "--ble-max-warp-span",
        type=float,
        default=_BLE_MAX_WARP_SPAN_DEFAULT,
        show_default=True,
        callback=_validate_cli_optional_positive_float,
        help="warp区間の累積歩行距離上限 [m]（省略時は制限なし）",
    )(f)
    f = click.option(
        "--ble-max-correction",
        type=float,
        default=_BLE_MAX_CORRECTION_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="1回のBLE補正移動量の上限 [m]",
    )(f)
    f = click.option(
        "--ble-preflight",
        type=click.Choice(_BLE_PREFLIGHT_CHOICES),
        default=_BLE_PREFLIGHT_DEFAULT,
        show_default=True,
        help="RSSI距離整合FAIL時の扱い",
    )(f)
    f = click.option(
        "--smoothing",
        "smoothing_mode",
        type=click.Choice(_SMOOTHING_MODE_CHOICES),
        default=_SMOOTHING_MODE_DEFAULT,
        show_default=True,
        help="適応推定の因果処理またはオフライン平滑化",
    )(f)
    f = click.option(
        "--motion-estimation",
        type=click.Choice(_MOTION_ESTIMATION_CHOICES),
        default=_MOTION_ESTIMATION_DEFAULT,
        show_default=True,
        help="運動状態・方位・歩幅の推定方式",
    )(f)
    f = click.option(
        "--no-plot", is_flag=True, default=False, help="グラフ表示を無効化"
    )(f)
    f = click.option(
        "--ble-sync-window",
        type=float,
        default=_BLE_SYNC_WINDOW_DEFAULT,
        show_default=True,
        callback=_validate_cli_non_negative_float,
        help="同時受信としてまとめる時刻窓 [s]",
    )(f)
    f = click.option(
        "--ble-release-streak",
        type=click.IntRange(min=1),
        default=_BLE_RELEASE_STREAK_DEFAULT,
        show_default=True,
        help="接近ラッチ解除に必要な連続観測数",
    )(f)
    f = click.option(
        "--ble-release-margin",
        type=float,
        default=_BLE_RELEASE_MARGIN_DEFAULT,
        show_default=True,
        callback=_validate_cli_non_negative_float,
        help="接近ラッチ解除のRSSI余裕 [dB]",
    )(f)
    f = click.option(
        "--ble-correction",
        type=click.Choice(_BLE_CORRECTION_CHOICES),
        default=_BLE_CORRECTION_DEFAULT,
        show_default=True,
        help="通常PDRへのランドマーク補正方式",
    )(f)
    f = click.option(
        "--ble-rssi-threshold",
        type=float,
        default=_BLE_RSSI_THRESHOLD_DEFAULT,
        show_default=True,
        callback=_validate_cli_finite_float,
        help="ランドマーク検出とみなす RSSI の下限 [dBm]",
    )(f)
    f = click.option(
        "--ble-detection-profile",
        type=click.Choice(["auto", "measured", "synthetic"]),
        default="auto",
        show_default=True,
        help="BLE検出条件（autoはログ形式・metadataから判定）",
    )(f)
    f = click.option(
        "--ble-data",
        "ble_data_path",
        default=_BLE_DATA_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="BLE RSSI CSV のパス",
    )(f)
    f = click.option(
        "--ble-landmark/--no-ble-landmark",
        default=_BLE_LANDMARK_DEFAULT,
        show_default=True,
        help="BLE ランドマークによる位置補正の有効・無効",
    )(f)
    f = click.option(
        "--sidestep-smoothing",
        type=click.Choice(_SIDESTEP_SMOOTHING_CHOICES),
        default=_SIDESTEP_SMOOTHING_DEFAULT,
        show_default=True,
        help="横歩き判定の平滑化",
    )(f)
    f = click.option(
        "--forward-heading-source",
        type=click.Choice(_FORWARD_HEADING_SOURCE_CHOICES),
        default=_FORWARD_HEADING_SOURCE_DEFAULT,
        show_default=True,
        help="forward 判定ステップの軌跡方位ソース",
    )(f)
    f = click.option(
        "--sidestep-heading-source",
        type=click.Choice(_SIDESTEP_HEADING_SOURCE_CHOICES),
        default=_SIDESTEP_HEADING_SOURCE_DEFAULT,
        show_default=True,
        help="確定横歩きステップの軌跡方位ソース",
    )(f)
    f = click.option(
        "--sidestep-suspect-mode",
        type=click.Choice(_SIDESTEP_SUSPECT_MODE_CHOICES),
        default=_SIDESTEP_SUSPECT_MODE_DEFAULT,
        show_default=True,
        help="横歩き疑いステップの軌跡反映モード",
    )(f)
    f = click.option(
        "--motion-heading-correction",
        type=click.Choice(_MOTION_HEADING_CORRECTION_CHOICES),
        default=_MOTION_HEADING_CORRECTION_DEFAULT,
        show_default=True,
        help="水平加速度移動方向の固定ずれ補正",
    )(f)
    f = click.option(
        "--sidestep-min-lateral-displacement",
        type=float,
        callback=_validate_cli_non_negative_float,
        default=_SIDESTEP_MIN_LATERAL_DISPLACEMENT_DEFAULT,
        show_default=True,
        help="横歩き判定に必要な横方向変位の最小値 [m]",
    )(f)
    f = click.option(
        "--sidestep-lateral-ratio",
        type=float,
        callback=_validate_cli_positive_float,
        default=_SIDESTEP_LATERAL_RATIO_DEFAULT,
        show_default=True,
        help="横歩き判定に使う 横方向/前方向 の最小比率",
    )(f)
    f = click.option(
        "--step-length-method",
        type=click.Choice(_STEP_LENGTH_CHOICES),
        default=_STEP_LENGTH_DEFAULT,
        show_default=True,
        help="歩幅推定手法",
    )(f)
    f = click.option(
        "--step-detection",
        type=click.Choice(_STEP_DETECTION_CHOICES),
        default=_STEP_DETECTION_DEFAULT,
        show_default=True,
        help="ステップ検出手法（adaptive_peak はオフライン）",
    )(f)
    f = click.option(
        "--heading-method",
        type=click.Choice(_HEADING_METHOD_CHOICES),
        default=_HEADING_METHOD_DEFAULT,
        show_default=True,
        help="方位推定手法",
    )(f)
    f = click.option(
        "--gyro-bias-method",
        type=click.Choice(_GYRO_BIAS_METHOD_CHOICES),
        default=_GYRO_BIAS_METHOD_DEFAULT,
        show_default=True,
        help="ジャイロバイアス推定手法",
    )(f)
    f = click.option(
        "--gyro-bias",
        type=float,
        callback=_validate_cli_finite_float,
        default=None,
        help="manual 指定時のジャイロバイアス [rad/s]",
    )(f)
    f = click.option(
        "--direction",
        type=float,
        callback=_validate_cli_finite_float,
        default=_DIRECTION_DEFAULT,
        show_default=True,
        help="歩行開始方向のオフセット [度]",
    )(f)
    f = click.option(
        "--height-m",
        type=float,
        callback=_validate_cli_positive_float,
        default=_HEIGHT_DEFAULT,
        show_default=True,
        help="歩幅推定に使うユーザー身長 [m]",
    )(f)
    f = click.option(
        "--scale",
        type=float,
        callback=_validate_cli_scale,
        default=_SCALE_DEFAULT,
        show_default=True,
        help="1ピクセルあたりのメートル数",
    )(f)
    f = click.option(
        "--origin-px",
        nargs=2,
        type=int,
        default=_ORIGIN_DEFAULT,
        show_default=True,
        help="軌跡起点のピクセル座標 X Y",
    )(f)
    f = click.option(
        "--floormap",
        "-f",
        default=_FLOORMAP_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="フロアマップ画像のパス",
    )(f)
    f = click.option(
        "--data-dir",
        "-d",
        default=_DATA_DIR_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="入力データフォルダ",
    )(f)
    return f


def _particle_options(f: click.decorators.FC) -> click.decorators.FC:
    """PF 専用オプションと共通オプションを元の順序で適用する。"""
    f = click.option(
        "--pf-backward/--no-pf-backward",
        default=False,
        show_default=True,
        help="検証用の独立した後方PFを実行し、前向き経路と比較する",
    )(f)
    f = _common_options(f)
    f = click.option(
        "--pf-hard-anchors",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="根拠付き確定位置イベントのJSON（歩境界時刻のみ）",
    )(f)
    f = click.option(
        "--floormap-floor-id",
        default=None,
        help="使用する地図の階ID（階ID付き確定位置との照合用）",
    )(f)
    f = click.option(
        "--save-animation",
        is_flag=True,
        default=False,
        help="--no-plot 指定時もパーティクルフィルタのアニメーションを保存",
    )(f)
    f = click.option(
        "--save-step-frames",
        is_flag=True,
        default=False,
        help="1歩ごとの段階別パーティクル画像を保存",
    )(f)
    f = click.option(
        "--step-frames-range",
        type=int,
        nargs=2,
        default=None,
        callback=_validate_cli_step_frames_range,
        metavar="A B",
        help="保存する歩の範囲（1始まり、両端含む）",
    )(f)
    f = click.option(
        "--step-frames-arrows",
        type=click.IntRange(min=0),
        default=_PF_STEP_FRAMES_ARROWS_DEFAULT,
        show_default=True,
        help="段階別画像へ描く重み上位の方位矢印数",
    )(f)
    f = click.option(
        "--step-frames-dpi",
        type=click.IntRange(min=1),
        default=_PF_STEP_FRAMES_DPI_DEFAULT,
        show_default=True,
        help="段階別画像と代表軌跡比較図の解像度",
    )(f)
    f = click.option(
        "--history-candidate",
        is_flag=True,
        default=False,
        help="前向きPFを維持し、過去軌跡の修正候補を別CSV・比較図へ保存（試作）",
    )(f)
    f = click.option(
        "--save-path-comparison",
        is_flag=True,
        default=False,
        help="代表軌跡候補の比較図を保存",
    )(f)
    f = click.option(
        "--pf-seed",
        type=int,
        default=None,
        help="パーティクルフィルタ乱数の seed（回帰検証用）",
    )(f)
    f = click.option(
        "--pf-particles",
        type=click.IntRange(min=1),
        default=_PF_NUM_PARTICLES_DEFAULT,
        show_default=True,
        help="パーティクルフィルタで使用する粒子数",
    )(f)
    f = click.option(
        "--motion-predictive-weight-power",
        type=float,
        callback=_validate_cli_non_negative_float,
        default=_PF_MOTION_PREDICTIVE_WEIGHT_POWER_DEFAULT,
        show_default=True,
        help="運動状態の予測尤度をPF重みに掛ける指数（0で無効）",
    )(f)
    f = click.option(
        "--pf-path-selection",
        type=click.Choice(_PF_PATH_SELECTION_CHOICES),
        default=_PF_PATH_SELECTION_DEFAULT,
        show_default=True,
        help="PFの代表軌跡選択方式",
    )(f)
    f = click.option(
        "--pf-landmark-mode",
        type=click.Choice(_PF_LANDMARK_MODE_CHOICES),
        default=_PF_LANDMARK_MODE_DEFAULT,
        show_default=True,
        help="particle filter へのランドマーク反映方式",
    )(f)
    f = click.option(
        "--pf-landmark-retrofit/--no-pf-landmark-retrofit",
        default=_PF_LANDMARK_RETROFIT_DEFAULT,
        show_default=True,
        help="PF代表軌跡のランドマーク不連続を相似補正",
    )(f)
    f = click.option(
        "--pf-candidates",
        type=click.IntRange(min=1),
        default=None,
        show_default=f"candidate-bank: {PF_CANDIDATE_COUNT}, その他: 1",
        help="candidate-bank方式の親粒子ごとの候補数（省略時は方式に合わせる）",
    )(f)
    f = click.option(
        "--pf-proposal",
        type=click.Choice(["legacy", "no-relocation", "candidate-bank"]),
        default=PF_PROPOSAL_METHOD,
        show_default=True,
        help="PFの移動候補生成方式",
    )(f)
    f = click.option(
        "--pf-init",
        type=click.Choice(["origin", "uniform"]),
        default=PF_INIT_MODE,
        show_default=True,
        help="PFの初期分布（uniformは初期位置・初期方位なし）",
    )(f)
    f = click.option(
        "--pf-converge-restore-normal/--no-pf-converge-restore-normal",
        default=PF_CONVERGENCE_RESTORE_NORMAL,
        show_default=True,
        help="収束後に通常PF（recovery・ESS 0.5）へ戻す。既定では戻さない",
    )(f)
    f = click.option(
        "--pf-converge-steps",
        type=click.IntRange(min=1),
        default=PF_CONVERGENCE_SINGLE_STEPS,
        show_default=True,
        help="一様初期化PFの収束条件が連続成立する歩数",
    )(f)
    f = click.option(
        "--pf-converge-spread",
        type=click.FloatRange(min=0.0, min_open=False),
        default=PF_CONVERGENCE_SPREAD_M,
        show_default=True,
        help="一様初期化PFの収束に許容するクラスタRMS広がり [m]",
    )(f)
    f = click.option(
        "--pf-converge-weight-ratio",
        type=click.FloatRange(min=0.0, max=1.0, min_open=True),
        default=PF_CONVERGENCE_WEIGHT_RATIO,
        show_default=True,
        help="一様初期化PFの収束に必要な最大クラスタ重み比",
    )(f)
    f = click.option(
        "--pf-converge-top-k",
        type=click.IntRange(min=1),
        default=PF_CONVERGENCE_TOP_K,
        show_default=True,
        help="一様初期化PFで記録する上位クラスタ数",
    )(f)
    f = click.option(
        "--pf-converge-cell",
        type=click.FloatRange(min=0.0, min_open=True),
        default=PF_CONVERGENCE_CELL_M,
        show_default=True,
        help="一様初期化PFのクラスタセル幅 [m]",
    )(f)
    return f


def _run_options(f: click.decorators.FC) -> click.decorators.FC:
    """run コマンドのオプションを所定の順序で適用する。"""
    f = click.option(
        "--free-origin",
        is_flag=True,
        default=False,
        help="初期位置・初期方位を与えず (0, 0) 起点・方位0度で推定する",
    )(f)
    return f


def _sensor_options(f: click.decorators.FC) -> click.decorators.FC:
    """sensor コマンドのオプションを所定の順序で適用する。"""
    f = click.option(
        "--gyro-bias",
        type=float,
        callback=_validate_cli_finite_float,
        default=None,
        help="manual 指定時のジャイロバイアス [rad/s]",
    )(f)
    f = click.option(
        "--gyro-bias-method",
        type=click.Choice(_GYRO_BIAS_METHOD_CHOICES),
        default=_GYRO_BIAS_METHOD_DEFAULT,
        show_default=True,
        help="ジャイロバイアス推定手法",
    )(f)
    f = click.option(
        "--step-detection",
        type=click.Choice(_STEP_DETECTION_CHOICES),
        default=_STEP_DETECTION_DEFAULT,
        show_default=True,
        help="ステップ検出手法（adaptive_peak はオフライン）",
    )(f)
    f = click.option(
        "--data-dir",
        "-d",
        default=_DATA_DIR_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="入力データフォルダ",
    )(f)
    return f


def _ble_sample_options(f: click.decorators.FC) -> click.decorators.FC:
    """ble_sample コマンドのオプションを所定の順序で適用する。"""
    f = click.option(
        "--seed",
        type=int,
        default=_BLE_SAMPLE_SEED_DEFAULT,
        show_default=True,
        help="サンプル生成の乱数シード",
    )(f)
    f = click.option(
        "--output",
        "-o",
        default=_BLE_DATA_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="生成する BLE RSSI CSV の保存先",
    )(f)
    f = click.option(
        "--data-dir",
        "-d",
        default=_DATA_DIR_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="時間軸の基準にする入力データフォルダ",
    )(f)
    f = click.option(
        "--truth-csv",
        default=_BLE_SAMPLE_TRUTH_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="source=truth で使う正解軌跡CSV",
    )(f)
    f = click.option(
        "--source",
        type=click.Choice(BLE_SAMPLE_SOURCES),
        default="pdr",
        show_default=True,
        help="distance方式で使う歩行者位置の取得元",
    )(f)
    f = click.option(
        "--mode",
        type=click.Choice(BLE_SAMPLE_MODES),
        default=_BLE_SAMPLE_MODE_DEFAULT,
        show_default=True,
        help="RSSI生成方式",
    )(f)
    return f


def _landmark_path_options(f: click.decorators.FC) -> click.decorators.FC:
    """landmark_path コマンドのオプションを所定の順序で適用する。"""
    f = click.option(
        "--no-plot", is_flag=True, default=False, help="グラフ表示を無効化"
    )(f)
    f = click.option(
        "--scale",
        type=float,
        callback=_validate_cli_scale,
        default=_SCALE_DEFAULT,
        show_default=True,
        help="1ピクセルあたりのメートル数",
    )(f)
    f = click.option(
        "--floormap",
        "-f",
        default=_FLOORMAP_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="フロアマップ画像のパス",
    )(f)
    f = click.option(
        "--order",
        type=click.Choice(("rssi-peak", "given")),
        default="rssi-peak",
        show_default=True,
        help="経路順。rssi-peak は計測BLEの平滑RSSIが最大の時刻順、given は指定順",
    )(f)
    f = click.option(
        "--landmark",
        "landmark_specs",
        multiple=True,
        help=(
            "経路順に結ぶランドマークを "
            "beacon_id:ピクセルX:ピクセルY で指定（複数指定可、"
            "指定すると計測フォルダの配置より優先する）"
        ),
    )(f)
    f = click.option(
        "--data-dir",
        "-d",
        default=_DATA_DIR_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="ランドマーク配置とBLEログを読む計測フォルダ",
    )(f)
    return f


def _evaluate_options(f: click.decorators.FC) -> click.decorators.FC:
    """evaluate コマンドのオプションを所定の順序で適用する。"""
    f = click.option(
        "--output-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=Path("output/evaluation"),
        show_default=True,
        help="評価CSVと集計JSONの出力先",
    )(f)
    f = click.option(
        "--tolerance-m",
        type=float,
        default=1.0,
        show_default=True,
        callback=_validate_cli_non_negative_float,
        help="許容誤差以内率に使う許容誤差 [m]",
    )(f)
    f = click.option(
        "--floor-scale",
        type=float,
        default=_SCALE_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="pixel座標の1pxあたりのメートル数",
    )(f)
    f = click.option(
        "--coordinate-system",
        type=click.Choice(("pixel", "meter")),
        default="pixel",
        show_default=True,
        help="推定・正解軌跡の座標系",
    )(f)
    f = click.option(
        "--truth-csv",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        required=True,
        help="比較対象の正解軌跡CSV",
    )(f)
    f = click.option(
        "--estimated-csv",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        required=True,
        help="評価対象の推定軌跡CSV",
    )(f)
    return f
