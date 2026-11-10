"""CLI オプション値を検証する。

役割:
    Click callback とオプション間の整合性検証を提供する。
依存元:
    Click の例外型と math の有限値判定を使用する。
利用先:
    option_groups と options が引数検証に使用する。
処理フロー:
    値域と組み合わせを検証し、不正値を Click の例外で通知する。
"""

from math import isfinite

import click


def _validate_cli_scale(
    _ctx: click.Context,
    _param: click.Parameter,
    value: float,
) -> float:
    """scale が正の値であることを確認する。"""
    if not isfinite(value) or value <= 0:
        raise click.BadParameter("scale は有限な正の値を指定してください。")
    return value


def _validate_cli_positive_float(
    _ctx: click.Context,
    param: click.Parameter,
    value: float,
) -> float:
    """正の float オプションであることを確認する。"""
    if not isfinite(value) or value <= 0:
        raise click.BadParameter(f"{param.name} は有限な正の値を指定してください。")
    return value


def _validate_cli_optional_positive_float(
    _ctx: click.Context,
    param: click.Parameter,
    value: float | None,
) -> float | None:
    """省略可能な正の float オプションを確認する。"""
    if value is not None and (not isfinite(value) or value <= 0):
        raise click.BadParameter(f"{param.name} は有限な正の値を指定してください。")
    return value


def _validate_cli_non_negative_float(
    _ctx: click.Context,
    param: click.Parameter,
    value: float,
) -> float:
    """0以上の float オプションであることを確認する。"""
    if not isfinite(value) or value < 0:
        raise click.BadParameter(f"{param.name} は有限な0以上の値を指定してください。")
    return value


def _validate_cli_step_frames_range(
    _ctx: click.Context,
    _param: click.Parameter,
    value: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """歩画像の範囲が1始まりで昇順であることを確認する。"""
    if value is None:
        return None
    first, last = value
    if first < 1 or first > last:
        raise click.BadParameter("1 <= A <= B を満たす範囲を指定してください。")
    return value


def _validate_cli_finite_float(
    _ctx: click.Context,
    param: click.Parameter,
    value: float | None,
) -> float | None:
    """任意符号の float オプションが有限であることを確認する。"""
    if value is None:
        return None
    if not isfinite(value):
        raise click.BadParameter(f"{param.name} は有限な値を指定してください。")
    return value


def _validate_gyro_bias_options(
    gyro_bias_method: str,
    gyro_bias: float | None,
) -> None:
    """manual 指定時は明示的なジャイロバイアス値を必須にする。"""
    if gyro_bias_method == "manual" and gyro_bias is None:
        raise click.UsageError(
            "--gyro-bias-method manual を使う場合は --gyro-bias を指定してください。"
        )
    if gyro_bias is not None and not isfinite(gyro_bias):
        raise click.BadParameter("gyro_bias は有限な値を指定してください。")
