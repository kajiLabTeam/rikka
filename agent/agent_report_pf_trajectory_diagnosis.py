"""保存済みPF診断データから、日本語の原因分析レポートを生成する。

役割: 再配置・重み・系譜・代表経路の測定値を対応付ける。
入力: 専用ディレクトリの集計CSV、分割粒子CSV、run_metadata.json。
出力: diagnostic_report.md。入力や製品処理は変更しない。
処理フロー: 集計と粒子データを読み、全実行・異常歩・過去との差を表にする。
"""

# Markdownの表と段落を文字列のまま保持する。
# ruff: noqa: E501

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def table(frame, columns):
    """追加依存なしでMarkdown表を作る。"""

    def value(item):
        if pd.isna(item):
            return "欠損"
        if isinstance(item, (float, np.floating)):
            return f"{item:.4f}"
        return str(item).replace("|", " / ").replace("\n", " ")

    keys, titles = zip(*columns, strict=True)
    lines = ["|" + "|".join(titles) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
    lines.extend(
        "|" + "|".join(value(row[k]) for k in keys) + "|" for _, row in frame.iterrows()
    )
    return "\n".join(lines)


def historical_comparison(trajectory):
    """過去保存座標との差を測り、条件一致の再現試験と区別する。"""
    source = (
        Path(__file__).resolve().parents[1]
        / "agent/docs-archive/pf-history-prototype-20261022/trajectories.csv"
    )
    if not source.exists():
        return "過去保存座標が見つからず比較未実施。"
    old = pd.read_csv(source)
    rows = []
    for (dataset, seed), current in trajectory.groupby(["dataset", "seed"]):
        prior = old[(old.dataset == dataset) & (old.seed == seed)]
        merged = current.merge(prior, on="step")
        if len(merged):
            delta = np.hypot(
                merged.final_x - merged.original_x_m,
                merged.final_y - merged.original_y_m,
            )
            rows.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    points=len(merged),
                    maximum=delta.max(),
                    endpoint=delta.iloc[-1],
                )
            )
    return (
        table(
            pd.DataFrame(rows),
            [
                ("dataset", "計測"),
                ("seed", "seed"),
                ("points", "対応点数"),
                ("maximum", "最大座標差m"),
                ("endpoint", "終点差m"),
            ],
        )
        if rows
        else "共通計測・seedなし。"
    )


def lineage_evidence(output, manifest, trajectory):
    """sequenceの最大変位辺を親対応した粒子の実座標と照合する。"""
    rows = []
    for item in manifest.to_dict("records"):
        path = trajectory[
            (trajectory.dataset == item["dataset"]) & (trajectory.seed == item["seed"])
        ]
        top = path.nlargest(2, "sequence_step_m")
        particles = pd.read_csv(output / item["path"])
        for _, point in top.iterrows():
            match = particles[
                (particles.step == point.step)
                & (particles.particle_id == point.sequence_particle_id)
            ]
            if match.empty:
                continue
            particle = match.iloc[0]
            previous = path[path.step == point.step - 1].iloc[0]
            after_error = np.hypot(
                point.sequence_x - particle.after_x,
                point.sequence_y - particle.after_y,
            )
            parent_error = np.hypot(
                previous.sequence_x - particle.after_parent_before_x,
                previous.sequence_y - particle.after_parent_before_y,
            )
            rows.append(
                dict(
                    dataset=item["dataset"],
                    seed=item["seed"],
                    step=point.step,
                    distance=point.sequence_step_m,
                    child=int(particle.particle_id),
                    parent=int(particle.parent_particle_id),
                    relocated=particle.after_parent_relocated,
                    from_x=particle.after_parent_before_x,
                    from_y=particle.after_parent_before_y,
                    to_x=particle.after_x,
                    to_y=particle.after_y,
                    equal=bool(after_error <= 1e-10 and parent_error <= 1e-10),
                )
            )
    return (
        table(
            pd.DataFrame(rows),
            [
                ("dataset", "計測"),
                ("seed", "seed"),
                ("step", "歩"),
                ("distance", "変位m"),
                ("parent", "親slot"),
                ("child", "子slot"),
                ("relocated", "再配置辺"),
                ("from_x", "親x"),
                ("from_y", "親y"),
                ("to_x", "子x"),
                ("to_y", "子y"),
                ("equal", "経路両端一致"),
            ],
        )
        if rows
        else "対応できた辺なし。"
    )


def write_report(output: Path):
    """測定済みCSVとJSONから①〜⑩の診断レポートを保存する。"""
    output = Path(output)
    summary = pd.read_csv(output / "summary.csv")
    steps = pd.read_csv(output / "step_diagnostics.csv")
    trajectory = pd.read_csv(output / "trajectory_comparison.csv")
    events = pd.read_csv(output / "anomaly_events.csv")
    intervals = pd.read_csv(output / "interval_comparison.csv")
    manifest = pd.read_csv(output / "particle_stages.csv")
    metadata = json.loads((output / "run_metadata.json").read_text())
    config = metadata["pf_config"]
    distances, pdr_angles, internal_angles = [], [], []
    particle_rows = 0
    for record in manifest.to_dict("records"):
        particles = pd.read_csv(
            output / record["path"],
            usecols=[
                "relocation_success",
                "realized_distance_m",
                "pdr_angle_diff_deg",
                "internal_angle_diff_deg",
            ],
        )
        particle_rows += len(particles)
        relocated = particles[particles.relocation_success]
        distances.extend(relocated.realized_distance_m.dropna())
        pdr_angles.extend(relocated.pdr_angle_diff_deg.dropna())
        internal_angles.extend(relocated.internal_angle_diff_deg.dropna())
    relocated_count = int(steps.relocation_count.sum())

    def fraction(column):
        """再配置成功粒子を母集団とする百分率を返す。"""
        return 100 * steps[column].sum() / max(relocated_count, 1)

    final_events = events[events.path == "final"]
    jumps = trajectory.sequence_step_m > 2
    sequence_jump_relocated = int(
        (jumps & trajectory.sequence_relocated.fillna(False)).sum()
    )
    important = pd.concat(
        [
            final_events.sort_values("distance_m", ascending=False)
            .drop_duplicates(["dataset", "seed", "step"])
            .head(12),
            final_events[final_events.event == "short_lt_0_5pdr"]
            .sort_values("score", ascending=False)
            .head(6),
        ]
    ).drop_duplicates(["dataset", "seed", "step"])
    run_rows = pd.DataFrame(
        [
            dict(
                dataset=r["dataset"],
                seed=r["seed"],
                steps=r["steps"],
                duration=r["accelerometer_duration_s"],
                first=r["first_step_time_s"],
                last=r["last_step_time_s"],
                y_sign=r["y_sign"],
            )
            for r in metadata["runs"]
        ]
    )
    report = f"""# 屋内PDR・地図制約付きPFの軌跡異常診断

## ① 実験概要

実行日時: {metadata["created_at"]}。HEAD: `{metadata["commit"]}`。{len(summary)}実行、延べ{len(steps)}歩、粒子段階{particle_rows:,}行を記録した。同一計測のseed違いは同じ歩行の確率的反復であり、独立計測ではない。

入力は `input/1019Experiment/PDR Sensor Logger (Research) 2026-10-19 18-52-40/` と `19-17-08/` のAccelerometer.csv・Gyroscope.csv。過去資料に約5mの歩間変位が保存されたため選定した。当初目視した異常と完全に同じ軌跡とは保証しない。

{table(run_rows, [("dataset", "計測"), ("seed", "seed"), ("steps", "歩数"), ("duration", "加速度計測時間s"), ("first", "最初の歩時刻s"), ("last", "最後の歩時刻s"), ("y_sign", "画像y符号")])}

地図は `input/Floormap_building14_5floor.png`。PF初期位置(0,0)m、画像起点(2972,245)px、0.01m/px。座標変換は `px=origin_x+x/scale; py=origin_y+y_sign*y/scale`。センサー平均値・y_signは各runのmetadataに保存した。

PDR設定: `{json.dumps(metadata["pdr_methods"], ensure_ascii=False)}`。

粒子数{int(summary.particles.iloc[0])}、再配置半径{config["PF_INVALID_RELOCATION_RADIUS_M"]}m・最大{config["PF_INVALID_RELOCATION_ATTEMPTS"]}試行、ESS閾値比{config["PF_RESAMPLE_ESS_RATIO"]}、歩幅倍率範囲[{config["PF_STRIDE_SCALE_MIN"]},{config["PF_STRIDE_SCALE_MAX"]}]。BLEと事後history候補は全実行で無効。

同じ関数を同じ引数で一度ずつ呼ぶ読取りラッパーを使用し、追加乱数を消費しない。通常実行と診断実行の完全一致は{int(summary.baseline_exact_equal.sum())}/{len(summary)}実行。比較はtrajectory、PDR歩幅・方位、particle結果全体（配列・stage・診断・経路比較、dtypeを含む）。入力SHA256不変={metadata.get("input_hashes_unchanged")}、src SHA256不変={metadata.get("source_unchanged")}。

方位列の重要な定義：既存のpdr_heading_rad/pdr_angle_diff_degと本レポートの「PF入力基準方位」は、PDR由来だがPF専用に連続化したctx.particle_headingであり、通常PDR軌跡の移動方位とは異なる。normal_pdr_*列がある場合は通常PDR方位との比較を追加測定した。

診断区分は暫定値：ジャンプ>2m、短い移動<PDR0.5倍、過大>PDR1.5倍、PF入力反対方位>90度、反転>135度。実際の誤りの確定基準ではない。PF入力基準方位差と経路線分間の反転を区別した。

```mermaid
flowchart LR
 A[加速度・ジャイロ] --> B[共有PDR歩幅・方位]
 B --> C[粒子の確率的移動提案]
 C --> D[移動先と線分の壁判定]
 D --> E[衝突粒子の円内再配置]
 E --> F[尤度と壁拘束後の重み]
 F --> G[低ESS全体再標本化 または 死亡枠補充]
 G --> H[粒子履歴と親対応]
 H --> I[currentクラスタ代表 と sequence祖先経路]
 I --> J[run全体で最終方式を選択]
```

過去資料は `agent/docs-archive/pf-history-prototype-20261022/README.md:42-59`。過去は5m再配置・歩幅倍率上限1.2・死亡枠のみ補充。過去manifestのhead=d9451f9は比較処理のスナップショット元でもあり、過去実行全体の完全ソース状態とは断定できない。今回と設定が異なるため、以下は歩番号で対応した**座標差の測定**であり、再現一致試験ではない。

{historical_comparison(trajectory)}

## ② 最も重要な発見

1. **再配置がPDRと異なる移動を直接作る。** 衝突提案{int(steps.collision_count.sum()):,}件、再配置成功{relocated_count:,}件。成功後実変位中央値{np.median(distances):.3f}m、最大{np.max(distances):.3f}m。PDR3倍超={fraction("relocated_ratio_gt_3_count"):.2f}%、内部方位差135度超={fraction("relocated_internal_gt135_count"):.2f}%。根拠はparticle_stagesのpredicted/relocated、realized_distance_m、distance_ratio、internal_angle_diff_deg。
2. **current代表点の接続は粒子の移動より大きなジャンプを作った。** current最大歩間変位は{summary.current_max_step_m.max():.3f}m、再配置半径{config["PF_INVALID_RELOCATION_RADIUS_M"]}mを超える。単一親の円内救済だけでは説明できず、時点ごとに異なる代表点を選び接続する段階で移動が増大した。currentは比較候補であり、実際の採用方式は③・⑦に区別した。
3. **sequenceは再配置を祖先経路へ引き継ぐ。** sequenceの>2m移動{int(jumps.sum())}件中、親対応を確認した再配置辺は{sequence_jump_relocated}件。採用辺の由来を直接対応付けた証拠である。ただし再配置を除く反実仮想は未実験。
4. **短い歩間移動と全体距離を区別する必要がある。** 最終>PDR1.5倍={int((final_events.event == "distance_gt_1_5pdr").sum())}件、<0.5倍={int((final_events.event == "short_lt_0_5pdr").sum())}件。最終/PDR比は③を参照。局所短縮があっても全体短縮とは限らず、全体長の増大も精度改善ではない。
5. **ESSと多様性は別指標。** 全体再標本化{int(steps.full_resampling.sum())}歩、死亡枠補充延べ{int(steps.partial_replacement_count.sum())}枠、更新後ESS最小{steps.ess_after_resampling.min():.2f}、10歩祖先最小{steps.ancestors_10.min():.0f}。重み一様化後も祖先の集中は残る。 **集団復旧の寄与も限定できる。** 今回の集団復旧は{int((steps.recovery_mode != "none").sum())}歩、failed_holdは{int((steps.recovery_mode == "failed_hold").sum())}歩。発動していない処理を今回の原因とはしない。

## ③ 現状の主要数値

{table(summary, [("dataset", "計測"), ("seed", "seed"), ("steps", "歩"), ("pdr_length_m", "PDR合計m"), ("final_length_m", "最終長m"), ("final_length_ratio", "最終/PDR"), ("final_max_step_m", "最大歩間m"), ("relocation_particle_fraction", "再配置粒子率"), ("ess_after_min", "再標本化後ESS最小"), ("ancestors10_min", "10歩祖先最小")])}

PDR合計は入力歩幅の総和。最終長は原点込み実座標差分の総和。`step_lengths.csv`を最終長の代用にしていない。

## ④ 異常が発生したステップ

{table(important, [("dataset", "計測"), ("seed", "seed"), ("step", "歩"), ("timestamp_s", "時刻s"), ("event", "暫定現象"), ("distance_m", "最終移動m"), ("pdr_step_m", "PDR歩幅m"), ("pdr_angle_diff_deg", "PF入力基準方位差度"), ("relocation_ratio", "再配置率"), ("current_mode", "current代表")])}

表は最大変位と最小歩幅比の上位。全イベントはanomaly_events.csv。1歩に複数ラベルが付くためイベント行数を異常歩数として合計しない。再配置率が高い歩との同時発生だけでは因果確定にならない。画像は⑩に掲載。

## ⑤ 壁衝突と再配置の分析

成功再配置を母集団とする。実変位は再配置後−前歩位置。再標本化後の変位はafter−after_parent_beforeで計算し、配列番号だけでは対応付けない。

|暫定歩幅比|成功再配置中の割合|
|---|---|
|0.5倍未満|{fraction("relocated_ratio_lt_0_5_count"):.2f}%|
|0.5〜1.5倍|{fraction("relocated_ratio_0_5_to_1_5_count"):.2f}%|
|1.5倍超〜3倍|{fraction("relocated_ratio_1_5_to_3_count"):.2f}%|
|3倍超|{fraction("relocated_ratio_gt_3_count"):.2f}%|

PDR歩幅≤{metadata["ratio_denominator_min_m"]}mでは比を欠損とする。欠損{int(summary.ratio_missing_count.sum())}件。再配置後PF入力基準方位差>90度={fraction("relocated_pdr_gt90_count"):.2f}%、>135度={fraction("relocated_pdr_gt135_count"):.2f}%。内部提案方位差>90度={fraction("relocated_internal_gt90_count"):.2f}%、>135度={fraction("relocated_internal_gt135_count"):.2f}%。PF入力基準方位差中央値{np.median(pdr_angles):.2f}度、内部方位差中央値{np.median(internal_angles):.2f}度。

前回実装調査で確認した「PDR提案と独立した円内救済、内部方位・歩幅を実変位へ合わせない」処理と整合する。地図で受理棄却された後の分布なので、成功粒子を無条件の円一様分布とは扱わない。壁拘束による法線方向の分布偏りの量は未測定。

## ⑥ 軌跡短縮の原因分析

{table(summary, [("dataset", "計測"), ("seed", "seed"), ("pdr_length_m", "PDR m"), ("parent_distance_weighted_sum_m", "親対応粒子累積m"), ("current_length_m", "current m"), ("sequence_length_m", "sequence m"), ("final_length_m", "最終m"), ("collective_recovery_steps", "集団復旧歩")])}

親対応粒子距離は各歩の更新後重みで平均した実変位の累積で、一本の粒子が辿った距離ではない。代表点の距離との差には方向相殺、再重み付け、クラスタ選択、代表点変更が混在する。粒子の距離減少をそのまま最終経路の短縮とみなさない。

問題区間は最大ジャンプ・最小歩幅比・最大PF入力基準方位差の前後2歩から自動抽出。重複区間は合算しない。

{table(intervals, [("dataset", "計測"), ("seed", "seed"), ("center_step", "中心歩"), ("start_step", "開始"), ("end_step", "終了"), ("pdr_length_m", "PDR m"), ("parent_weighted_distance_m", "親対応粒子m"), ("current_length_m", "current m"), ("sequence_length_m", "sequence m"), ("final_length_m", "最終m"), ("relocation_count", "再配置数")])}

前歩重みのまま再配置後位置を平均した変位総和は{steps.prior_weighted_mean_move_m.sum():.3f}m、posterior重みで平均した位置変位総和は{steps.posterior_weighted_mean_move_m.sum():.3f}m。これは最終currentのクラスタ代表ではない。`directional_cancellation_ratio`は前歩の同じ重みを用いた「平均変位ベクトルの長さ / 粒子実変位長の平均」。1未満なら方向相殺がある。最終クラスタ代表の因果寄与率ではない。

|仮説|判定|根拠と限界|
|---|---|---|
|1 円内再配置による方向喪失|確認済み|内部・PDR方位と実変位の差を直接測定。真の歩行に対する誤りとは未判定。|
|2 加重平均による短縮|可能性あり|粒子距離と平均・代表点の移動量を比較。最終経路への単独因果寄与は未分離。|
|3 壁拘束による分布偏り|確認済み（救済後のPF入力逆方向への偏りまで）|⑤の符号付き平均射影で条件別に測定。壁法線方向の偏り・最終異常への単独因果は未確定。|
|4 再標本化による多様性喪失|確認済み（系譜縮退まで）|親数・祖先数・位置重複を測定。最終異常の主因かは未判定。|
|5 PDRと地図補正の不整合|確認済み（内部状態と実変位まで）|再配置後変位と内部状態が一致しない。PDRと異なる候補自体を不具合としない。|

## ⑦ currentとsequenceの比較

{table(summary, [("dataset", "計測"), ("seed", "seed"), ("selected_mode", "採用"), ("current_jump_gt2m_count", "current >2m"), ("sequence_jump_gt2m_count", "sequence >2m"), ("current_unsupported_reversals", "current未支持反転"), ("sequence_unsupported_reversals", "sequence未支持反転"), ("current_wall_crossings", "current壁交差"), ("sequence_wall_crossings", "sequence壁交差"), ("sequence_relocated_steps", "sequence再配置辺"), ("current_mode_switches", "current代表切替")])}

current/sequence選択はrun全体で一つ。歩ごとに両方式を切り替えていない。sequence未支持反転がcurrentより少なければsequence、同数ならcurrent。current内部のクラスタ平均・粒子fallback切替はtrajectory_comparison.csvのcurrent_mode/current_source/current_mode_switchで追跡する。切替とジャンプの同時発生は08_selection_and_jump図で照合できるが、それだけを原因と断定しない。

再標本化による平均位置シフト最大{steps.resampling_mean_shift_m.max():.4f}m。sequenceは保存親を辿るため円内再配置も経路として残る。壁を横切らない経路でも歩間距離が適切とは限らない。

## ⑧ 改善候補

今回は改善コードや製品パラメータ調整を行っていない。以下は比較候補であり採用済みではない。

|優先|候補|期待・トレードオフ・構造への影響|
|---|---|---|
|1|再配置候補分布と内部状態の整合を比較|relocation/proposeでPDR中心と広域成分の混合、または生存粒子複製を比較。厳密歩幅強制を避け、広域復旧・方向多様性の低下を評価する。|
|2|経路選択の大変位と短縮を別評価|sequence_path/finalize周辺。反転減少のみでは距離を保証しない。正解分岐・通過点を先に確保する。|
|3|ESS・位置・系譜の併用評価|ESSは既に適用されている。resampling/evaluate_mapの比較では祖先数と重複も測る。追加ノイズは壁違反や拡散の代償がある。|
|4|歩幅尤度の情報源を確認|予測で使用したPDR歩幅を再度ガウス・シグモイド尤度として強く掛けると二重計上になり得る。独立距離・通過点観測との区別が必要。|
|5|祖先後処理と状態推定を比較|history候補はジャンプを減らしても短縮を促す。既存の方位ドリフト・歩幅倍率状態を利用し、位置だけの再接続と区別する。|

## ⑨ 未確認事項・制約

- 対象計測と現地図に対応する正解軌跡・真の区間距離は未確定。RMSE、終点誤差、真の逆走、PDR歩幅自体の過小評価を主張しない。
- BLE観測/reset/hybrid/retrofit分岐は未実施。今回の原因として扱わない。
- 再配置無効化や半径変更の介入は未実施。実際の採用辺の由来と、反実仮想による完全な因果分離を区別する。
- 同じseedでも過去とは設定・乱数消費の変化があり得る。過去座標との不一致を診断ラッパーの非干渉失敗と混同しない。
- 粒子IDは配列スロット。parent_particle_idで対応付ける。checkpoint等で履歴置換された場合はonline_matches_final_historyで区別し、再配置の由来を確定できない辺は欠損。
- 記録した重みは移動前、地図/観測後posterior、正規化posterior、再標本化後。全ての尤度処理の途中を個別分解した記録ではない。
- 衝突率とジャンプの相関は地図形状・重み・代表選択等の共通原因を含み得る。同期だけでは因果としない。

## ⑩ 出力データ一覧

|ファイル|用途・主要列|
|---|---|
|run_metadata.json|HEAD、設定、SHA256、座標系、計測時間、通常/診断完全一致範囲|
|summary.csv|計測×seedの経路長・ESS・再配置・系譜・選択方式|
|particle_stages.csv|分割manifest：dataset,seed,rows,path。粒子データ本体ではない|
|各run/particle_stages.csv.gz|before/predicted/relocated/after、親ID、衝突、再配置、内部状態、実変位、重み|
|step_diagnostics.csv|衝突/再配置率、距離比、方位差、ESS前後、親/祖先、位置分散、平均シフト|
|trajectory_comparison.csv|原点込みPDR/current/sequence/final座標と実変位、current代表、sequence再配置辺|
|anomaly_events.csv|ジャンプ・短縮・方位反対・線分反転の複数ラベル|
|interval_comparison.csv|自動抽出区間の各距離・再配置数。重複区間あり|
|各run/trajectory.csv|最終座標。原点込み、歩時刻はセンサー時刻、原点時刻は欠損|
|各run/existing_particle_diagnostics.csv|製品の既存診断構造体を保存|
|各run/figures/|seed 0の両計測と、最大current/最終変位を含む追加2runの地図重畳・距離・衝突・再配置・方位・多様性・異常拡大|
|figures/|全2計測×6seedの集計比較図。粒子詳細画像はseed 0の両計測と追加2run|

粒子CSVの同じ行iではbefore/predicted/relocatedは入力スロットi、afterは出力スロットi。afterの親はparent_particle_id、親座標はafter_parent_before_x/y。realized_distance_mは入力スロットの再配置後実変位、after_parent_distance_mは再標本化後の親対応実変位。posterior_weightは正規化前、normalized_posterior_weightは正規化後、after_weightは再標本化後。motion_stateとafter_motion_stateも区別する。

収集・通常実行との一致確認・派生集計・代表画像・レポートまでの再実行例（リポジトリルートで実行、出力先は未使用のディレクトリにする）:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/rikka-stage-mpl UV_CACHE_DIR=.uv-cache \
  uv run python agent/agent_diagnose_pf_trajectory_stages.py \
  --datasets 18-52-40 19-17-08 --seeds 0 1 2 10 42 100 \
  --output-dir /private/tmp/rikka-stage-reproduction-new
```

診断コードは `agent/agent_diagnose_pf_trajectory_stages.py`、描画は `agent/agent_plot_pf_trajectory_diagnosis.py`、レポート生成は `agent/agent_report_pf_trajectory_diagnosis.py`。出力先が既存なら収集は停止し、既存成果物を上書きしない。

レポートだけの再生成（診断実行・乱数消費なし）:

```sh
uv run python agent/agent_report_pf_trajectory_diagnosis.py --output-dir {output}
```
"""
    figures = sorted([*output.glob("*/figures/*.png"), *output.glob("figures/*.png")])
    evidence = lineage_evidence(output, manifest, trajectory)
    report = report.replace(
        "## ⑧ 改善候補",
        "各runのsequence最大2辺を、親粒子と子粒子の実座標へ対応付けた。"
        "経路両端一致は64bit CSV丸めを考慮した1e-10m許容。"
        "再配置辺=Trueかつ一致=Trueなら、その大変位は同じ親の円内救済辺を"
        "そのまま経路として用いたことを示す。\n\n" + evidence + "\n\n## ⑧ 改善候補",
    )
    report = report.replace(
        "## ⑦ currentとsequenceの比較",
        f"currentの最大歩間変位は全run最大{summary.current_max_step_m.max():.3f}m。"
        f"再配置半径{config['PF_INVALID_RELOCATION_RADIUS_M']}mを超える代表点接続が"
        "ある場合、単一親の円内移動量だけでは説明できず、時点ごとの代表点の"
        "選択・接続段階で増大した移動である。\n\n## ⑦ currentとsequenceの比較",
    )
    ratio_min = float(summary.final_length_ratio.min())
    ratio_max = float(summary.final_length_ratio.max())
    moving_rows = trajectory[trajectory.step > 0]
    local_ratios = moving_rows.final_step_m / moving_rows.pdr_step_m
    local_under_pdr = int((local_ratios < 1).sum())
    shortening = (
        "今回の全実行で最終長はPDR歩幅合計以上となり、全体短縮は再現しなかった。"
        if ratio_min >= 1
        else "最終長がPDR歩幅合計より短い実行があった。実行別の距離比で判断する。"
    )
    report = report.replace(
        "## ③ 現状の主要数値",
        f"**{shortening}** 最終/PDR比の範囲は{ratio_min:.4f}〜{ratio_max:.4f}。"
        f"1歩単位では{local_under_pdr}/{len(moving_rows)}歩でPDR歩幅未満、"
        f"最小比は{local_ratios.min():.4f}。局所の短い歩間移動と全体を分ける。\n\n"
        "## ③ 現状の主要数値",
    )
    examples = []
    for dataset, group in trajectory[trajectory.seed == 0].groupby("dataset"):
        current_max = group.loc[group.current_step_m.idxmax()]
        final_max = group.loc[group.final_step_m.idxmax()]
        run_name = f"{dataset}_seed0"
        examples.append(
            f"- {dataset} seed 0：current最大は歩{int(current_max.step)}、"
            f"{current_max.current_step_m:.3f}m、モード{current_max.current_mode}、"
            f"同歩の最終移動は{current_max.final_step_m:.3f}m。"
            f"[代表モードとジャンプ]({run_name}/figures/08_selection_and_jump.png)。"
            f"最終最大は歩{int(final_max.step)}、{final_max.final_step_m:.3f}m、"
            f"sequence再配置辺={final_max.sequence_relocated}。"
            f"[その前後の粒子段階]({run_name}/figures/07_particle_stages_step_{int(final_max.step):04d}.png)。"
            "根拠列はtrajectory_comparison.csvのcurrent_step_m/final_step_m/"
            "current_mode/sequence_relocated。"
        )
    if examples:
        report = report.replace(
            "## ③ 現状の主要数値",
            "具体的なseed 0の対応例（数値は全12実行。粒子詳細画像はseed 0の両計測と、最大current/最終変位を含む追加2run）:\n\n"
            + "\n".join(examples)
            + "\n\n## ③ 現状の主要数値",
        )
    if "relocated_normal_pdr_gt90_fraction" in summary:
        normal_table = table(
            summary,
            [
                ("dataset", "計測"),
                ("seed", "seed"),
                ("relocated_normal_pdr_gt90_fraction", "通常PDR差>90度率"),
                ("relocated_normal_pdr_gt135_fraction", "通常PDR差>135度率"),
                ("normal_pf_input_heading_diff_max_deg", "通常PDR/PF入力最大差度"),
            ],
        )
        report = report.replace(
            "## ⑥ 軌跡短縮の原因分析",
            "通常PDR軌跡の座標差分方位を基準にした再配置成功粒子の別集計を示す。"
            "上のPF入力基準方位差と混同しない。率は0〜1。\n\n"
            + normal_table
            + "\n\n## ⑥ 軌跡短縮の原因分析",
        )
    if "relocated_forward_projection_mean_m" in summary:
        projection_table = table(
            summary,
            [
                ("dataset", "計測"),
                ("seed", "seed"),
                ("relocated_forward_projection_mean_m", "成功救済の前方射影平均m"),
                ("unrelocated_forward_projection_mean_m", "非衝突の前方射影平均m"),
            ],
        )
        backward = int((summary.relocated_forward_projection_mean_m < 0).sum())
        report = report.replace(
            "## ⑥ 軌跡短縮の原因分析",
            "### 衝突後の救済と非衝突粒子の方向比較\n\n"
            "前歩からの実変位をPF入力基準方位の単位ベクトルへ射影した。"
            "正は前方、負は反対方向。成功救済粒子と衝突しなかった粒子を別の母集団として"
            "粒子件数で平均した値であり、重み付き平均でも壁法線への射影でもない。\n\n"
            + projection_table
            + f"\n\n救済後の平均射影が負となった実行は{backward}/{len(summary)}。"
            "この条件付き観測は、壁衝突後の受理救済がPF入力方向と逆向きに偏ったことを"
            "定量化する。非衝突粒子との比較は同じ粒子の反実仮想ではない。"
            "壁法線方向の偏り・真の歩行方向との誤差・最終異常への単独因果は未確定。"
            "対応列はsummaryおよびstep_diagnosticsのrelocated_forward_projection_mean_m/"
            "unrelocated_forward_projection_mean_m、粒子列pdr_forward_projection_m。"
            "\n\n## ⑥ 軌跡短縮の原因分析",
        )
        if backward == 0:
            report = report.replace(
                "確認済み（救済後のPF入力逆方向への偏りまで）",
                "可能性あり（今回、実行全体の平均逆向き射影は未確認）",
            )
    association_rows = []
    for (dataset, seed), group in steps.groupby(["dataset", "seed"]):
        threshold = float(group.relocation_ratio.quantile(0.75))
        upper = group.relocation_ratio >= threshold
        for label, selected in (("q75以上", upper), ("残り", ~upper)):
            subset = group[selected]
            association_rows.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    group=label,
                    q75=threshold,
                    steps=len(subset),
                    relocation_ratio_mean=subset.relocation_ratio.mean(),
                    final_jump_fraction=(subset.final_step_m > 2).mean(),
                    final_short_fraction=(
                        subset.final_step_m < 0.5 * subset.pdr_step_m
                    ).mean(),
                    final_opposed_pf_fraction=(
                        subset.final_pdr_angle_diff_deg > 90
                    ).mean(),
                )
            )
    association = pd.DataFrame(association_rows)
    association.to_csv(output / "relocation_association.csv", index=False)
    association_text = (
        "### 多数再配置と症状の同時発生\n\n"
        "各計測×seed内で再配置率の75%点(q75)以上の歩と残りを比較した。"
        "q75同率の歩も上位群へ含めるため、上位群は全歩の25%より多い場合がある。"
        "表の症状列は群内の歩数割合(0〜1)。最終PF差はPF入力基準方位との角度差であり、"
        "隣接経路線分間の反転ではない。\n\n"
        + table(
            association,
            [
                ("dataset", "計測"),
                ("seed", "seed"),
                ("group", "群"),
                ("q75", "再配置率q75"),
                ("steps", "歩数"),
                ("relocation_ratio_mean", "平均再配置率"),
                ("final_jump_fraction", "最終>2m率"),
                ("final_short_fraction", "最終<PDR0.5倍率"),
                ("final_opposed_pf_fraction", "最終PF差>90度率"),
            ],
        )
        + "\n\nこれは多数再配置と症状の**同時発生を調べる観測比較**。"
        "地図形状・歩行状態・重み等の交絡を除いておらず、因果や再配置数の"
        "単調な効果は示さない。sequenceジャンプ辺を実際の救済親子へ対応付ける"
        "由来の証拠とは役割が異なる。\n\n"
    )
    report = report.replace(
        "## ⑥ 軌跡短縮の原因分析", association_text + "## ⑥ 軌跡短縮の原因分析"
    )
    report = report.replace(
        "|interval_comparison.csv|",
        "|relocation_association.csv|計測×seed内q75以上/残り群の再配置率と症状の同時発生比較|\n"
        "|interval_comparison.csv|",
    )
    phase_path = output / "phase_mean_effects.csv"
    if phase_path.exists():
        phases = pd.read_csv(phase_path)
        phase_summary = (
            phases.groupby("phase", sort=False)
            .agg(
                instances=("step", "size"),
                shift_sum=("shift_m", "sum"),
                forward_sum=("pf_forward_shift_m", "sum"),
                negative_count=("pf_forward_shift_m", lambda v: int((v < -1e-9).sum())),
            )
            .reset_index()
        )
        phase_text = (
            "### 段階ごとの平均位置変化の台帳\n\n"
            "同じ歩の重み付き平均をbefore→predicted→relocated→posterior→afterの順に"
            "接続した。prediction/relocationは前歩重みを保持、weightingは位置を保持して"
            "重みだけを更新する。pf_forward_shift_mはPF入力基準方位への符号付き射影。"
            "負ならその段階の平均位置変化が基準方位に逆らう。逆向き歩数は丸め誤差を除くため射影<-1e-9mで数える。ベクトル差の恒等分解であり、"
            "最終クラスタ・単一祖先経路の因果寄与率ではない。\n\n"
            + table(
                phase_summary,
                [
                    ("phase", "段階"),
                    ("instances", "歩数"),
                    ("shift_sum", "平均シフト長合計m"),
                    ("forward_sum", "前方射影合計m"),
                    ("negative_count", "逆向き平均シフト歩"),
                ],
            )
            + "\n\n逆向き射影の大きい段階上位:\n\n"
            + table(
                phases.nsmallest(10, "pf_forward_shift_m"),
                [
                    ("dataset", "計測"),
                    ("seed", "seed"),
                    ("step", "歩"),
                    ("phase", "段階"),
                    ("shift_m", "シフトm"),
                    ("pf_forward_shift_m", "前方射影m"),
                ],
            )
        )
        report = report.replace(
            "## ⑦ currentとsequenceの比較",
            phase_text + "\n\n## ⑦ currentとsequenceの比較",
        )
        report = report.replace(
            "|interval_comparison.csv|",
            "|phase_mean_effects.csv|prediction/relocation/weighting/resamplingの重み付き平均位置差と符号付き射影|\n"
            "|validation_checks.csv|保存CSVの親座標・予測式・posterior重みの数値検査|\n"
            "|interval_comparison.csv|",
        )
    switches_path = output / "selection_switch_events.csv"
    if switches_path.exists():
        switches = pd.read_csv(switches_path)
        large = switches[switches.current_step_m > 2]
        switch_summary = pd.DataFrame(
            [
                dict(
                    switched=switched, count=int((large.mode_switch == switched).sum())
                )
                for switched in (False, True)
            ]
        )
        switch_text = (
            "### current代表モード切替と大変位\n\n"
            "current>2mの歩について、代表モード名が前歩から変わったかで集計。"
            "モード名が変わらなくても、クラスタ構成や重み・代表位置は変わり得る。"
            "したがってweighted_mean→weighted_meanのジャンプも、粒子の単一親移動とは"
            "区別する。current_sourceの同じ番号も永続個体IDではない。\n\n"
            + table(
                switch_summary,
                [("switched", "モード切替"), ("count", "current>2m件数")],
            )
            + "\n\ncurrent大変位上位10歩:\n\n"
            + table(
                large.nlargest(10, "current_step_m"),
                [
                    ("dataset", "計測"),
                    ("seed", "seed"),
                    ("step", "歩"),
                    ("previous_mode", "前モード"),
                    ("current_mode", "後モード"),
                    ("current_step_m", "current m"),
                    ("sequence_step_m", "sequence m"),
                    ("final_step_m", "最終m"),
                    ("selected_mode", "採用"),
                ],
            )
        )
        report = report.replace("## ⑧ 改善候補", switch_text + "\n\n## ⑧ 改善候補")
        report = report.replace(
            "|anomaly_events.csv|",
            "|selection_switch_events.csv|currentモード切替またはcurrent>2mの前後モード・source・各経路変位|\n"
            "|anomaly_events.csv|",
        )
    if "final_path_turn_gt135_count" in summary:
        report = report.replace(
            "## ⑤ 壁衝突と再配置の分析",
            "最終経路の反転とPF入力方位反対を別集計する。path_turnは隣接する最終実変位の"
            "角度が135度超で、センサー旋回支持を問わない。⑦の未支持反転数とは異なる。\n\n"
            + table(
                summary,
                [
                    ("dataset", "計測"),
                    ("seed", "seed"),
                    ("final_short_lt_0_5pdr_count", "最終<PDR0.5倍"),
                    ("final_opposed_pdr_gt90_count", "PF入力差>90度"),
                    ("final_opposed_pdr_gt135_count", "PF入力差>135度"),
                    ("final_path_turn_gt135_count", "隣接線分>135度"),
                ],
            )
            + "\n\n## ⑤ 壁衝突と再配置の分析",
        )
    report = report.replace(
        "## ⑧ 改善候補",
        "実装根拠は前回の [実装調査レポート](../pdr-pf-implementation-investigation-2026-10-22.md)"
        "と対応付けた。現HEADの主要箇所は `src/rikka/particle/lib/relocate.py:40` "
        "`relocate_invalid_particles`、`propose.py:128` `propose`、"
        "`evaluate_map.py:53` `_resample_or_keep`、`sequence_path.py:79` "
        "`_select_sequence_map_path`、`finalize.py:76` `finalize`。"
        "再配置・予測・重み・親履歴・代表点選択の各測定段階をこれらへ対応させる。\n\n"
        "## ⑧ 改善候補",
    )
    report += f"\n今回保存されたPNGは{len(figures)}枚。動画は生成していない。\n"
    for figure in figures:
        report += f"\n![{figure.parent.parent.name}: {figure.stem}]({figure.relative_to(output).as_posix()})\n"
    (output / "diagnostic_report.md").write_text(report)


def main():
    """専用出力ディレクトリを指定してレポートだけを再生成する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    write_report(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
