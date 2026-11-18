# アプリ向け共通解析API

`rikka.analysis_api` は解析方式の公開窓口です。CLI、`run_pdr()`、
`run_particle()` は従来どおり利用できます。研究用関数の統一や変更は不要です。
HTTP、S3、実行DB、通知は呼び出し元の責務です。

```python
from rikka.analysis_api import execute_method, list_methods, render_artifact
from rikka.analysis_contracts import ExecutionOptions

methods = list_methods()
result = execute_method(
    "pdr",
    {"acce": acce_bytes, "gyro": gyro_bytes, "metadata": metadata_bytes},
    {"initial_direction": 90.0, "user_height_m": 1.68},
    options=ExecutionOptions(ground_truth_csv=truth_bytes),
)
trajectory = result["trajectory"]
```

## 契約

`list_methods()` は独立した `MethodDefinition` の組を返します。
`method_id`、表示名、`definition_version`、入力スロット、パラメータスキーマ、
出力スロット、成果物の名前・形式・Content-Type・遅延生成の有無を公開します。
入力スロットは `slot_id`、`accepted_contracts`、省略時Trueの`required`を持ちます。
asset契約は `kind=asset`、`data_type`、`schema_version`、`format`、
内部値契約は `kind=internal_value`、`value_type` を持ちます。

解析の入力は `{slot_id: bytes}`、パラメータはJSON互換辞書です。
`ExecutionOptions` は成果物名の組と任意の正解CSVを受け取ります。
通信先、アップロードURL、認証情報は渡しません。
`validate_parameters()` は共通契約の部分集合（objectのproperties/required/
additionalProperties、typeとnull許容、enum、minimum/maximum/exclusiveMinimum）を
検証します。配列要素や入れ子の制約、oneOf等は対象外です。
座標組の整合性、地図内範囲、方式固有のデータ検証は実行処理が行います。
修正が必要な入力エラーは `AnalysisInputError` を送出します。

出力は `{output_slot_id: value}` です。既存4方式の `trajectory` は従来どおり、
`schema_version=1`、`coordinate_system`（pixel / local_meter）、`component_id`、
`points`（step_index / timestamp_s / x / y）、`step_count`、
`landmark_correction_count` を持ちます。初期点の時刻はNoneです。
評価は `evaluation` にstatusとsummaryまたはerrorを返します。
評価失敗でも軌跡を返し、CSV・JSONの成果物形式も維持します。

即時成果物は `trajectory["_artifact_payloads"]` に
`{name: {"body": bytes}}` として格納します。
遅延成果物は `trajectory["_deferred_artifacts"]` に
`{name: DeferredArtifact}` として格納します。呼び出し元はこれらを通常のJSON出力から
取り除き、軌跡を先に公開した後で `artifact.render()` を実行します。
`render()` も `{name: {"body": bytes}}` を返します。
動画のMP4フォールバック名は `particle_animation_mp4` を維持します。

## 動画の保存と再生成

`artifact.serialize()` は方式・契約・パッケージのバージョン、素材形式の識別子を
含む不透明なbytesを返します。DB等に保存し、別プロセスでも
`render_artifact(source)` に渡して再生成できます。素材に粒子、地図画像、
BLE描画情報を含むため、一時ファイルや元の解析実行は不要です。
従来Nozomiが保存したヘッダーなしNPZも復元します（pickleは使いません）。
未対応の素材バージョンは拒否します。

素材形式の識別子は `DeferredArtifact.source_kind` に定義します。
同じ成果物名でも異なる保存形式を使用でき、`ARTIFACT_RENDERERS` へ明示登録します。
保存済み素材が残る間は古い識別子の復元処理を維持してください。
新形式を含む素材を作成後に旧版へロールバックすると、旧版は復元できません。

## 新方式の追加

1. Rikka内にアルゴリズムまたは公開用アダプターを実装します。
2. `MethodDefinition` でID・入力・パラメータ・出力・成果物・契約バージョンを定義します。
3. `analysis_api.METHODS` に `Method(definition, executor)` を明示登録します。
   executorは `(inputs, parameters, options)` を受け取り共通形式を返します。
   既存4方式は共有アダプターに構成を明示して登録し、ID文字列を解析しません。
4. 新しい遅延保存形式が必要なら、`source_kind` と復元処理も登録します。
5. 共通APIの検証、既存方式の回帰、保存素材の互換テストを実施します。
6. Rikkaを公開し、NozomiのGit固定コミットとlockを更新して再起動します。

共通契約内の方式追加ではNozomiのカタログやexecutor変更は不要です。
Nozomiの入力契約にない新data_type、新しい取得元、任意の処理グラフ、
軌跡以外を主出力とするアップロード、新しい通知・再試行HTTP契約を必要とする場合は、
Nozomi（必要に応じてKaede等）の対応も必要です。
既存の動画再試行HTTPパスとアップロード先の命名はNozomiの互換契約として残ります。

`legacy_analysis.analyze_to_csv()` は旧 `/analyze` 用の互換窓口です。
センサー正規化、既存設定の既定値、旧CSV列順、最初の歩を0秒とする時刻、
座標変換を担当し、実行自体は同じ `execute_method("pdr", ...)` を使います。
