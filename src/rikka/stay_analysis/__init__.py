"""滞在分析の公開pipelineと実装領域。

役割:
    PDR・particleと同列の滞在分析機能を独立したパッケージとして提供する。
依存元:
    公開処理はpipeline、結果形式はmodels、内部実装はlib配下に配置する。
利用先:
    Nozomiなどの呼び出し側がpipelineを通してtrajectoryを分析する。
処理フロー:
    package初期化時の処理は持たず、利用側がpipelineの必要な処理を呼び出す。
"""
