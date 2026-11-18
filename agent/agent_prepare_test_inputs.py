"""既存回帰テストの元入力をGit履歴から準備する。

役割: 現在は追跡対象外のセンサーCSVを、保存済みblobから再現する。
入力: このリポジトリのGitオブジェクト。ネットワークや製品設定には依存しない。
出力: input/sensor_data/natsuki配下の不足CSVのみ。既存ファイルは上書きしない。
利用先: goldenおよびランドマークの既存pytestを実行する開発者・CI。
処理フロー: 不足を確認し、固定blobの内容を検証して排他的に書き込む。
"""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "10steps_stride_length_check_1m-1": (
        "8e4aff5b1887e9e284ae9c209fe33f759bbb024d",
        "0bd5025c5106a5ef6dccc798a76d8d734a4a2014",
    ),
    "10steps_stride_length_check_50cm-1": (
        "23c7c94c659d4239ed233d7e509061afad39bc9c",
        "93fde25d95bd0292bc6d0aae63aa2ebfb45a0cbc",
    ),
    "1turn_rightsidestep_3turn_leftsidestep": (
        "7b0bd16a258814630eafc01e64afeead00d0836e",
        "99baabf6fe017fa1b5871f254666b388003a9c57",
    ),
}


def main() -> None:
    for dataset, blobs in SOURCES.items():
        for filename, blob in zip(
            ("Accelerometer.csv", "Gyroscope.csv"), blobs, strict=True
        ):
            target = ROOT / "input/sensor_data/natsuki" / dataset / filename
            if target.exists():
                print(f"既存ファイルを維持: {target.relative_to(ROOT)}")
                continue
            source = subprocess.run(
                ["git", "cat-file", "blob", blob],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            header = f"blob {len(source)}\0".encode()
            if hashlib.sha1(header + source).hexdigest() != blob:
                raise ValueError(f"Git blobの検証に失敗: {blob}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(source)
            print(f"復元: {target.relative_to(ROOT)} ({blob})")


if __name__ == "__main__":
    main()
