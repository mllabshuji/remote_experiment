import subprocess
import json
import datetime
import time
import os

REMOTE_DIR = "KS-NU_project:KS-NU_project"
REMOTE_FILE = f"{REMOTE_DIR}/analysis_datas.csv"

LOCAL_DIR = "."
DATA_DIR = f"{LOCAL_DIR}/_datas"
SCRIPT_DIR = f"{LOCAL_DIR}/_scripts"

CHECK_INTERVAL = 5  # 5秒ごと

last_update = None

while True:
    try:
        result = subprocess.run(
            ["rclone", "lsjson", REMOTE_FILE],
            capture_output=True,
            text=True,
            check=True
        )

        items = json.loads(result.stdout)
        if not items:
            print("ファイルが存在しません")
            time.sleep(CHECK_INTERVAL)
            continue

        # ファイルの最終更新日時を取得
        mod_time = datetime.datetime.fromisoformat(items[0]["ModTime"].replace("Z", "+00:00"))

        if last_update is None or mod_time > last_update:
            print(f"analysis_datas.csvの新しい更新を検知: {mod_time}, ダウンロード開始...")
            subprocess.run(
                ["rclone", "copy", REMOTE_FILE, DATA_DIR, "--update"],
                check=True
            )
            print("ダウンロード完了")

            print("BO開始")
            # bo_run.py を実行
            try:
                subprocess.run(
                    ["python", f"{SCRIPT_DIR}/bo_run.py"],
                    check=True
                )
                print("実験終了")

                
                # bo_run.py が正常終了したら parameters.csv を上書きアップロード
                if os.path.exists(f"{DATA_DIR}/parameters.csv"):
                    subprocess.run(
                        ["rclone", "copy", f"{DATA_DIR}/parameters.csv", REMOTE_DIR, "--update"],
                        check=True
                    )
                    print("parameters.csv を Box にアップロードしました")
                else:
                    print("parameters.csv が存在しません")


            except subprocess.CalledProcessError as e:
                print("bo_run.py 実行エラー:", e)
            print("BO終了")

            last_update = mod_time
        else:
            print("更新なし")

    except subprocess.CalledProcessError as e:
        print("rclone 実行エラー:", e.stderr)
    except json.JSONDecodeError as e:
        print("JSON デコードエラー:", e)

    time.sleep(CHECK_INTERVAL)
