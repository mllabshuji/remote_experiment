# rclone と BOXを使った同期実験

## rclone導入方法

rclone導入方法.pdfを参照
rcloneのプロジェクト名とbox_reload_*.pyのREMOTE_DIRのプロジェクト名を一致させてください．
BOXフォルダに, REMOTE_DIRのフォルダ名と一致するフォルダを作成し, 必要なcsvファイルを入れてください．

## 実験方法

ローカルA: 最適化を行う方
```bash
python box_reload_NU.py
```

ローカルB: 解析を行う方
```bash
python box_reload_KS.py
```

