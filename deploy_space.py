"""把整個專案上傳到 Hugging Face Space（由 GitHub Actions 每日更新後呼叫）。"""
import os
import sys

from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN', '').strip()
space = os.environ.get('HF_SPACE', '').strip()      # 格式：使用者名稱/space名稱
if not token or not space:
    print('尚未設定 HF_TOKEN 或 HF_SPACE，略過上傳到 Hugging Face。')
    sys.exit(0)

api = HfApi(token=token)
api.create_repo(space, repo_type='space', space_sdk='gradio', private=True, exist_ok=True)
api.upload_folder(folder_path='.', repo_id=space, repo_type='space',
                  ignore_patterns=['.git/*', '.github/*', '__pycache__/*', '*.pyc', 'SETUP.md'],
                  commit_message='每日資料更新')
print(f'已上傳到 https://huggingface.co/spaces/{space}')
