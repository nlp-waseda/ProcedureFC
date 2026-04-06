# ProcedureFC

LLMの手順型応答を自動でファクトチェックするフレームワークです。

<img width="600" alt="ProcedureFCの外観図" src="https://github.com/user-attachments/assets/1746cc58-005d-44b3-adb1-836b8d45bc90" />

## セットアップ

スクリプトを実行する前に、OpenAI APIキー、Google APIキー、および Google Custom Search Engine ID を環境変数として設定する必要があります。

**Mac / Linux の場合:**

```bash
export OPENAI_API_KEY={your_openai_api_key}
export ANTHROPIC_API_KEY={your_anthropic_api_key}
export GOOGLE_API_KEY={your_google_api_key}
export GOOGLE_CSE_ID={your_google_cse_id}
export HF_TOKEN={your_hugging_face_token}
```

**Windows の場合:**

PowerShell:

```pwsh
$env:OPENAI_API_KEY="{your_openai_api_key}"
$env:ANTHROPIC_API_KEY="{your_anthropic_api_key}"
$env:GOOGLE_API_KEY="{your_google_api_key}"
$env:GOOGLE_CSE_ID="{your_google_cse_id}"
$env:HF_TOKEN="{your_hugging_face_token}"
```

コマンドプロンプト:

```cmd
set OPENAI_API_KEY={your_openai_api_key}
set ANTHROPIC_API_KEY={your_anthropic_api_key}
set GOOGLE_API_KEY={your_google_api_key}
set GOOGLE_CSE_ID={your_google_cse_id}
set HF_TOKEN={your_hugging_face_token}
```

`{your_openai_api_key}`、`{your_anthropic_api_key}`、`{your_google_api_key}`、`{your_google_cse_id}`、`{your_hugging_face_token}` は、それぞれ実際のキー／IDに置き換えてください。

## インストール

このフレームワークを実行する際は、仮想環境の利用を推奨します。`uv` を使用すると、指定した Python バージョンで簡単に環境を作成できます。

### uv を使用する場合

**Windows（PowerShell）:**

```pwsh
uv venv --python=3.11
uv sync
```

**Mac / Linux:**

```bash
uv venv --python=3.11
uv sync
```

### venv を使用する場合

**Windows（PowerShell）:**

```pwsh
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Mac / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ベンチマーク実行

```bash
python Main.py <option>
```
\<option\>を変更することで、様々なケースのベンチマークを実行できます。optionの一覧は `.vscode/launch.json` を参照。
ベンチマークの実行結果は `results` に保存されます。
