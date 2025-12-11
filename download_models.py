import os
import json
import gdown

MODELS_DIR = "models"
CONFIG_FILE = "models_config.json"

os.makedirs(MODELS_DIR, exist_ok=True)

print("📥 Loading model configuration...")
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

for filename, file_id in config.items():
    output_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(output_path):
        print(f"✔ {filename} already exists — skipping download.")
        continue

    print(f"⬇ Downloading {filename} ...")
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        output_path,
        quiet=False
    )

print("\n✅ All model files downloaded successfully!")
