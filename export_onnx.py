import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

import xgboost as xgb
import joblib

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

print("Memuat XGBoost model...")
model = xgb.XGBClassifier()
model.load_model(str(ARTIFACTS_DIR / "artifact_xgb_model.json"))

print("Memuat scaler...")
scaler = joblib.load(str(ARTIFACTS_DIR / "artifact_scaler.pkl"))
n_features = scaler.n_features_in_
print(f"Jumlah fitur: {n_features}")

try:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    output_path = ARTIFACTS_DIR / "artifact_xgb_model.onnx"
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model berhasil disimpan di {output_path}")
except ImportError:
    print("onnxmltools tidak tersedia, skip export ONNX")
except Exception as e:
    print(f"Gagal export ONNX: {e}")
