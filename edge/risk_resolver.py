import onnxruntime as ort
import numpy as np
from pathlib import Path
from .digital_twin import DigitalTwin

class RiskResolver:
    def __init__(self, onnx_path=None):
        if onnx_path is None:
            base = Path(__file__).resolve().parent.parent
            onnx_path = base / "artifacts" / "artifact_xgb_model.onnx"
        self.session = ort.InferenceSession(str(onnx_path))
        self.twin = DigitalTwin()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def resolve(self, raw_features: list, features_scaled: np.ndarray):
        raw_output = self.session.run([self.output_name], {self.input_name: features_scaled})[0]

        if raw_output.ndim == 1:
            risk_score = float(raw_output[0]) if len(raw_output) > 0 else 0.0
        elif raw_output.ndim == 2:
            if raw_output.shape[1] > 2:
                risk_score = float(raw_output[0][2])
            else:
                risk_score = float(raw_output[0][1])
        else:
            raise ValueError(f"Bentuk output ONNX tidak dikenali: {raw_output.shape}")

        twin_state = self.twin.update(risk_score)
        physics_estimates = self.twin.cross_check(raw_features)
        twin_state["physics"] = physics_estimates
        return risk_score, twin_state
