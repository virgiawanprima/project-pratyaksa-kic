from .preprocessor import EdgePreprocessor
from .risk_resolver import RiskResolver

class EdgeInference:
    def __init__(self):
        self.preprocessor = EdgePreprocessor()
        self.resolver = RiskResolver()

    def predict(self, sensor_data: dict):
        raw_features = [sensor_data.get(name, 0.0) for name in self.preprocessor.feature_names]
        feat_scaled = self.preprocessor.transform(sensor_data)
        risk, twin_state = self.resolver.resolve(raw_features, feat_scaled)
        return risk, twin_state