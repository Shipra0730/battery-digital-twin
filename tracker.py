import os
import torch
import mlflow
import mlflow.pytorch
from typing import Any, Dict

class MLOpsModelTracker:
    """
    MLOps orchestration class handling:
    1. MLflow Experiment and parameter/metric logging.
    2. MLflow Model Registry integration.
    3. ONNX model export for production deployment.
    """
    def __init__(self, experiment_name: str = "Battery_PINN_Digital_Twin"):
        self.experiment_name = experiment_name
        # Set experimental registry in MLflow
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"MLflow connection warning: {e}. Running local tracking.")

    def log_training_run(self, 
                         hyperparams: Dict[str, Any], 
                         metrics: Dict[str, float], 
                         model: Any, 
                         run_name: str = "pinn_train_run"):
        """
        Logs hyperparameters, validation metrics, and PyTorch model artifact to MLflow.
        """
        try:
            with mlflow.start_run(run_name=run_name):
                # Log hyperparameters
                mlflow.log_params(hyperparams)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log PyTorch Model
                mlflow.pytorch.log_model(model, "model")
                print(f"Successfully logged run '{run_name}' to MLflow.")
        except Exception as e:
            print(f"Failed to log run to MLflow: {e}")
            
    def export_to_onnx(self, model: torch.nn.Module, output_path: str = "models/pinn_model.onnx") -> str:
        """
        Exports PyTorch model to ONNX format for sub-millisecond production inference.
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set to evaluation mode
        model.eval()
        
        # Create dummy input that matches feature dimension:
        # [cycle, current, voltage, temperature, internal_resistance, ica_peak_value, ica_peak_voltage]
        dummy_input = torch.randn(1, 7, dtype=torch.float32)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["battery_features"],
                output_names=["soh", "capacity", "resistance", "rul", "sei_thickness"],
                dynamic_axes={
                    "battery_features": {0: "batch_size"},
                    "soh": {0: "batch_size"},
                    "capacity": {0: "batch_size"},
                    "resistance": {0: "batch_size"},
                    "rul": {0: "batch_size"},
                    "sei_thickness": {0: "batch_size"}
                }
            )
            print(f"Successfully exported PINN model to ONNX format at {output_path}")
            return output_path
        except Exception as e:
            print(f"Failed to export model to ONNX: {e}")
            raise
