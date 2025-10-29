# model_manager.py
import os
import torch
import joblib
from model.factory import build_model
from model.predict_model import PredictModel

class ModelManager:
    def __init__(self, device=None):
        """
        Model Manager: Handles model creation and loading.
        Specifically handles models with a pre-trained backbone (like ESM/PepBERT)
        and a trainable classifier head.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self, name, task, max_len, device=None, random_state=42, **kwargs):
        """
        Creates a model instance based on the name.
        For ESM/PepBERT, this returns the backbone model.
        """
        model = build_model(
            name=name,
            task=task,
            max_len=max_len,
            device=device,
            random_state=random_state,
            **kwargs
        )

        # Move the model to the device
        if hasattr(model, 'to'):
            target_device = device or self.device
            model.to(target_device)

        return model

    def load_model(self, path, name, task, max_len, device=None, random_state=42):
        model = self.create_model(
            name=name,
            task=task,
            max_len=max_len,
            device=device,
            random_state=random_state
        )

        if name in ['transformer', 'lstm']:
            # Saved state is the entire model's state_dict
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            return model

        elif name in ['esm', 'pepbert']:
            # Saved state is the classifier head's state_dict
            # Initialize the classifier head
            predicts_model = PredictModel(hidden_size=model.hidden_size).to(self.device)
            checkpoint = torch.load(path, map_location=self.device)
            predicts_model.load_state_dict(checkpoint)
            predicts_model.eval()
            return predicts_model

        else:
            # Traditional model
            model = joblib.load(path)
            return model
        
    def get_best_model_path(self, output_dir, model_name, feature_type, task, data_name, random_state=111):
        # Get the specific address where the model is saved
        pickle_models = ['rf', 'svm', 'xgb']  # Define file extensions based on model types
        description = feature_type
    
        if model_name in pickle_models:
            ext = '.pkl'
        else:
            ext = '.pt'
        
        filename = f"BEST_{model_name}_{description}_{task}_{data_name}_seed{random_state}{ext}"
        return os.path.join(output_dir, filename)

def model_type_identify(model):
    if model in ['rf', 'svm', 'xgb']:
        model_type = 'ml'
    elif model in ['lstm', 'transformer']:
        model_type = 'dl'
    else:
        model_type = 'll'
    return model_type
