# backend.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import logging
from flask import Flask, request, jsonify
from waitress import serve
from functools import wraps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='sum_over_batch_size', name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else tf.ones(3, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.cast(tf.one_hot(y_true, depth=num_classes), tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1 - y_pred, self.gamma) * self.class_weights
        return tf.reduce_mean(weight * ce)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'class_weights': self.class_weights.numpy().tolist() if self.class_weights is not None else None,
            'reduction': self.reduction,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class HealthPredictionBackend:
    def __init__(self, ctg_model_path, maternal_model_path, rf_ctg_path, xgb_ctg_path, rf_maternal_path, xgb_maternal_path, scaler_ctg_path, scaler_maternal_path, weights_ctg_path, weights_maternal_path, ctg_data_path, maternal_data_path):
        self.ctg_features = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Variance', 'Tendency']
        self.maternal_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'BP_Ratio']
        self.class_names_ctg = ['Normal', 'Suspect', 'Pathological']
        self.class_names_maternal = ['Low Risk', 'Mid Risk', 'High Risk']

        try:
            self.ctg_model = tf.keras.models.load_model(ctg_model_path, custom_objects={'FocalLoss': FocalLoss})
            self.maternal_model = tf.keras.models.load_model(maternal_model_path, custom_objects={'FocalLoss': FocalLoss})
            self.rf_ctg = joblib.load(rf_ctg_path)
            self.xgb_ctg = joblib.load(xgb_ctg_path)
            self.rf_maternal = joblib.load(rf_maternal_path)
            self.xgb_maternal = joblib.load(xgb_maternal_path)
            self.scaler_ctg = joblib.load(scaler_ctg_path)
            self.scaler_maternal = joblib.load(scaler_maternal_path)
            self.weights_ctg = joblib.load(weights_ctg_path)
            self.weights_maternal = joblib.load(weights_maternal_path)

            if None in (self.ctg_model, self.maternal_model, self.rf_ctg, self.xgb_ctg, self.rf_maternal, self.xgb_maternal, self.scaler_ctg, self.scaler_maternal):
                raise RuntimeError("Failed to load one or more required resources")

            self._fit_scalers(ctg_data_path, maternal_data_path)
            logger.info("Backend initialized successfully.")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _fit_scalers(self, ctg_data_path, maternal_data_path):
        try:
            ctg_data = pd.read_excel(ctg_data_path, sheet_name='Raw Data', usecols=self.ctg_features)
            ctg_data = ctg_data[self.ctg_features].apply(pd.to_numeric, errors='coerce').dropna()
            maternal_data = pd.read_csv(maternal_data_path, usecols=self.maternal_features[:-1])
            maternal_data = maternal_data.apply(pd.to_numeric, errors='coerce').dropna()
            maternal_data['BodyTemp'] = (maternal_data['BodyTemp'] - 32) * 5 / 9
            maternal_data['BP_Ratio'] = maternal_data['SystolicBP'] / maternal_data['DiastolicBP'].replace(0, np.nan)
            maternal_data['BP_Ratio'] = maternal_data['BP_Ratio'].fillna(0)
            logger.info("Scaler validation completed with training data.")
        except Exception as e:
            logger.error(f"Error validating scalers: {str(e)}")
            raise

    def preprocess_input(self, data, model_type='ctg'):
        try:
            if model_type == 'ctg':
                features = self.ctg_features
                scaler = self.scaler_ctg
            else:
                features = self.maternal_features
                scaler = self.scaler_maternal

            df = pd.DataFrame([data], columns=features)
            df = df.apply(pd.to_numeric, errors='coerce')
            if model_type == 'ctg':
                df['AC'] = df['AC'] / 1200
                df['FM'] = df['FM'] / 1200
                df['DL'] = df['DL'] / 1200
                df['DS'] = df['DS'] / 1200
                df['DP'] = df['DP'] / 1200
                df['UC'] = df['UC'] / 600

                df['LB'] = np.clip(df['LB'], 100, 180)
                df['AC'] = np.clip(df['AC'], 0.0, 10/1200)
                df['FM'] = np.clip(df['FM'], 0.0, 100/1200)
                df['UC'] = np.clip(df['UC'], 0.0, 15/600)
                df['DL'] = np.clip(df['DL'], 0.0, 20/1200)
                df['DS'] = np.clip(df['DS'], 0.0, 5/1200)
                df['DP'] = np.clip(df['DP'], 0.0, 5/1200)
                df['ASTV'] = np.clip(df['ASTV'], 0.0, 100.0)
                df['MSTV'] = np.clip(df['MSTV'], 0.0, 10.0)
                df['ALTV'] = np.clip(df['ALTV'], 0.0, 100.0)
                df['MLTV'] = np.clip(df['MLTV'], 0.0, 60.0)
                df['Width'] = np.clip(df['Width'], 0, 200)
                df['Variance'] = np.clip(df['Variance'], 0, 300)
                df['Tendency'] = np.clip(df['Tendency'], -1, 1)
            else:
                df['Age'] = np.clip(df['Age'], 10, 70)
                df['SystolicBP'] = np.clip(df['SystolicBP'], 70, 180)
                df['DiastolicBP'] = np.clip(df['DiastolicBP'], 40, 120)
                df['BS'] = np.clip(df['BS'], 4.0, 20.0)
                df['BodyTemp'] = np.clip(df['BodyTemp'], 36.0, 41.0)
                df['HeartRate'] = np.clip(df['HeartRate'], 40, 120)
                df['BP_Ratio'] = df['SystolicBP'] / df['DiastolicBP']
                df['BP_Ratio'] = np.clip(df['BP_Ratio'], 0.5, 4.0)

            df = df.fillna(0)
            X = scaler.transform(df.values)
            X = np.expand_dims(X, axis=2)
            return X
        except Exception as e:
            logger.error(f"Error preprocessing input for {model_type}: {str(e)}")
            raise

    def predict(self, data, model_type='ctg'):
        try:
            X = self.preprocess_input(data, model_type)
            if model_type == 'ctg':
                dl_model = self.ctg_model
                rf_model = self.rf_ctg
                xgb_model = self.xgb_ctg
                class_names = self.class_names_ctg
                dl_weight, rf_weight, xgb_weight = self.weights_ctg
            else:
                dl_model = self.maternal_model
                rf_model = self.rf_maternal
                xgb_model = self.xgb_maternal
                class_names = self.class_names_maternal
                dl_weight, rf_weight, xgb_weight = self.weights_maternal

            dl_preds_proba = dl_model.predict(X, verbose=0)
            X_2d = X.squeeze(axis=2)
            rf_preds_proba = rf_model.predict_proba(X_2d)
            xgb_preds_proba = xgb_model.predict_proba(X_2d)

            ensemble_preds = dl_weight * dl_preds_proba + rf_weight * rf_preds_proba + xgb_weight * xgb_preds_proba
            prediction = np.argmax(ensemble_preds, axis=1)[0]

            return class_names[prediction], ensemble_preds[0].tolist()
        except Exception as e:
            logger.error(f"Prediction error for {model_type}: {str(e)}")
            raise

backend = HealthPredictionBackend(
    ctg_model_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\ctg_model.keras',
    maternal_model_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\maternal_model.keras',
    rf_ctg_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\rf_ctg_model.pkl',
    xgb_ctg_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\xgb_ctg_model.pkl',
    rf_maternal_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\rf_maternal_model.pkl',
    xgb_maternal_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\xgb_maternal_model.pkl',
    scaler_ctg_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\scaler_ctg.pkl',
    scaler_maternal_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\scaler_maternal.pkl',
    weights_ctg_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\weights_ctg.pkl',
    weights_maternal_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\weights_maternal.pkl',
    ctg_data_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\CTG Dataset.xls',
    maternal_data_path=r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\Maternal Health Risk Data Set.csv'
)

CTG_FEATURE_COUNT = 14
MATERNAL_FEATURE_COUNT = 7

def validate_features(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            required_fields = ['features', 'dataset']
            if not all(field in data for field in required_fields):
                return jsonify({"error": "Missing required fields"}), 400

            features = data['features']
            dataset = data['dataset']

            if not isinstance(features, list):
                return jsonify({"error": "Features must be a list"}), 400

            expected_count = CTG_FEATURE_COUNT if dataset == "CTG (Fetal Health)" else MATERNAL_FEATURE_COUNT
            if len(features) != expected_count:
                return jsonify({
                    "error": f"Expected {expected_count} features for {dataset}, got {len(features)}"
                }), 400

            if not all(isinstance(x, (int, float)) for x in features):
                return jsonify({"error": "All features must be numeric"}), 400

            valid_datasets = ["CTG (Fetal Health)", "Maternal Health"]
            if dataset not in valid_datasets:
                return jsonify({"error": f"Invalid dataset. Must be one of {valid_datasets}"}), 400

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": "Invalid request format"}), 400

        return func(*args, **kwargs)
    return wrapper

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": all(x is not None for x in [backend.ctg_model, backend.maternal_model, backend.rf_ctg, backend.xgb_ctg, backend.rf_maternal, backend.xgb_maternal, backend.scaler_ctg, backend.scaler_maternal])
    })

@app.route('/predict', methods=['POST'])
@validate_features
def predict():
    try:
        data = request.get_json()
        features = data['features']
        dataset = data['dataset']

        if dataset == "CTG (Fetal Health)":
            input_data = dict(zip(backend.ctg_features, features))
            model_type = 'ctg'
        else:
            input_data = dict(zip(backend.maternal_features, features))
            model_type = 'maternal'

        class_name, probabilities = backend.predict(input_data, model_type)
        response = {
            "prediction": int(np.argmax(probabilities)),
            "class": class_name,
            "probabilities": probabilities,
            "features": features,
            "processing_time": "N/A"
        }

        logger.info(f"Successful prediction for {dataset}: {class_name}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Error making prediction"}), 500

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    serve(
        app,
        host='0.0.0.0',
        port=5000,
        threads=16,
        max_request_body_size=10485760,
        connection_limit=2000,
        cleanup_interval=60,
        channel_timeout=60
    )