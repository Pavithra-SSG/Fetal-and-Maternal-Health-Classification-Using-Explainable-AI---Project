# main.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, MultiHeadAttention, LayerNormalization, Flatten, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
from sklearn.mixture import GaussianMixture

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_MKL'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
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

class HealthPredictionPipeline:
    def __init__(self):
        self.ctg_features = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Variance', 'Tendency']
        self.ctg_target = 'NSP'
        self.maternal_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        self.maternal_target = 'RiskLevel'
        self.scaler_ctg = MinMaxScaler()
        self.scaler_maternal = MinMaxScaler()
        self.class_names_ctg = ['Normal', 'Suspect', 'Pathological']
        self.class_names_maternal = ['Low Risk', 'Mid Risk', 'High Risk']

    def load_data(self, ctg_path, maternal_path):
        if not os.path.exists(ctg_path) or not os.path.exists(maternal_path):
            logger.error(f"Data file not found: CTG={ctg_path}, Maternal={maternal_path}")
            return False
        try:
            self.ctg_data = pd.read_excel(ctg_path, sheet_name='Raw Data', usecols=self.ctg_features + [self.ctg_target])
            self.maternal_data = pd.read_csv(maternal_path, usecols=self.maternal_features + [self.maternal_target])
            logger.info(f"CTG data shape: {self.ctg_data.shape}, Maternal data shape: {self.maternal_data.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def preprocess_data(self):
        try:
            # CTG Preprocessing
            self.ctg_data[self.ctg_features] = self.ctg_data[self.ctg_features].apply(pd.to_numeric, errors='coerce')
            self.ctg_data[self.ctg_target] = pd.to_numeric(self.ctg_data[self.ctg_target], errors='coerce') - 1
            self.ctg_data = self.ctg_data.dropna()
            X_ctg = self.ctg_data[self.ctg_features].values.astype(float)
            y_ctg = self.ctg_data[self.ctg_target].values.astype(int)

            # Maternal Preprocessing
            self.maternal_data[self.maternal_features] = self.maternal_data[self.maternal_features].apply(pd.to_numeric, errors='coerce')
            self.maternal_data[self.maternal_target] = self.maternal_data[self.maternal_target].map({'low risk': 0, 'mid risk': 1, 'high risk': 2})
            self.maternal_data = self.maternal_data[self.maternal_data['DiastolicBP'] > 0]
            self.maternal_data['BodyTemp'] = (self.maternal_data['BodyTemp'] - 32) * 5 / 9
            self.maternal_data['BP_Ratio'] = self.maternal_data['SystolicBP'] / self.maternal_data['DiastolicBP']
            self.maternal_data = self.maternal_data.dropna()
            self.maternal_features.append('BP_Ratio')
            X_maternal = self.maternal_data[self.maternal_features].values.astype(float)
            y_maternal = self.maternal_data[self.maternal_target].values.astype(int)

            # Clip features
            for i, feature in enumerate(self.ctg_features):
                X_ctg[:, i] = np.clip(X_ctg[:, i], np.percentile(X_ctg[:, i], 0.1), np.percentile(X_ctg[:, i], 99.9))
            for i, feature in enumerate(self.maternal_features):
                X_maternal[:, i] = np.clip(X_maternal[:, i], np.percentile(X_maternal[:, i], 0.1), np.percentile(X_maternal[:, i], 99.9))

            self.scaler_ctg.fit(X_ctg)
            self.scaler_maternal.fit(X_maternal)

            sm = SMOTE(random_state=42, k_neighbors=5)
            X_ctg, y_ctg = sm.fit_resample(X_ctg, y_ctg)
            X_maternal, y_maternal = sm.fit_resample(X_maternal, y_maternal)

            # Adjusted augmentation for CTG to prevent overfitting
            X_ctg_aug = X_ctg + np.random.normal(0, 0.05, X_ctg.shape)  # Increased noise from 0.01 to 0.05 for more variability
            gmm = GaussianMixture(n_components=3, random_state=42).fit(X_maternal)
            X_maternal_gmm, y_maternal_gmm = gmm.sample(n_samples=len(X_maternal))
            probs = gmm.score_samples(X_maternal_gmm)
            mask = probs > np.percentile(probs, 25)
            X_maternal_gmm = X_maternal_gmm[mask][:len(X_maternal)] + np.random.normal(0, 0.0005, X_maternal_gmm[mask][:len(X_maternal)].shape)
            y_maternal_gmm = np.clip(np.round(y_maternal_gmm[mask][:len(X_maternal)]), 0, 2).astype(int)

            X_ctg = np.vstack([X_ctg, X_ctg_aug])
            y_ctg = np.hstack([y_ctg, y_ctg])
            X_maternal = np.vstack([X_maternal, X_maternal_gmm])
            y_maternal = np.hstack([y_maternal, y_maternal_gmm])

            self.X_ctg, self.y_ctg = X_ctg, y_ctg
            self.X_maternal, self.y_maternal = X_maternal, y_maternal
            logger.info(f"CTG: {self.X_ctg.shape}, Maternal: {self.X_maternal.shape}")
            return True
        except Exception as e:
            logger.error(f"Error preprocessing: {e}")
            return False

    def create_model(self, input_shape, num_classes, model_type='ctg'):
        inputs = Input(shape=input_shape)
        if model_type == 'ctg':
            # Adjusted CTG model to reduce overfitting (targeting ~97% accuracy)
            x = Conv1D(12, 3, activation='relu', padding='same')(inputs)  # Reduced from 16 to 12 filters
            x = MaxPooling1D(2)(x)
            x = Conv1D(6, 3, activation='relu', padding='same')(x)  # Reduced from 8 to 6 filters
            x = MaxPooling1D(2)(x)
            lstm = Bidirectional(LSTM(6, return_sequences=True, recurrent_dropout=0.6))(x)  # Reduced from 8 to 6 units, increased dropout to 0.6
            attn = MultiHeadAttention(num_heads=2, key_dim=3)(lstm, lstm)  # Reduced key_dim from 4 to 3
            x = Add()([lstm, attn])
            x = LayerNormalization()(x)
            x = Flatten()(x)
            x = Dense(12, activation='relu', kernel_regularizer=l2(0.07))(x)  # Reduced from 16 to 12 units, increased L2 from 0.01 to 0.07
            x = Dropout(0.7)(x)  # Increased dropout from 0.5 to 0.7
            x = Dense(6, activation='relu', kernel_regularizer=l2(0.07))(x)  # Reduced from 8 to 6 units
            x = Dropout(0.6)(x)  # Increased dropout from 0.4 to 0.6
        else:  # Maternal (unchanged)
            x = Conv1D(128, 3, activation='relu', padding='same')(inputs)
            x = MaxPooling1D(2)(x)
            x = Conv1D(64, 3, activation='relu', padding='same')(x)
            x = MaxPooling1D(2)(x)
            lstm = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.3))(x)
            attn = MultiHeadAttention(num_heads=4, key_dim=16)(lstm, lstm)
            x = Add()([lstm, attn])
            x = LayerNormalization()(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = Dropout(0.5)(x)
            x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = Dropout(0.4)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        class_weights = tf.constant([1.0, 1.5, 2.0], dtype=tf.float32) if model_type == 'ctg' else tf.constant([1.0, 1.8, 3.0], dtype=tf.float32)  # Reduced class weights for CTG to balance emphasis
        model.compile(optimizer=Adam(learning_rate=0.00002, clipnorm=1.0),
                      loss=FocalLoss(class_weights=class_weights),
                      metrics=['accuracy'])
        return model

    def train_and_evaluate(self, model_type='ctg'):
        X = self.X_ctg if model_type == 'ctg' else self.X_maternal
        y = self.y_ctg if model_type == 'ctg' else self.y_maternal
        scaler = self.scaler_ctg if model_type == 'ctg' else self.scaler_maternal
        class_names = self.class_names_ctg if model_type == 'ctg' else self.class_names_maternal

        X = scaler.transform(X)
        X = np.expand_dims(X, axis=2)

        batch_size, dropout, l2_lambda = (32, 0.7, 0.07) if model_type == 'ctg' else (16, 0.5, 0.01)  # Adjusted for CTG
        epochs = 20 if model_type == 'ctg' else 50  # Reduced epochs for CTG from 25 to 20 to prevent over-convergence

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        all_y_test = []
        all_ensemble_preds = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            skf_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_idx_inner, val_idx = next(skf_val.split(X_train, y_train))
            X_train_inner, X_val = X_train[train_idx_inner], X_train[val_idx]
            y_train_inner, y_val = y_train[train_idx_inner], y_train[val_idx]

            model = self.create_model((X_train.shape[1], 1), len(np.unique(y)), model_type)
            for layer in model.layers:
                if isinstance(layer, Dropout):
                    layer.rate = dropout
                if isinstance(layer, Dense):
                    layer.kernel_regularizer = l2(l2_lambda)

            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            xgb_model = xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1)

            class_counts = np.bincount(y_train_inner)
            total_samples = len(y_train_inner)
            class_weights_dict = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}

            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),  # Reduced patience for CTG from 10 to 8
                ModelCheckpoint(f'best_{model_type}_model_fold{fold}.keras', save_best_only=True, monitor='val_accuracy'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]

            model.fit(X_train_inner, y_train_inner, batch_size=batch_size, epochs=epochs,
                      validation_data=(X_val, y_val), class_weight=class_weights_dict, callbacks=callbacks, verbose=1)
            rf.fit(X_train.squeeze(), y_train)
            xgb_model.fit(X_train.squeeze(), y_train)

            if fold == 4:
                model.save(f'{model_type}_model.keras')
                joblib.dump(rf, f'rf_{model_type}_model.pkl')
                joblib.dump(xgb_model, f'xgb_{model_type}_model.pkl')
                joblib.dump(scaler, f'scaler_{model_type}.pkl')
                X_train_flattened = X_train.squeeze(axis=2)
                np.save(f'X_train_{model_type}.npy', X_train_flattened)
                np.save(f'y_train_{model_type}.npy', y_train)
                logger.info(f"Saved {model_type} models, scaler, and training data for fold {fold+1}")

            dl_val_preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
            rf_val_preds = rf.predict(X_val.squeeze())
            xgb_val_preds = xgb_model.predict(X_val.squeeze())
            dl_acc = accuracy_score(y_val, dl_val_preds)
            rf_acc = accuracy_score(y_val, rf_val_preds)
            xgb_acc = accuracy_score(y_val, xgb_val_preds)
            dl_weight = dl_acc * 1.5
            rf_weight = rf_acc
            xgb_weight = xgb_acc
            total_weight = dl_weight + rf_weight + xgb_weight
            dl_weight, rf_weight, xgb_weight = dl_weight / total_weight, rf_weight / total_weight, xgb_weight / total_weight

            if fold == 4:
                joblib.dump((dl_weight, rf_weight, xgb_weight), f'weights_{model_type}.pkl')
                logger.info(f"Saved ensemble weights for {model_type}: DL={dl_weight:.3f}, RF={rf_weight:.3f}, XGB={xgb_weight:.3f}")

            dl_preds_proba = model.predict(X_test, verbose=0)
            rf_preds_proba = rf.predict_proba(X_test.squeeze())
            xgb_preds_proba = xgb_model.predict_proba(X_test.squeeze())
            ensemble_preds = np.argmax(dl_weight * dl_preds_proba + rf_weight * rf_preds_proba + xgb_weight * xgb_preds_proba, axis=1)

            acc = accuracy_score(y_test, ensemble_preds)
            accuracies.append(acc)
            all_y_test.extend(y_test)
            all_ensemble_preds.extend(ensemble_preds)
            logger.info(f"Fold {fold+1} Accuracy: {acc:.4f} (Weights: DL={dl_weight:.3f}, RF={rf_weight:.3f}, XGB={xgb_weight:.3f})")

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        return mean_acc > (0.93 if model_type == 'maternal' else 0.97), mean_acc, std_acc, np.array(all_y_test), np.array(all_ensemble_preds), class_names

if __name__ == "__main__":
    pipeline = HealthPredictionPipeline()
    ctg_path = r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\CTG Dataset.xls'
    maternal_path = r'C:\Users\Admin\OneDrive\Desktop\22127036 - Pavithra S\Project\Maternal Health Risk Data Set.csv'
    
    if pipeline.load_data(ctg_path, maternal_path) and pipeline.preprocess_data():
        ctg_success, ctg_mean_acc, ctg_std_acc, ctg_y_test, ctg_ensemble_preds, ctg_class_names = pipeline.train_and_evaluate('ctg')
        maternal_success, maternal_mean_acc, maternal_std_acc, maternal_y_test, maternal_ensemble_preds, maternal_class_names = pipeline.train_and_evaluate('maternal')
        
        logger.info(f"CTG Mean CV Accuracy: {ctg_mean_acc:.4f} ± {ctg_std_acc:.4f}")
        logger.info(f"Final Fold Classification Report (CTG):\n{classification_report(ctg_y_test, ctg_ensemble_preds, target_names=ctg_class_names)}")
        
        logger.info(f"MATERNAL Mean CV Accuracy: {maternal_mean_acc:.4f} ± {maternal_std_acc:.4f}")
        logger.info(f"Final Fold Classification Report (Maternal):\n{classification_report(maternal_y_test, maternal_ensemble_preds, target_names=maternal_class_names)}")