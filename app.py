import streamlit as st
import numpy as np
import pandas as pd
import requests
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
import joblib
import io
from PIL import Image
import time
import logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

@st.cache_resource
def load_training_data(dataset):
    try:
        prefix = 'ctg' if dataset == "CTG (Fetal Health)" else 'maternal'
        X_train = np.load(f"C:\\Users\\Admin\\OneDrive\\Desktop\\22127036 - Pavithra S\\Project\\X_train_{prefix}.npy")
        logger.info(f"Loaded training data for {dataset}: shape {X_train.shape}")
        return X_train
    except Exception as e:
        logger.error(f"Error loading training data for {dataset}: {str(e)}")
        st.error(f"Error loading training data: {str(e)}")
        return None

def explain_with_shap(model, feature_names, dataset_name, single_sample=None, cache=None, ensemble_probs=None, rf_model=None, xgb_model=None, dl_weight=0.4, rf_weight=0.3, xgb_weight=0.3):
    cache_key = f"shap_{dataset_name}_{hash(str(single_sample))}"
    if cache and cache_key in cache:
        return cache[cache_key]

    try:
        X_background = load_training_data(dataset_name)
        if X_background is None:
            raise ValueError("Failed to load background data for SHAP analysis")

        if single_sample is not None and len(single_sample.shape) == 2:
            single_sample = np.expand_dims(single_sample, axis=2)

        single_sample_flattened = single_sample.squeeze(axis=2) if single_sample is not None else None

        if X_background.size == 0 or (single_sample_flattened is not None and single_sample_flattened.size == 0):
            raise ValueError("No valid data for SHAP analysis")

        feature_names_list = [str(name) for name in feature_names]
        background_data = X_background[:min(100, X_background.shape[0])]

        def wrapped_model_predict(data):
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=2)
            dl_preds = model.predict(data, batch_size=1, verbose=0)
            data_2d = data.squeeze(axis=2)
            rf_preds = rf_model.predict_proba(data_2d)
            xgb_preds = xgb_model.predict_proba(data_2d)
            return dl_weight * dl_preds + rf_weight * rf_preds + xgb_weight * xgb_preds

        explainer = shap.KernelExplainer(wrapped_model_predict, background_data, nsamples=100)
        single_sample_shap = single_sample_flattened.reshape(1, -1)
        shap_values = explainer.shap_values(single_sample_shap, nsamples=100)

        predicted_class = np.argmax(ensemble_probs)

        if isinstance(shap_values, list):
            shap_values_class = shap_values[predicted_class][0]
        else:
            shap_values_class = shap_values[0, :, predicted_class]

        base_value = (explainer.expected_value[predicted_class] 
                     if isinstance(explainer.expected_value, (list, np.ndarray)) 
                     else explainer.expected_value)

        plt.figure(figsize=(6, 3), dpi=50)
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_class,
                base_values=base_value,
                data=single_sample_flattened.flatten(),
                feature_names=feature_names_list
            ),
            show=False
        )

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=50)
        plt.close()
        buf.seek(0)
        image_bytes = buf.getvalue()

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        image_bytes = buf.getvalue()

        if cache:
            cache[cache_key] = image_bytes
        logger.info("SHAP explanation generated successfully")
        return image_bytes

    except Exception as e:
        logger.error(f"Error in SHAP explanation: {str(e)}")
        st.error(f"Error in SHAP explanation: {str(e)}")
        return None

def explain_with_lime(model, rf_model, xgb_model, dl_weight, rf_weight, xgb_weight, X, feature_names, class_names, single_sample=None, cache=None, ensemble_probs=None):
    cache_key = f"lime_{hash(str(single_sample))}"
    if cache and cache_key in cache:
        return cache[cache_key]

    try:
        training_data = X.squeeze(axis=2)[:min(100, X.shape[0])]
        
        explainer = LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=False,
            sample_around_instance=True
        )

        def model_predict(data):
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            data_3d = np.expand_dims(data, axis=2)
            dl_preds = model.predict(data_3d, verbose=0)
            data_2d = data_3d.squeeze(axis=2)
            rf_preds = rf_model.predict_proba(data_2d)
            xgb_preds = xgb_model.predict_proba(data_2d)
            ensemble_proba = dl_weight * dl_preds + rf_weight * rf_preds + xgb_weight * xgb_preds
            return ensemble_proba

        single_sample_flat = single_sample.squeeze(axis=2).reshape(1, -1)
        
        predicted_class = np.argmax(ensemble_probs)
        exp = explainer.explain_instance(
            single_sample_flat[0],
            model_predict,
            num_features=min(10, len(feature_names)),
            num_samples=2000,
            labels=(predicted_class,)
        )

        exp.local_pred = ensemble_probs
        exp.predict_proba = ensemble_probs

        if cache:
            cache[cache_key] = exp
        logger.info("LIME explanation generated successfully")
        return exp

    except Exception as e:
        logger.error(f"Error in LIME explanation: {str(e)}")
        st.error(f"Error in LIME explanation: {str(e)}")
        return None

st.set_page_config(
    page_title="Health Classification System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        🩺 Fetal & Maternal Health Classification
    </h1>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_cache():
    return {}

cache = get_cache()

@st.cache_resource
def load_data_ranges():
    ctg_data = pd.read_excel(r"E:\Solve 4\CTG Dataset.xls", sheet_name='Raw Data')
    maternal_data = pd.read_csv(r"E:\Solve 4\Maternal Health Risk Data Set.csv")
    
    ctg_ranges = {}
    for feature in ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Variance', 'Tendency']:
        ctg_data[feature] = pd.to_numeric(ctg_data[feature], errors='coerce')
        ctg_ranges[feature] = (ctg_data[feature].min(), ctg_data[feature].max())
    
    maternal_ranges = {}
    for feature in ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate']:
        maternal_data[feature] = pd.to_numeric(maternal_data[feature], errors='coerce')
        maternal_ranges[feature] = (maternal_data[feature].min(), maternal_data[feature].max())
    
    maternal_data['BodyTemp'] = pd.to_numeric(maternal_data['BodyTemp'], errors='coerce')
    maternal_data['BodyTemp'] = (maternal_data['BodyTemp'] - 32) * 5 / 9
    maternal_ranges['BodyTemp'] = (maternal_data['BodyTemp'].min(), maternal_data['BodyTemp'].max())
    
    return ctg_ranges, maternal_ranges

ctg_ranges, maternal_ranges = load_data_ranges()

with st.sidebar:
    st.markdown("## 📊 Configuration")
    dataset_choice = st.selectbox(
        "Choose Dataset",
        ["CTG (Fetal Health)", "Maternal Health"],
        key="dataset_select"
    )

    if dataset_choice == "Maternal Health":
        bs_unit = st.selectbox(
            "Blood Sugar Unit",
            ["mmol/L", "mg/dL"],
            key="bs_unit_select",
            help="Select the unit for blood sugar measurement."
        )
        temp_unit = st.selectbox(
            "Temperature Unit",
            ["°C", "°F"],
            key="temp_unit_select",
            help="Select the unit for body temperature."
        )

user_features = []

if dataset_choice == "CTG (Fetal Health)":
    st.markdown("### 📈 CTG Features Input")
    col1, col2, col3 = st.columns(3)
    with col1:
        lb = st.number_input("Baseline FHR (bpm)", min_value=100, max_value=180, value=160, step=1, help="Baseline fetal heart rate in beats per minute (normal: 110-160 bpm)")
        ac = st.number_input("Accelerations (in 20 min)", min_value=0, max_value=10, value=4, step=1, help="Number of accelerations in a 20-minute tracing (normal: 2-5)")
        fm = st.number_input("Fetal Movements (in 20 min)", min_value=0, max_value=100, value=10, step=1, help="Number of fetal movements detected in a 20-minute tracing (normal: 10-50)")
        uc = st.number_input("Uterine Contractions (in 10 min)", min_value=0, max_value=15, value=8, step=1, help="Number of uterine contractions in a 10-minute period (normal: 3-5 during labor)")
        dl = st.number_input("Light Decelerations (in 20 min)", min_value=0, max_value=20, value=5, step=1, help="Number of light decelerations in a 20-minute tracing (normal: 0)")
    with col2:
        ds = st.number_input("Severe Decelerations (in 20 min)", min_value=0, max_value=5, value=1, step=1, help="Number of severe decelerations in a 20-minute tracing (normal: 0)")
        dp = st.number_input("Prolonged Decelerations (in 20 min)", min_value=0, max_value=5, value=1, step=1, help="Number of prolonged decelerations in a 20-minute tracing (normal: 0)")
        astv = st.number_input("Abnormal Short-Term Variability (%)", min_value=0, max_value=100, value=40, step=1, help="Percentage of time with abnormal short-term variability (0-100%)")
        mstv = st.number_input("Mean Short-Term Variability (bpm)", min_value=0, max_value=10, value=1, step=1, help="Mean short-term variability in beats per minute (normal: 5-10 bpm)")
        altv = st.number_input("Abnormal Long-Term Variability (%)", min_value=0, max_value=100, value=30, step=1, help="Percentage of time with abnormal long-term variability (0-100%)")
    with col3:
        mltv = st.number_input("Mean Long-Term Variability (bpm)", min_value=0, max_value=60, value=5, step=1, help="Mean long-term variability in beats per minute (normal: 10-25 bpm)")
        width = st.number_input("Histogram Width (bpm)", min_value=0, max_value=200, value=79, step=1, help="Width of the FHR histogram in beats per minute (0-200 bpm)")
        variance = st.number_input("Histogram Variance", min_value=0, max_value=300, value=50, step=1, help="Variance of the FHR histogram (0-300)")
        tendency = st.number_input("Tendency", min_value=-1, max_value=1, value=-1, step=1, help="Tendency of the FHR histogram (-1: left, 0: neutral, 1: right)")

    user_features = [lb, ac, fm, uc, dl, ds, dp, astv, mstv, altv, mltv, width, variance, tendency]
    feature_names = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Variance', 'Tendency']
    class_names = ['Normal', 'Suspect', 'Pathological']

elif dataset_choice == "Maternal Health":
    st.markdown("### 🩸 Maternal Health Features Input")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=10, max_value=70, value=25, step=1, help="Mother's age in years (10-70)")
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=180, value=110, step=1, help="Systolic blood pressure in mmHg (normal: 90-120 mmHg)")
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=120, value=70, step=1, help="Diastolic blood pressure in mmHg (normal: 60-80 mmHg)")
        
        if bs_unit == "mmol/L":
            bs_label = "Blood Sugar Level (mmol/L)"
            bs_help = "Blood sugar level in mmol/L (normal: 4-7 mmol/L, prediabetes: 5.5-6.9 mmol/L, diabetes: ≥7.0 mmol/L)"
            bs_min = 4
            bs_max = 30
            bs_value = 5
        else:
            bs_label = "Blood Sugar Level (mg/dL)"
            bs_help = "Blood sugar level in mg/dL (normal: 72-126 mg/dL, prediabetes: 100-125 mg/dL, diabetes: ≥126 mg/dL)"
            bs_min = 72
            bs_max = 540
            bs_value = 90
        
        bs_input = st.number_input(bs_label, min_value=bs_min, max_value=bs_max, value=bs_value, step=1, help=bs_help)
        bs = bs_input if bs_unit == "mmol/L" else bs_input / 18.0

    with col2:
        if temp_unit == "°C":
            temp_label = "Body Temperature (°C)"
            temp_help = "Body temperature in Celsius (normal: 36.1-37.2 °C)"
            temp_min = 36.0
            temp_max = 41.0
            temp_value = 36.6
            temp_step = 0.1
        else:
            temp_label = "Body Temperature (°F)"
            temp_help = "Body temperature in Fahrenheit (normal: 97.0-99.0 °F)"
            temp_min = 96.8
            temp_max = 105.8
            temp_value = 97.9
            temp_step = 0.1
        
        temp_input = st.number_input(temp_label, min_value=temp_min, max_value=temp_max, value=temp_value, step=temp_step, help=temp_help)
        body_temp = temp_input if temp_unit == "°C" else (temp_input - 32) * 5 / 9

        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=70, step=1, help="Heart rate in beats per minute (normal: 60-100 bpm)")
        bp_ratio = f"{int(systolic_bp)}/{int(diastolic_bp)}" if diastolic_bp != 0 else "0/0"

    user_features = [age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate, float(systolic_bp / diastolic_bp if diastolic_bp != 0 else 0)]
    feature_names = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'BP_Ratio']
    class_names = ['Low Risk', 'Mid Risk', 'High Risk']

if st.button("🔍 Generate Prediction", type="primary"):
    if any(pd.isna(value) for value in user_features):
        st.error("❌ Please fill in all required fields!")
    elif dataset_choice == "Maternal Health" and diastolic_bp == 0:
        st.error("❌ Diastolic BP must be greater than 0!")
    else:
        with st.spinner("Analyzing..."):
            try:
                flask_api_url = "http://localhost:5000/predict"

                @st.cache_data(ttl=300)
                def get_prediction(features, dataset):
                    response = requests.post(
                        flask_api_url,
                        json={"features": list(features), "dataset": dataset},
                        timeout=10.0
                    )
                    response.raise_for_status()
                    return response.json()

                prediction = get_prediction(tuple(user_features), dataset_choice)
                ensemble_probs = np.array(prediction['probabilities'], dtype=np.float32)
                
                expected_classes = len(class_names)
                if ensemble_probs.shape != (expected_classes,):
                    raise ValueError(f"Expected {expected_classes} probabilities, but got {ensemble_probs.shape}")

                feature_display = "" if dataset_choice == "CTG (Fetal Health)" else f"BP Ratio: {bp_ratio}"

                st.markdown(f"""
                    <div style='padding: 20px; border-radius: 10px; background-color: #E8F6F3;'>
                        <h3 style='color: #2E86C1;'>Prediction Results</h3>
                        <p style='font-size: 18px;'>Classification: {prediction['class']}</p>
                        <p style='font-size: 16px;'>Probabilities: {', '.join([f'{class_names[i]}: {prob:.3f}' for i, prob in enumerate(ensemble_probs)])}</p>
                        {f"<p style='font-size: 16px;'>Input Features:<br>{feature_display}</p>" if feature_display else ""}
                    </div>
                """, unsafe_allow_html=True)

                @st.cache_resource
                def load_resources(dataset):
                    prefix = 'ctg' if dataset == "CTG (Fetal Health)" else 'maternal'
                    model = load_model(f"C:\\Users\\Admin\\OneDrive\\Desktop\\22127036 - Pavithra S\\Project\\{prefix}_model.keras", custom_objects={'FocalLoss': FocalLoss})
                    scaler = joblib.load(f"C:\\Users\\Admin\\OneDrive\\Desktop\\22127036 - Pavithra S\\Project\\scaler_{prefix}.pkl")
                    rf_model = joblib.load(f"C:\\Users\\Admin\\OneDrive\\Desktop\\22127036 - Pavithra S\\Project\\rf_{prefix}_model.pkl")
                    xgb_model = joblib.load(f"C:\\Users\\Admin\\OneDrive\\Desktop\\22127036 - Pavithra S\\Project\\xgb_{prefix}_model.pkl")
                    weights = joblib.load(f"C:\\Users\\Admin\\OneDrive\\Desktop\\22127036 - Pavithra S\\Project\\weights_{prefix}.pkl")
                    return model, scaler, rf_model, xgb_model, weights

                model, scaler, rf_model, xgb_model, (dl_weight, rf_weight, xgb_weight) = load_resources(dataset_choice)

                X_background = load_training_data(dataset_choice)
                if X_background is None:
                    raise ValueError("Failed to load training data for explanations")

                input_data = scaler.transform(np.array(user_features).reshape(1, -1))
                X_background_for_lime = np.expand_dims(X_background, axis=2)
                input_data = np.expand_dims(input_data, axis=2)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔍 SHAP Explanation")
                    shap_image = explain_with_shap(
                        model, feature_names, dataset_choice,
                        single_sample=input_data, cache=cache, ensemble_probs=ensemble_probs,
                        rf_model=rf_model, xgb_model=xgb_model, dl_weight=dl_weight, rf_weight=rf_weight, xgb_weight=xgb_weight
                    )
                    if shap_image:
                        st.image(shap_image, caption="SHAP Waterfall Plot", width=450, use_container_width=True)
                    else:
                        st.error("Failed to generate SHAP explanation.")

                with col2:
                    st.markdown("### 🔍 LIME Explanation")
                    lime_exp = explain_with_lime(
                        model, rf_model, xgb_model, dl_weight, rf_weight, xgb_weight,
                        X_background_for_lime, feature_names, class_names,
                        single_sample=input_data, cache=cache, ensemble_probs=ensemble_probs
                    )
                    if lime_exp:
                        st.components.v1.html(lime_exp.as_html(), height=600, scrolling=True)
                    else:
                        st.error("Failed to generate LIME explanation.")

            except Exception as e:
                logger.error(f"Error in prediction pipeline: {str(e)}")
                st.error(f"❌ Error: {str(e)}")

with st.sidebar:
    if st.button("🗑 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        cache.clear()
        st.success("✅ Cache cleared successfully!")