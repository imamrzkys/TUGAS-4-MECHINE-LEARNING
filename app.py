from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import h5py
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'lstm_btc_daily_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_minmax.joblib')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder='static', template_folder='templates')

model = None
scaler = None
model_error = None
scaler_error = None

def try_load_assets():
    global model, scaler, model_error, scaler_error
    try:
        from tensorflow import keras
        # Prefer H5 if present; otherwise try SavedModel directory
        if os.path.exists(MODEL_PATH):
            # Compatibility shim for models saved with 'batch_shape' in InputLayer config
            class InputLayerCompat(keras.layers.InputLayer):
                def __init__(self, *args, **kwargs):
                    if 'batch_shape' in kwargs and 'batch_input_shape' not in kwargs:
                        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                    super().__init__(*args, **kwargs)
            try:
                model = keras.models.load_model(
                    MODEL_PATH, compile=False,
                    custom_objects={'InputLayer': InputLayerCompat}
                )
            except Exception:
                # Legacy HDF5 path
                try:
                    from tensorflow.python.keras.saving import hdf5_format  # type: ignore
                    with h5py.File(MODEL_PATH, mode='r') as f:
                        model = hdf5_format.load_model_from_hdf5(f)
                except Exception:
                    # Try SavedModel directory if exists
                    saved_dir = os.path.join(BASE_DIR, 'model_saved_tf')
                    if os.path.isdir(saved_dir):
                        model = keras.models.load_model(saved_dir, compile=False)
                    else:
                        # Rebuild from model_config then load weights if available
                        with h5py.File(MODEL_PATH, 'r') as f:
                            if 'model_config' in f:
                                raw = f['model_config'][()]
                                text = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
                                text_fixed = text.replace('"batch_shape"', '"batch_input_shape"')
                                arch = json.loads(text_fixed)
                                model = keras.models.model_from_json(json.dumps(arch))
                                model.load_weights(MODEL_PATH)
                            else:
                                raise FileNotFoundError('model_config not found in H5')
        else:
            # No H5: try SavedModel directory
            saved_dir = os.path.join(BASE_DIR, 'model_saved_tf')
            if os.path.isdir(saved_dir):
                model = keras.models.load_model(saved_dir, compile=False)
            else:
                model_error = 'Model file not found'
    except Exception as e:
        model_error = str(e)

    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            scaler_error = 'Scaler file not found'
    except Exception as e:
        scaler_error = str(e)

try_load_assets()

def list_static_charts():
    if not os.path.isdir(STATIC_DIR):
        return []
    files = []
    for name in os.listdir(STATIC_DIR):
        lower = name.lower()
        if lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg')):
            files.append(name)
    files.sort()
    return files

@app.route('/')
def index():
    charts = list_static_charts()
    return render_template('index.html',
                           charts=charts,
                           model_loaded=model is not None,
                           scaler_loaded=scaler is not None,
                           model_error=model_error,
                           scaler_error=scaler_error,
                           predict_error=None,
                           predict_value=None)

def _parse_series(text):
    if not text:
        return []
    parts = [p.strip() for p in text.replace('\n', ',').split(',') if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            return []
    return vals

@app.route('/predict', methods=['POST'])
def predict():
    charts = list_static_charts()
    if model is None or scaler is None:
        return render_template('index.html', charts=charts, model_loaded=model is not None, scaler_loaded=scaler is not None, model_error=model_error, scaler_error=scaler_error, predict_error='Model atau scaler belum siap.', predict_value=None)
    text = request.form.get('values', '')
    series = _parse_series(text)
    if len(series) < 60:
        return render_template('index.html', charts=charts, model_loaded=True, scaler_loaded=True, model_error=model_error, scaler_error=scaler_error, predict_error='Minimal 60 nilai penutupan diperlukan.', predict_value=None)
    window = np.array(series[-60:], dtype=float).reshape(-1, 1)
    try:
        scaled = scaler.transform(window)
        x = scaled.reshape(1, 60, 1)
        from tensorflow.keras.models import Model  # ensure TF present at call time
        y_scaled = model.predict(x, verbose=0)
        y = scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
        return render_template('index.html', charts=charts, model_loaded=True, scaler_loaded=True, model_error=model_error, scaler_error=scaler_error, predict_error=None, predict_value=float(y))
    except Exception as e:
        return render_template('index.html', charts=charts, model_loaded=True, scaler_loaded=True, model_error=model_error, scaler_error=scaler_error, predict_error=str(e), predict_value=None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG') == '1')
