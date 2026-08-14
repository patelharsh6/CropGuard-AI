"""
Path B: Re-convert the Keras model to dynamic-range TFLite with a fixed static
batch size of 1 to bypass the LiteRT.js dynamic batch dimension mismatch bug.
"""
import os
import numpy as np
import tensorflow as tf
from src.data_pipeline import IMG_SIZE

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CLASS_NAMES = [
    'Corn_Cercospora_Gray_leaf_spot', 'Corn_Common_rust', 'Corn_Northern_Leaf_Blight',
    'Corn_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites',
    'Tomato_Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus',
    'Tomato_healthy'
]

PATHS = [
    'data/plantvillage_dataset/color/Corn_(maize)___Common_rust_/RS_Rust 2370.JPG',
    'data/plantvillage_dataset/color/Potato___healthy/ef7005dc-1d44-412e-b858-145a2d7a6fa9___RS_HL 1951.JPG',
    'data/plantvillage_dataset/color/Tomato___Leaf_Mold/a9ed3e3f-7c8c-4a0e-ae96-8bf55585a522___Crnl_L.Mold 7082.JPG',
]

def load_and_fix_batch_size(keras_path):
    print("Loading original Keras model...")
    # Using tf.keras instead of legacy to load Keras 3 format correctly
    model = tf.keras.models.load_model(keras_path)
    
    # Rebuild the model with a fixed batch_size=1
    print("Rebuilding model with static batch size 1...")
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), batch_size=1, name='input_layer_1')
    outputs = model(inputs)
    static_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return static_model

def convert_to_dynrange_tflite(static_model, output_path):
    print("Converting to dynamic-range TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Dynamic range doesn't use representative dataset or INT8 target specs
    tflite_bytes = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_bytes)
    size = os.path.getsize(output_path)
    print(f"Saved: {output_path} ({size:,} bytes)")
    return tflite_bytes

def verify_predictions(tflite_bytes):
    print("\nVerifying predictions on reference images...")
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    
    print(f"Model Input Shape : {inp['shape']}")
    print(f"Model Output Shape: {out['shape']}")

    for path in PATHS:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        x = img.numpy()[np.newaxis].astype(np.float32)
        
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        probs = interp.get_tensor(out['index'])[0]
        
        argmax = int(np.argmax(probs))
        fname = os.path.basename(path)
        print(f"\nFile: {fname}")
        print(f"Predicted: {CLASS_NAMES[argmax]} (index {argmax})")
        print("Top-3 probs:")
        top3 = np.argsort(probs)[::-1][:3]
        for i in top3:
            print(f"  [{i:2d}] {CLASS_NAMES[i]:35s}  {probs[i]:.6f}")

if __name__ == '__main__':
    keras_path = 'models/cropguard_v1.keras'
    tflite_path = 'models/cropguard_v1_production.tflite'
    
    static_model = load_and_fix_batch_size(keras_path)
    tflite_bytes = convert_to_dynrange_tflite(static_model, tflite_path)
    verify_predictions(tflite_bytes)
