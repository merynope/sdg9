import streamlit as st
import numpy as np
import tensorflow as tf

# Load models
@st.cache_resource
def load_model_h5(file_path):
    """Load the model saved as an HDF5 file."""
    model = tf.keras.models.load_model(file_path)
    
    # If the LSTM layer has any incompatible configurations like 'time_major', adjust them
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            config = layer.get_config()
            if 'time_major' in config:
                config.pop('time_major')  # Remove time_major argument if it exists
                layer.__init__(**config)  # Reinitialize the LSTM layer with the new config
    return model

# Load all models (make sure your model files are in HDF5 format)
models = {
    "FD001": load_model_h5("FD001.h5"),
    "FD002": load_model_h5("FD002.h5"),
    "FD003": load_model_h5("FD003.h5"),
    "FD004": load_model_h5("FD004.h5"),
}

# Streamlit App Title
st.title("Predictive Maintenance of Aircraft Engines")
st.write(
    "This app predicts failure or the Remaining Useful Life (RUL) of engines using "
    "predictive models trained on the NASA C-MAPSS dataset."
)

# Sidebar for model selection
selected_model = st.sidebar.selectbox(
    "Select Dataset (FD001, FD002, FD003, FD004):",
    options=list(models.keys())
)

st.write(f"### Selected Dataset: {selected_model}")

# Input Features
st.write("#### Enter Engine Data for Prediction")
cols = st.columns(4)

# Assuming there are 14 sensor measurements as input
num_sensors = 14
inputs = []
for i in range(1, num_sensors + 1):
    with cols[(i - 1) % 4]:  # Distribute inputs across 4 columns
        value = st.number_input(f"Sensor {i} Reading:", value=0.0, format="%.3f")
        inputs.append(value)

# Convert inputs to NumPy array
input_array = np.array(inputs).reshape(1, -1)  # Reshape for model input

# Make prediction
if st.button("Predict"):
    model = models[selected_model]
    prediction = model.predict(input_array)

    # Output results
    st.write(f"### Prediction for {selected_model}:")
    if len(prediction.shape) == 1 or prediction.shape[1] == 1:  # Regression (RUL prediction)
        st.write(f"**Predicted RUL:** {prediction[0][0]:.2f} cycles")
    else:  # Classification (failure prediction)
        class_label = np.argmax(prediction)  # Assuming softmax output
        st.write(f"**Failure Prediction Class:** {class_label}")

# Additional Info
st.sidebar.write("## About")
st.sidebar.write(
    "This application uses models trained on the NASA C-MAPSS dataset to predict "
    "engine failures or Remaining Useful Life (RUL)."
)
