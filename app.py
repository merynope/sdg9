import streamlit as st
import numpy as np
import pandas as pd

# Define the number of sensors and acceptable ranges for the sensor inputs
NUM_SENSORS = 14
SENSOR_RANGES = [(0, 100) for _ in range(NUM_SENSORS)]  # Example: Sensor values range from 0 to 100

# Function to generate random believable RUL and TTF
def generate_random_rul_ttf():
    rul = np.random.randint(50, 300)  # Random RUL between 50 and 300 cycles
    ttf = rul + np.random.randint(1, 50)  # TTF slightly higher than RUL
    return rul, ttf

# Streamlit App
st.title("Predict Remaining Useful Life (RUL) and Time to Failure (TTF)")
st.write("Enter the values for the sensors below:")

# Sensor input form
sensor_inputs = []
for i in range(NUM_SENSORS):
    sensor_value = st.number_input(f"Sensor {i+1} Value", min_value=SENSOR_RANGES[i][0], max_value=SENSOR_RANGES[i][1], value=(SENSOR_RANGES[i][1] // 2))
    sensor_inputs.append(sensor_value)

# Display input values in a table
st.write("### Input Sensor Values")
sensor_df = pd.DataFrame([sensor_inputs], columns=[f"Sensor {i+1}" for i in range(NUM_SENSORS)])
st.dataframe(sensor_df)

# Predict button
if st.button("Predict"):
    rul, ttf = generate_random_rul_ttf()
    st.success(f"Predicted Remaining Useful Life (RUL): *{rul} cycles*")
    st.success(f"Predicted Time to Failure (TTF): *{ttf} cycles*")

    # Visualize the RUL and TTF values in a bar chart
    st.write("### Visualization")
    st.bar_chart(pd.DataFrame({"Metric": ["RUL", "TTF"], "Value": [rul, ttf]}).set_index("Metric"))

# Footer
st.write("Note: These values are generated randomly for demonstration purposes and are not based on actual predictive models.")
