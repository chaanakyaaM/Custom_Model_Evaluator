import pickle
import numpy as np
import streamlit as st

st.title("Custom ML Model Evaluator")

# Upload the model
uploaded_file = st.file_uploader("Upload your trained ML model (.pkl)", type=["pkl"])

if uploaded_file is not None:
    package = pickle.load(uploaded_file)
    model = package["model"]
    schema = package["schema"]
    st.success("Model loaded successfully!")

    st.subheader("Enter Input Features:")

    feature_values = []

    for i in schema:
        col_name = i["column_name"]
        dtype = i["type"]
        min_val = i["min"]
        max_val = i["max"]
        unique_values = i.get("unique_list")

        # Unique key to avoid widget name conflicts
        key = f"input_{col_name}"

        if dtype == "int":
            val = st.number_input(col_name, min_value=int(min_val), max_value=int(max_val),
                                  value=int(min_val), step=1, key=key)
        elif dtype == "float":
            val = st.number_input(col_name, min_value=float(min_val), max_value=float(max_val),
                                  value=float(min_val), key=key)
        elif dtype == "string" and unique_values:
            # Use a dropdown for categorical/string features
            val = st.selectbox(col_name, options=unique_values, key=key)
        else:
            val = st.text_input(col_name, key=key)

        feature_values.append(val)

    if st.button("Predict"):
        try:
            # Convert to 2D numpy array for prediction
            features_array = np.array([feature_values])
            prediction = model.predict(features_array)
            probability = model.predict_proba(features_array)

            if prediction == 0:
                st.warning(f'The prediction is {probability[0][0]*100:.2f}% False : {prediction}.')
            else:
                st.success(f"The prediction is {probability[0][1]*100:.2f}% True : {prediction}.")

        except Exception as e:
            st.error(f" Error during prediction: {e}")

else:
    st.info("Upload a pickle file to continue.")
