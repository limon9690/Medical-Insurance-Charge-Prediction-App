# Imports
import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load the Model
with open("pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# The Logic Function
def predict_output(age, bmi, children, sex, smoker, region):
    input_df = pd.DataFrame([[
        age, bmi, children, sex, smoker, region
    ]],
      columns=[
        'age', 'bmi', 'children', 'sex', 'smoker', 'region'
    ])

    # Predict
    prediction = model.predict(input_df)[0]

    return f"Predicted Insurance Charge: {prediction:.2f}"

# Defining inputs in a list to keep it clean
inputs = [
    gr.Number(label="age", value=19),
    gr.Number(label="bmi", value=22),
    gr.Number(label="children", value=0),
    gr.Radio(["male", "female"], label="Gender"),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Radio(["southwest", "northwest", "southeast", "northeast"], label="Region"),
]


# The App Interface
app = gr.Interface(
    fn=predict_output,
      inputs=inputs,
        outputs="text", 
        title="Insurance Charge Predictor")

app.launch()