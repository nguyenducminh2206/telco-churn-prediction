"""
Gradio web UI for the churn model.

Run with:  python src/app/app.py
Then open the URL it prints (usually http://127.0.0.1:7860).
"""

import os
import sys
import gradio as gr

# allow importing from src/ when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.serving.inference import predict_churn


YES_NO = ["No", "Yes"]
SERVICE_OPTIONS = ["No", "Yes", "No internet service"]
PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def _predict(
    gender, SeniorCitizen, Partner, Dependents,
    PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod,
    tenure, MonthlyCharges, TotalCharges,
):
    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    result = predict_churn(customer)
    prob = float(result["probability"])
    return {"Will churn": prob, "Will stay": 1 - prob}


with gr.Blocks(title="Telco Churn Predictor") as demo:
    gr.Markdown(
        """
        # Telco Customer Churn Predictor
        Estimate the probability that a customer will churn from their account
        profile and service mix. The model is an XGBoost classifier trained on the
        IBM Telco Customer Churn dataset.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("Demographics", open=True):
                with gr.Row():
                    gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
                    senior = gr.Radio(YES_NO, value="No", label="Senior Citizen")
                with gr.Row():
                    partner = gr.Radio(YES_NO, value="No", label="Partner")
                    dependents = gr.Radio(YES_NO, value="No", label="Dependents")

            with gr.Accordion("Account & billing", open=True):
                tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months)")
                with gr.Row():
                    contract = gr.Dropdown(
                        ["Month-to-month", "One year", "Two year"],
                        value="Month-to-month",
                        label="Contract",
                    )
                    paperless = gr.Radio(YES_NO, value="Yes", label="Paperless Billing")
                payment = gr.Dropdown(
                    PAYMENT_OPTIONS,
                    value="Electronic check",
                    label="Payment Method",
                )
                with gr.Row():
                    monthly = gr.Number(value=70.0, label="Monthly Charges ($)")
                    total = gr.Number(value=840.0, label="Total Charges ($)")

            with gr.Accordion("Services", open=False):
                with gr.Row():
                    phone = gr.Radio(YES_NO, value="Yes", label="Phone Service")
                    mlines = gr.Dropdown(
                        ["No", "Yes", "No phone service"],
                        value="No",
                        label="Multiple Lines",
                    )
                internet = gr.Dropdown(
                    ["DSL", "Fiber optic", "No"],
                    value="Fiber optic",
                    label="Internet Service",
                )
                with gr.Row():
                    online_sec = gr.Dropdown(SERVICE_OPTIONS, value="No", label="Online Security")
                    online_bk = gr.Dropdown(SERVICE_OPTIONS, value="No", label="Online Backup")
                with gr.Row():
                    device = gr.Dropdown(SERVICE_OPTIONS, value="No", label="Device Protection")
                    tech = gr.Dropdown(SERVICE_OPTIONS, value="No", label="Tech Support")
                with gr.Row():
                    stream_tv = gr.Dropdown(SERVICE_OPTIONS, value="No", label="Streaming TV")
                    stream_mv = gr.Dropdown(SERVICE_OPTIONS, value="No", label="Streaming Movies")

            with gr.Row():
                predict_btn = gr.Button("Predict churn", variant="primary")
                clear_btn = gr.ClearButton(value="Reset")

        with gr.Column(scale=1):
            label_out = gr.Label(label="Prediction", num_top_classes=2)
            gr.Markdown(
                """
                ### How to read this
                - **Will churn** — model's estimated probability the customer leaves.
                - **Will stay** — complement.
                - The label decision threshold lives in
                  `src/serving/inference.py`.
                """
            )

    inputs = [
        gender, senior, partner, dependents,
        phone, mlines,
        internet, online_sec, online_bk, device,
        tech, stream_tv, stream_mv,
        contract, paperless, payment,
        tenure, monthly, total,
    ]

    predict_btn.click(_predict, inputs=inputs, outputs=label_out)
    clear_btn.add(inputs + [label_out])

    gr.Examples(
        examples=[
            # likely churn: short tenure, fiber optic, month-to-month, electronic check, no add-ons
            ["Female", "No", "No", "No", "Yes", "No", "Fiber optic",
             "No", "No", "No", "No", "No", "No",
             "Month-to-month", "Yes", "Electronic check", 2, 95.0, 190.0],
            # likely stay: long tenure, DSL, two-year contract, auto bank transfer, full add-on bundle
            ["Male", "No", "Yes", "Yes", "Yes", "Yes", "DSL",
             "Yes", "Yes", "Yes", "Yes", "Yes", "Yes",
             "Two year", "No", "Bank transfer (automatic)", 60, 55.0, 3300.0],
        ],
        inputs=inputs,
        label="Sample customers (click to load)",
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
