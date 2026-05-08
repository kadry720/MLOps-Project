"""Gradio prediction interface for the fraud detection model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from multipart.multipart import parse_options_header as _parse_options_header  # noqa: F401
except ModuleNotFoundError:
    from python_multipart import multipart as _python_multipart

    sys.modules.setdefault("multipart.multipart", _python_multipart)

import gradio as gr  # noqa: E402

from src.serving.model_service import PredictionService  # noqa: E402

APP_TITLE = "Fraud Prediction Console"

CUSTOM_CSS = """
.gradio-container {
    background: #f7f8fb;
    color: #172033;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.hero {
    background: linear-gradient(135deg, #14213d 0%, #1f7a8c 55%, #2a9d8f 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 8px;
    margin-bottom: 18px;
}
.hero h1 {
    margin: 0;
    font-size: 30px;
    line-height: 1.1;
    letter-spacing: 0;
}
.hero p {
    margin: 8px 0 0;
    max-width: 760px;
    color: #eef7f8;
}
.metric-box textarea,
.metric-box input {
    font-weight: 650;
}
button.primary {
    border-radius: 8px !important;
}
"""


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _parse_payload(payload: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Invalid JSON: {exc.msg}") from exc

    if isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], dict):
        parsed = parsed["features"]
    if not isinstance(parsed, dict) or not parsed:
        raise gr.Error("Prediction input must be a non-empty JSON object.")
    return parsed


def create_interface() -> gr.Blocks:
    service = PredictionService()

    def load_sample(row_index: int | float) -> str:
        return service.sample_as_json(int(row_index or 0))

    def predict(payload: str, confidence_level: float) -> tuple[str, str, str, dict[str, Any]]:
        result = service.predict(
            _parse_payload(payload),
            confidence_level=float(confidence_level),
        )
        interval = result["confidence_interval"]
        prediction_summary = (
            f"{result['label']} "
            f"(threshold {result['decision_threshold']:.2f}, "
            f"confidence {_format_percentage(result['confidence'])})"
        )
        probability = _format_percentage(result["fraud_probability"])
        interval_text = (
            f"{_format_percentage(interval['lower'])} - {_format_percentage(interval['upper'])} "
            f"at {_format_percentage(interval['confidence_level'])}"
        )
        return prediction_summary, probability, interval_text, result

    try:
        initial_sample = service.sample_as_json(0)
    except (FileNotFoundError, ValueError):
        initial_sample = json.dumps(
            {
                "TransactionDT": 762552,
                "TransactionAmt": 25.0,
                "ProductCD": "H",
                "card1": 7585,
                "card4": "visa",
                "card6": "credit",
                "P_emaildomain": "gmail.com",
            },
            indent=2,
        )

    with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <section class="hero">
              <h1>Fraud Prediction Console</h1>
              <p>Load a held-out sample, edit the raw transaction JSON, and score it with the saved best model.</p>
            </section>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                with gr.Row():
                    sample_index = gr.Number(
                        value=0,
                        precision=0,
                        label="Test row",
                    )
                    load_button = gr.Button("Load sample", variant="secondary")
                payload = gr.Textbox(
                    value=initial_sample,
                    lines=24,
                    label="Prediction sample",
                    show_copy_button=True,
                )
                confidence_level = gr.Slider(
                    minimum=0.5,
                    maximum=0.999,
                    value=0.95,
                    step=0.001,
                    label="Confidence level",
                )
                predict_button = gr.Button("Predict", variant="primary")

            with gr.Column(scale=2):
                prediction = gr.Textbox(label="Prediction", elem_classes=["metric-box"])
                probability = gr.Textbox(label="Fraud probability", elem_classes=["metric-box"])
                interval = gr.Textbox(label="Confidence interval", elem_classes=["metric-box"])
                details = gr.JSON(label="Response details")

        load_button.click(load_sample, inputs=sample_index, outputs=payload)
        predict_button.click(
            predict,
            inputs=[payload, confidence_level],
            outputs=[prediction, probability, interval, details],
        )

    return demo


if __name__ == "__main__":
    create_interface().launch(server_name="127.0.0.1", server_port=7860)
