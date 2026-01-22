# Skinbot

Skinbot is a small CNN-based skin type classifier with a Tkinter desktop UI.
It supports image uploads or live camera capture to classify skin as dry, normal,
or oily. Training and evaluation are handled by standalone scripts.

## Project Layout

- `skinbot.py`: Desktop UI for inference (upload or camera capture).
- `model_train.py`: Training script that builds and saves `skin_type_model.h5`.
- `test_data.py`: Evaluation script with MLflow logging and report output.
- `skin_type_model.h5`: Saved TensorFlow/Keras model.
- `mlruns/`: MLflow tracking data.
- `docs/architecture.md`: Architecture and ML pipeline documentation.

## Quick Start

1. Train a model (edit dataset path first):
   - `python model_train.py`
2. Evaluate the model (edit test dataset path first):
   - `python test_data.py`
3. Run the desktop app:
   - `python skinbot.py`

## Dependencies

Common runtime dependencies include TensorFlow, OpenCV, Pillow, NumPy,
scikit-learn, MLflow, Matplotlib, and Seaborn. Install the versions that match
your environment and GPU setup.

## Docs

See `docs/architecture.md` for higher-level architecture, ML details, and
Mermaid diagrams.
