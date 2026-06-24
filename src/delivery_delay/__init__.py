"""Food Delivery Delay Prediction System.

End-to-end pipeline: synthetic + public data, weather/geo/temporal feature
engineering, dual XGBoost models (ETA regression + delay-probability
classification), a FastAPI serving layer, and a Streamlit decision-support
dashboard.
"""

__version__ = "0.1.0"
