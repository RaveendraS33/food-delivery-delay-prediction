"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    restaurant_lat: float = Field(..., ge=-90, le=90, examples=[42.36])
    restaurant_lon: float = Field(..., ge=-180, le=180, examples=[-71.06])
    customer_lat: float = Field(..., ge=-90, le=90, examples=[42.35])
    customer_lon: float = Field(..., ge=-180, le=180, examples=[-71.08])
    order_time: datetime | None = Field(
        None, description="ISO timestamp of the order; defaults to now."
    )
    traffic_level: str | None = Field(
        None, description="low|medium|high|jam; inferred from the hour if omitted."
    )
    vehicle_type: str = Field("scooter", description="bike|scooter|car")
    active_orders: int | None = Field(
        None, ge=0, description="Kitchen/area load; inferred if omitted."
    )
    prep_time_minutes: float | None = Field(
        None, gt=0, description="Restaurant prep estimate; defaults from config."
    )


class PredictResponse(BaseModel):
    predicted_eta_minutes: float
    promised_minutes: float
    predicted_delay_minutes: float
    delay_probability: float
    on_time_probability: float
    risk_level: str
    distance_km: float
    weather_precip_mm: float


class ScheduleSlot(BaseModel):
    offset_hours: int
    time: str
    hour: int
    predicted_eta_minutes: float
    delay_probability: float
    risk_level: str


class RecommendRequest(BaseModel):
    restaurant_lat: float = Field(..., ge=-90, le=90)
    restaurant_lon: float = Field(..., ge=-180, le=180)
    customer_lat: float = Field(..., ge=-90, le=90)
    customer_lon: float = Field(..., ge=-180, le=180)
    start_time: datetime | None = None
    lookahead_hours: int | None = Field(None, ge=1, le=24)
    vehicle_type: str = "scooter"
    active_orders: int | None = Field(None, ge=0)


class RecommendResponse(BaseModel):
    recommend_now: bool
    advice: str
    acceptable_delay_prob: float
    current: ScheduleSlot
    best: ScheduleSlot
    schedule: list[ScheduleSlot]


class ModelInfo(BaseModel):
    loaded: bool
    trained_at: str | None = None
    source: str | None = None
    n_train_rows: int | None = None
    n_features: int | None = None
    metrics: dict | None = None
