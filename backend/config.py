# backend/config.py
# -----------------

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys (loaded from .env automatically)
    gemini_api_key: Optional[str] = None
    
    # App Settings
    app_name: str = "Demand Forecasting API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Data Requirements (REVISED - More Flexible)
    min_months_for_analysis: int = 2  # Absolute minimum to create any forecast
    min_months_for_seasonality: int = 12  # Recommended for yearly seasonality
    optimal_months: int = 24  # Ideal for robust forecasting
    
    default_forecast_horizon: int = 1
    max_forecast_horizon: int = 6
    
    # Prophet Model Settings (Base Configuration)
    # These will be overridden dynamically based on data availability
    base_yearly_seasonality: bool = True
    weekly_seasonality: bool = False
    daily_seasonality: bool = False
    base_seasonality_mode: str = "multiplicative"
    base_changepoint_prior_scale: float = 0.05
    
    # Adaptive Model Settings
    # More flexible changepoint detection for limited data
    limited_data_changepoint_scale: float = 0.01  # More conservative
    sufficient_data_changepoint_scale: float = 0.05  # Standard
    
    # AI Settings - Gemini
    gemini_model: str = "gemini-1.5-flash"
    ai_temperature: float = 0.3
    ai_max_tokens: int = 350
    
    # Confidence Thresholds (Based on Data Quality)
    excellent_confidence_months: int = 24
    high_confidence_months: int = 18
    medium_confidence_months: int = 12
    low_confidence_months: int = 6
    
    # Trend Detection Thresholds
    trend_strong_up: float = 10.0
    trend_up_threshold: float = 5.0
    trend_down_threshold: float = -5.0
    trend_strong_down: float = -10.0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Create global settings instance
settings = Settings()


# Festival mapping for India (Enhanced)
FESTIVAL_MAP = {
    "January": ["Makar Sankranti", "Republic Day", "Pongal"],
    "February": [],
    "March": ["Holi", "Ugadi"],
    "April": ["Ram Navami", "Mahavir Jayanti"],
    "May": ["Eid al-Fitr (varies)"],
    "June": ["Eid al-Adha (varies)"],
    "July": ["Guru Purnima"],
    "August": ["Independence Day", "Raksha Bandhan", "Janmashtami"],
    "September": ["Ganesh Chaturthi", "Onam"],
    "October": ["Navratri", "Dussehra", "Durga Puja"],
    "November": ["Diwali", "Bhai Dooj", "Guru Nanak Jayanti"],
    "December": ["Christmas", "New Year Eve"]
}


def get_festivals_for_month(month_name: str) -> list[str]:
    """Get festivals for a given month name"""
    return FESTIVAL_MAP.get(month_name, [])


def get_data_quality_tier(data_months: int) -> dict:
    """
    Determine data quality tier and provide appropriate messaging.
    
    Returns:
        dict: Quality tier, message, and recommendations
    """
    if data_months >= settings.optimal_months:
        return {
            "tier": "excellent",
            "label": "Excellent",
            "confidence": "Very High",
            "message": "You have excellent historical data for highly accurate forecasting.",
            "enable_yearly_seasonality": True,
            "enable_holidays": True,
            "warning": None
        }
    elif data_months >= settings.high_confidence_months:
        return {
            "tier": "high",
            "label": "High",
            "confidence": "High",
            "message": "You have sufficient data for reliable forecasting with seasonal patterns.",
            "enable_yearly_seasonality": True,
            "enable_holidays": True,
            "warning": None
        }
    elif data_months >= settings.min_months_for_seasonality:
        return {
            "tier": "medium",
            "label": "Medium",
            "confidence": "Medium",
            "message": "You have adequate data for forecasting. 18+ months recommended for better accuracy.",
            "enable_yearly_seasonality": True,
            "enable_holidays": True,
            "warning": "Limited data for robust yearly seasonality detection"
        }
    elif data_months >= settings.low_confidence_months:
        return {
            "tier": "low",
            "label": "Low",
            "confidence": "Low",
            "message": "Limited data available. Forecast will be based on trend analysis only.",
            "enable_yearly_seasonality": False,
            "enable_holidays": False,
            "warning": "⚠️ Less than 12 months of data - seasonal patterns cannot be reliably detected. Consider collecting more historical data."
        }
    elif data_months >= settings.min_months_for_analysis:
        return {
            "tier": "minimal",
            "label": "Minimal",
            "confidence": "Very Low",
            "message": "Very limited data. Forecast will have high uncertainty.",
            "enable_yearly_seasonality": False,
            "enable_holidays": False,
            "warning": "⚠️ CRITICAL: Only 2-5 months of data. Forecasts will be based purely on trend and have very high uncertainty. Strongly recommend collecting at least 12 months of data."
        }
    else:
        return {
            "tier": "insufficient",
            "label": "Insufficient",
            "confidence": "None",
            "message": "Insufficient data for forecasting.",
            "enable_yearly_seasonality": False,
            "enable_holidays": False,
            "warning": "❌ Cannot generate forecast: Less than 2 months of data"
        }