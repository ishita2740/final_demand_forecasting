# backend/main.py
# ---------------
# FastAPI application for Demand Forecasting API

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd

from .data_preparation import prepare_category_data, get_data_summary
from .forecast_service import run_demand_forecast
from .ai_insight_service import generate_ai_insight
from .evaluation import evaluate_forecast_accuracy, get_model_diagnostics
from .config import settings, get_festivals_for_month

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Adaptive AI-powered demand forecasting with comprehensive insights"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "features": [
            "Adaptive forecasting (2+ months of data)",
            "Seasonal pattern detection (12+ months optimal)",
            "Festival impact analysis",
            "Multi-horizon forecasting (1-6 months)",
            "AI-powered insights"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "min_months_required": settings.min_months_for_analysis,
        "recommended_months": settings.min_months_for_seasonality,
        "optimal_months": settings.optimal_months,
        "ai_model": settings.groq_model,
        "max_forecast_horizon": settings.max_forecast_horizon
    }


@app.post("/forecast/upload")
async def upload_and_forecast(
    file: UploadFile,
    category: str = Form(...),
    date_col: str = Form(...),
    category_col: str = Form(...),
    units_col: str = Form(...),
    horizon: int = Form(1)
):
    """
    Upload sales data and generate adaptive AI-powered demand forecast.
    Works with as little as 2 months of data, with warnings for limited data.
    """
    
    try:
        # Validate horizon
        if horizon < 1 or horizon > settings.max_forecast_horizon:
            raise HTTPException(
                status_code=400,
                detail=f"Forecast horizon must be between 1 and {settings.max_forecast_horizon} months"
            )
        
        # 1. Read uploaded file
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read CSV file: {str(e)}"
            )

        # 2. Prepare and aggregate data
        try:
            monthly_df = prepare_category_data(
                df=df,
                category=category,
                date_col=date_col,
                category_col=category_col,
                units_col=units_col
            )
        except ValueError as ve:
            # Data preparation errors (category not found, insufficient data)
            raise HTTPException(
                status_code=400,
                detail=str(ve)
            )
        
        # Get data summary
        data_summary = get_data_summary(monthly_df)

        # 3. Run ADAPTIVE forecast
        try:
            forecast_result = run_demand_forecast(
                monthly_df=monthly_df,
                periods=horizon
            )
        except ValueError as ve:
            # Forecasting errors (insufficient data for model)
            raise HTTPException(
                status_code=400,
                detail=str(ve)
            )

        # 4. Prepare context for AI insights
        next_month = monthly_df["ds"].max() + pd.DateOffset(months=1)
        month_name = next_month.strftime("%B %Y")
        month_only = next_month.strftime("%B")
        
        # Get festivals for the forecast month
        festivals = get_festivals_for_month(month_only)

        # 5. Generate ENHANCED AI Insight with full context
        ai_insight = generate_ai_insight(
            category=category,
            forecasted_units=forecast_result["forecasted_units"],
            mom_change=forecast_result["mom_change_percent"],
            trend=forecast_result["trend"],
            month=month_name,
            lower_bound=forecast_result.get("lower_bound"),
            upper_bound=forecast_result.get("upper_bound"),
            historical_avg=forecast_result.get("historical_avg"),
            yoy_change=forecast_result.get("yoy_change_percent"),
            data_months=forecast_result.get("data_months"),
            confidence=forecast_result.get("confidence"),
            region="India",
            festivals=festivals,
            seasonality=forecast_result.get("seasonality"),
            warnings=forecast_result.get("warnings"),
            coefficient_of_variation=forecast_result.get("coefficient_of_variation")
        )

        # 6. Return COMPREHENSIVE response
        return {
            # Core forecast data
            **forecast_result,
            
            # AI insight
            "ai_insight": ai_insight,
            
            # Additional context
            "data_summary": data_summary,
            "forecast_month": month_name,
            "festivals": festivals,
            
            # User guidance
            "data_quality_message": forecast_result.get("data_quality_message"),
            "warnings": forecast_result.get("warnings", []),
            "recommendations": forecast_result.get("recommendations", [])
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(e)}"
        )


@app.post("/forecast/evaluate")
async def evaluate_model(
    file: UploadFile,
    category: str = Form(...),
    date_col: str = Form(...),
    category_col: str = Form(...),
    units_col: str = Form(...),
    holdout_months: int = Form(3)
):
    """
    Evaluate forecast model accuracy using holdout validation.
    Requires sufficient historical data.
    """
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        monthly_df = prepare_category_data(
            df=df,
            category=category,
            date_col=date_col,
            category_col=category_col,
            units_col=units_col
        )
        
        # Check if enough data for evaluation
        if len(monthly_df) < holdout_months + settings.min_months_for_analysis:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for evaluation. Need at least {holdout_months + settings.min_months_for_analysis} months, have {len(monthly_df)}"
            )
        
        # Run evaluation
        evaluation_result = evaluate_forecast_accuracy(
            monthly_df=monthly_df,
            holdout_months=holdout_months
        )
        
        # Get diagnostics
        diagnostics = get_model_diagnostics(monthly_df)
        
        return {
            "category": category,
            "evaluation": evaluation_result,
            "diagnostics": diagnostics
        }

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(e)}"
        )


@app.post("/data/summary")
async def get_data_info(
    file: UploadFile,
    category: str = Form(...),
    date_col: str = Form(...),
    category_col: str = Form(...),
    units_col: str = Form(...)
):
    """
    Get data summary and quality assessment without running forecast.
    Useful for data validation before forecasting.
    """
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        monthly_df = prepare_category_data(
            df=df,
            category=category,
            date_col=date_col,
            category_col=category_col,
            units_col=units_col
        )
        
        summary = get_data_summary(monthly_df)
        diagnostics = get_model_diagnostics(monthly_df)
        
        data_months = len(monthly_df)
        
        # Determine readiness
        if data_months >= settings.optimal_months:
            readiness = "optimal"
            message = "Excellent data quality - ready for highly accurate forecasting"
        elif data_months >= settings.min_months_for_seasonality:
            readiness = "good"
            message = "Good data quality - ready for seasonal forecasting"
        elif data_months >= settings.min_months_for_analysis:
            readiness = "limited"
            message = "Limited data - forecast will be trend-based only without seasonal patterns"
        else:
            readiness = "insufficient"
            message = f"Insufficient data - need at least {settings.min_months_for_analysis} months"
        
        return {
            "category": category,
            "summary": summary,
            "diagnostics": diagnostics,
            "readiness": readiness,
            "readiness_message": message,
            "ready_for_forecast": data_months >= settings.min_months_for_analysis,
            "can_detect_seasonality": data_months >= settings.min_months_for_seasonality
        }

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(e)}"
        )