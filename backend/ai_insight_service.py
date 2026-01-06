# backend/ai_insight_service.py
# -----------------------------

import google.generativeai as genai
from .config import settings, get_festivals_for_month
from typing import Optional

# Initialize Gemini client with API key from settings
genai.configure(api_key=settings.gemini_api_key)


def generate_ai_insight(
    category: str,
    forecasted_units: int,
    mom_change: float,
    trend: str,
    month: str,
    lower_bound: int = None,
    upper_bound: int = None,
    historical_avg: float = None,
    yoy_change: float = None,
    data_months: int = None,
    confidence: str = None,
    region: str = "India",
    festivals: list[str] | None = None,
    seasonality: dict = None,
    warnings: list[str] = None,
    coefficient_of_variation: float = None
) -> str:
    """
    Generate AI-powered explanation for demand forecast with emphasis on
    seasonal patterns and external factors.
    """

    # Get festivals if not provided
    if festivals is None:
        month_name = month.split()[0] if " " in month else month
        festivals = get_festivals_for_month(month_name)
    
    festivals_text = ", ".join(festivals) if festivals else "No major festivals"
    has_festivals = len(festivals) > 0 if festivals else False
    
    # === SEASONALITY ANALYSIS ===
    seasonality_text = ""
    seasonality_emphasis = ""
    if seasonality:
        yearly_strength = seasonality.get("yearly_seasonality_strength", 0)
        holiday_strength = seasonality.get("holiday_impact_strength", 0)
        seasonality_detected = seasonality.get("seasonality_detected", False)
        interpretation = seasonality.get("interpretation", "")
        
        if seasonality_detected:
            seasonality_text = f"""
**SEASONALITY ANALYSIS (CRITICAL):**
- Yearly seasonal pattern strength: {yearly_strength:.1f}%
- Holiday impact strength: {holiday_strength:.1f}%
- Pattern interpretation: {interpretation}
- Seasonality is a KEY factor in this forecast
"""
            seasonality_emphasis = "⚠️ IMPORTANT: This forecast is significantly influenced by seasonal patterns. "
        else:
            seasonality_text = f"""
**SEASONALITY ANALYSIS:**
- Yearly seasonal pattern strength: {yearly_strength:.1f}%
- Pattern interpretation: {interpretation}
- Demand appears relatively stable without strong seasonal variation
"""
    
    # === CONFIDENCE INTERVAL ===
    confidence_text = ""
    uncertainty_level = ""
    if lower_bound is not None and upper_bound is not None:
        range_pct = ((upper_bound - lower_bound) / forecasted_units * 100) if forecasted_units > 0 else 0
        confidence_text = f"- Expected range: {lower_bound:,} to {upper_bound:,} units (±{range_pct:.0f}%)"
        
        if range_pct > 40:
            uncertainty_level = "HIGH uncertainty - wide forecast range"
        elif range_pct > 25:
            uncertainty_level = "MODERATE uncertainty"
        else:
            uncertainty_level = "Low uncertainty - narrow forecast range"
    
    # === HISTORICAL CONTEXT ===
    historical_context = ""
    if historical_avg is not None:
        deviation = ((forecasted_units - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
        
        if abs(deviation) > 20:
            deviation_desc = "SIGNIFICANTLY" if abs(deviation) > 40 else "NOTABLY"
            direction = "above" if deviation > 0 else "below"
            historical_context = f"- Historical context: Forecast is {deviation_desc} {direction} average ({deviation:+.1f}% from {historical_avg:,.0f} avg)"
        else:
            historical_context = f"- Historical context: Forecast aligns with average ({historical_avg:,.0f} units, {deviation:+.1f}%)"
    
    # === VARIABILITY CONTEXT ===
    variability_text = ""
    if coefficient_of_variation is not None:
        if coefficient_of_variation > 50:
            variability_text = f"- Data variability: HIGH ({coefficient_of_variation:.0f}% CoV) - historical demand is volatile"
        elif coefficient_of_variation > 30:
            variability_text = f"- Data variability: MODERATE ({coefficient_of_variation:.0f}% CoV)"
        else:
            variability_text = f"- Data variability: LOW ({coefficient_of_variation:.0f}% CoV) - stable demand pattern"
    
    # === YOY COMPARISON ===
    yoy_context = ""
    if yoy_change is not None:
        if abs(yoy_change) > 20:
            yoy_emphasis = "MAJOR" if abs(yoy_change) > 40 else "SIGNIFICANT"
            yoy_context = f"- Year-over-year: {yoy_emphasis} {'growth' if yoy_change > 0 else 'decline'} ({yoy_change:+.1f}%)"
        else:
            yoy_context = f"- Year-over-year: Relatively stable ({yoy_change:+.1f}%)"
    
    # === DATA QUALITY WARNING ===
    data_quality_text = ""
    if data_months is not None and confidence is not None:
        data_quality_text = f"- Data quality: {confidence} confidence (based on {data_months} months)"
        
        if data_months < 12:
            data_quality_text += "\n- ⚠️ LIMITED DATA: Seasonal patterns cannot be reliably detected with less than 12 months"
    
    # === WARNINGS ===
    warnings_text = ""
    if warnings and len(warnings) > 0:
        warnings_text = "\n**WARNINGS:**\n" + "\n".join(f"- {w}" for w in warnings)

    # === FESTIVAL CONTEXT ===
    festival_context = ""
    if has_festivals:
        festival_context = f"""
**EXTERNAL FACTORS:**
- Upcoming festivals: {festivals_text}
- Festival impact: These celebrations typically drive increased consumer demand in India
"""

    prompt = f"""You are an expert supply chain analyst providing actionable insights for business decision-makers.

## FORECAST SUMMARY
- **Category**: {category}
- **Forecasted demand for {month}**: {forecasted_units:,} units
- **Month-over-month change**: {mom_change:+.1f}%
- **Trend**: {trend}
{confidence_text}
{historical_context}
{yoy_context}
{variability_text}
{data_quality_text}

{seasonality_text}

{festival_context}

{warnings_text}

## YOUR TASK
Write a concise business insight (3-5 sentences) that:

1. **LEAD with the most important factor** (seasonality if detected, festivals if present, otherwise trend)
2. **Explain WHY this forecast makes sense** based on:
   - Seasonal patterns (if detected - THIS IS CRITICAL)
   - Festival/holiday effects (if applicable)
   - Year-over-year trends
   - Historical patterns
3. **Highlight KEY RISKS or OPPORTUNITIES:**
   - Stock shortages if demand spikes
   - Excess inventory if demand drops
   - Uncertainty level ({uncertainty_level})
4. **Provide 1-2 SPECIFIC, ACTIONABLE recommendations:**
   - Inventory targets
   - Safety stock considerations
   - Supply chain preparations

## CRITICAL INSTRUCTIONS:
- {seasonality_emphasis}Use simple, direct language
- Start with the MOST IMPORTANT insight first
- Be specific with numbers when relevant
- If seasonality is strong (>25%), EMPHASIZE it prominently
- If festivals are present, explain their likely impact
- Don't repeat the numbers already shown above
- Focus on WHY and WHAT TO DO

Write your insight now:"""

    try:
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.ai_temperature,
                max_output_tokens=settings.ai_max_tokens,
            )
        )
        return response.text.strip()
    
    except Exception as e:
        # Enhanced fallback response
        fallback = f"Forecast indicates {trend.lower()} demand trend with {forecasted_units:,} units expected for {month} ({mom_change:+.1f}% vs last month). "
        
        if seasonality and seasonality.get("seasonality_detected"):
            fallback += f"Seasonal patterns detected - {seasonality.get('interpretation', '')}. "
        
        if has_festivals:
            fallback += f"Festival season ({festivals_text}) may boost demand. "
        
        if data_months and data_months < 12:
            fallback += "⚠️ Note: Limited historical data may affect accuracy. "
        
        fallback += "Review inventory levels and supply chain capacity accordingly."
        
        return fallback


def generate_inventory_recommendation(
    category: str,
    forecasted_units: int,
    current_stock: int = None,
    lead_time_days: int = None,
    safety_stock_days: int = 7,
    seasonality_strength: float = 0
) -> str:
    """
    Generate AI-powered inventory recommendations based on forecast.
    Enhanced with seasonality considerations.
    """
    
    daily_forecast = forecasted_units / 30
    safety_stock = daily_forecast * safety_stock_days
    
    # Adjust safety stock based on seasonality
    if seasonality_strength > 50:
        safety_stock *= 1.5
        seasonality_note = "⚠️ Strong seasonality detected - increased safety stock recommended"
    elif seasonality_strength > 25:
        safety_stock *= 1.2
        seasonality_note = "Moderate seasonality - slightly increased safety stock"
    else:
        seasonality_note = "Stable demand pattern"
    
    context = f"""
Category: {category}
Monthly Forecast: {forecasted_units} units
Daily Forecast: {daily_forecast:.1f} units
Recommended Safety Stock: {safety_stock:.0f} units
Seasonality: {seasonality_note}
"""
    
    if current_stock is not None:
        days_of_stock = current_stock / daily_forecast if daily_forecast > 0 else 0
        context += f"""Current Stock: {current_stock} units
Days of Stock: {days_of_stock:.1f} days
"""
    
    if lead_time_days is not None:
        reorder_point = daily_forecast * (lead_time_days + safety_stock_days)
        context += f"""Supplier Lead Time: {lead_time_days} days
Recommended Reorder Point: {reorder_point:.0f} units
"""

    prompt = f"""You are an inventory optimization expert.

{context}

Provide a brief (2-3 sentences) inventory recommendation:
1. Whether to reorder now or wait
2. Suggested order quantity if applicable  
3. Consider seasonality in your recommendation

Be specific with numbers."""

    try:
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=200,
            )
        )
        return response.text.strip()
    except Exception:
        return f"Based on forecast: Maintain safety stock of {safety_stock:.0f} units. {seasonality_note}. Review supply chain capacity."