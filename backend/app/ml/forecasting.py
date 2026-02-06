"""
Forecasting service using Facebook Prophet for time series predictions.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import json

logger = logging.getLogger(__name__)


class ForecastingService:
    """
    Time series forecasting using Facebook Prophet.

    Features:
    - Automatic seasonality detection
    - Confidence intervals
    - Model caching (avoid retraining)
    - Breakdown by dimensions (region, product)
    - Goal tracking and on-track analysis
    """

    def __init__(self):
        """Initialize forecasting service with cache."""
        self.model_cache = {}  # Cache trained models
        self.cache_ttl = 3600  # Cache for 1 hour
        logger.info("ForecastingService initialized")

    def forecast_metric(
        self,
        historical_data: List[Dict],
        periods_ahead: int,
        granularity: str = "month",
        breakdown_by: Optional[str] = None,
        metric_name: str = "volume"
    ) -> Dict:
        """
        Generate forecast for a metric.

        Args:
            historical_data: List of dicts with 'date' and 'value' keys
            periods_ahead: Number of periods to forecast
            granularity: 'day', 'week', 'month', 'quarter'
            breakdown_by: Optional dimension for separate forecasts (e.g., 'region')
            metric_name: Name of metric being forecast (for display)

        Returns:
            Dict with forecast results and metadata
        """
        logger.info(f"[FORECAST] Starting forecast: {metric_name}, {periods_ahead} {granularity}s ahead")

        try:
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)

            # Validate data
            if len(df) < 10:
                return {
                    "success": False,
                    "error": "Insufficient historical data. Need at least 10 data points for forecasting."
                }

            # Check if breakdown requested
            if breakdown_by and breakdown_by in df.columns:
                return self._forecast_by_dimension(
                    df, breakdown_by, periods_ahead, granularity, metric_name
                )
            else:
                return self._forecast_single(
                    df, periods_ahead, granularity, metric_name
                )

        except Exception as e:
            logger.error(f"[FORECAST] Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _forecast_single(
        self,
        df: pd.DataFrame,
        periods_ahead: int,
        granularity: str,
        metric_name: str
    ) -> Dict:
        """Forecast single time series."""
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df['value'].astype(float)
        })

        # Sort by date
        prophet_df = prophet_df.sort_values('ds')

        # Remove duplicates (aggregate if multiple values per date)
        prophet_df = prophet_df.groupby('ds').agg({'y': 'sum'}).reset_index()

        logger.info(f"[FORECAST] Training on {len(prophet_df)} historical points")

        # Calculate trend metrics before modeling
        trend_analysis = self._analyze_trend(prophet_df)
        logger.info(f"[FORECAST] Trend analysis: YoY growth {trend_analysis['yoy_growth_pct']:.1f}%, Recent trend: {trend_analysis['recent_trend']}")

        # Adjust changepoint prior based on detected trend
        # If business is stable (low growth), be more aggressive in detecting recent changes
        if abs(trend_analysis['yoy_growth_pct']) < 3:
            changepoint_prior = 0.20  # More sensitive to recent changes
            logger.info(f"[FORECAST] Using higher changepoint sensitivity for stable business")
        else:
            changepoint_prior = 0.15  # Standard sensitivity

        # Train Prophet model with better trend detection
        model = Prophet(
            yearly_seasonality=True if len(prophet_df) >= 365 else False,
            weekly_seasonality=True if granularity == 'day' else False,
            daily_seasonality=False,
            interval_width=0.80,  # 80% confidence interval
            changepoint_prior_scale=changepoint_prior,  # Adaptive based on trend
            seasonality_mode='multiplicative',  # Better for business data with growth
            growth='linear',  # Explicitly model linear trend
            n_changepoints=25  # More changepoints to detect trend shifts
        )

        # Suppress Prophet's verbose output
        import logging as prophet_logging
        prophet_logging.getLogger('prophet').setLevel(prophet_logging.WARNING)
        prophet_logging.getLogger('cmdstanpy').setLevel(prophet_logging.WARNING)

        try:
            model.fit(prophet_df, verbose=False)
        except AttributeError as e:
            # Fallback if stan_backend issue
            logger.warning(f"[FORECAST] Prophet backend error, trying without verbose: {e}")
            model.fit(prophet_df)

        # Generate future dates
        future = model.make_future_dataframe(
            periods=periods_ahead,
            freq=self._get_freq(granularity)
        )

        # Make predictions
        forecast = model.predict(future)

        # Extract forecast results (only future periods)
        historical_len = len(prophet_df)
        forecast_results = forecast.iloc[historical_len:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Calculate accuracy metrics on historical data
        accuracy_metrics = self._calculate_accuracy(prophet_df, forecast.iloc[:historical_len])

        # Format results
        forecast_data = []
        for _, row in forecast_results.iterrows():
            forecast_data.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'forecast': round(float(row['yhat']), 2),
                'lower_bound': round(float(row['yhat_lower']), 2),
                'upper_bound': round(float(row['yhat_upper']), 2)
            })

        # Calculate total forecast
        total_forecast = sum([f['forecast'] for f in forecast_data])

        logger.info(f"[FORECAST] Complete. Total forecast: {total_forecast:.2f}")

        return {
            "success": True,
            "metric": metric_name,
            "granularity": granularity,
            "periods_ahead": periods_ahead,
            "forecast": forecast_data,
            "total_forecast": round(total_forecast, 2),
            "accuracy_metrics": accuracy_metrics,
            "trend_analysis": trend_analysis,
            "historical_data_points": len(prophet_df),
            "confidence_level": "80%"
        }

    def _forecast_by_dimension(
        self,
        df: pd.DataFrame,
        dimension: str,
        periods_ahead: int,
        granularity: str,
        metric_name: str
    ) -> Dict:
        """Forecast separately for each dimension value."""
        logger.info(f"[FORECAST] Forecasting by {dimension}")

        dimension_forecasts = {}
        dimension_values = df[dimension].unique()

        for dim_value in dimension_values:
            # Filter data for this dimension
            dim_df = df[df[dimension] == dim_value][['date', 'value']]

            if len(dim_df) < 10:
                logger.warning(f"[FORECAST] Skipping {dim_value}: insufficient data ({len(dim_df)} points)")
                continue

            # Forecast for this dimension
            result = self._forecast_single(dim_df, periods_ahead, granularity, metric_name)

            if result['success']:
                dimension_forecasts[str(dim_value)] = result['forecast']

        return {
            "success": True,
            "metric": metric_name,
            "granularity": granularity,
            "periods_ahead": periods_ahead,
            "breakdown_by": dimension,
            "forecasts": dimension_forecasts
        }

    def analyze_goal_tracking(
        self,
        current_value: float,
        target_value: float,
        forecast_final: float,
        periods_total: int,
        periods_elapsed: int,
        metric_name: str = "volume"
    ) -> Dict:
        """
        Analyze progress toward goal and forecast likelihood of hitting target.

        Args:
            current_value: Current YTD value
            target_value: Goal/target value
            forecast_final: Forecasted final value
            periods_total: Total periods (e.g., 12 months)
            periods_elapsed: Periods completed so far
            metric_name: Name of metric

        Returns:
            Goal tracking analysis
        """
        # Calculate metrics
        progress_pct = (current_value / target_value * 100) if target_value > 0 else 0
        expected_progress_pct = (periods_elapsed / periods_total * 100) if periods_total > 0 else 0
        forecast_achievement_pct = (forecast_final / target_value * 100) if target_value > 0 else 0

        gap = target_value - current_value
        forecast_gap = target_value - forecast_final

        # Determine status
        if forecast_achievement_pct >= 100:
            status = "on_track"
            message = f"On track to exceed target by {forecast_final - target_value:.2f}"
        elif forecast_achievement_pct >= 95:
            status = "likely"
            message = f"Likely to hit target (projected {forecast_achievement_pct:.1f}%)"
        elif forecast_achievement_pct >= 85:
            status = "at_risk"
            message = f"At risk: projected {forecast_achievement_pct:.1f}% of target"
        else:
            status = "off_track"
            message = f"Off track: projected shortfall of {abs(forecast_gap):.2f}"

        # Calculate required run rate
        periods_remaining = periods_total - periods_elapsed
        if periods_remaining > 0:
            current_run_rate = current_value / periods_elapsed if periods_elapsed > 0 else 0
            required_run_rate = gap / periods_remaining
            run_rate_increase_needed = ((required_run_rate / current_run_rate) - 1) * 100 if current_run_rate > 0 else 0
        else:
            current_run_rate = 0
            required_run_rate = 0
            run_rate_increase_needed = 0

        return {
            "metric": metric_name,
            "current_value": round(current_value, 2),
            "target_value": round(target_value, 2),
            "forecast_final": round(forecast_final, 2),
            "progress_pct": round(progress_pct, 2),
            "expected_progress_pct": round(expected_progress_pct, 2),
            "forecast_achievement_pct": round(forecast_achievement_pct, 2),
            "status": status,
            "message": message,
            "gap": round(gap, 2),
            "forecast_gap": round(forecast_gap, 2),
            "periods_elapsed": periods_elapsed,
            "periods_remaining": periods_remaining,
            "current_run_rate": round(current_run_rate, 2),
            "required_run_rate": round(required_run_rate, 2),
            "run_rate_increase_pct": round(run_rate_increase_needed, 2)
        }

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        Analyze historical trend to understand growth patterns.

        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns

        Returns:
            Dict with trend metrics
        """
        try:
            if len(df) < 24:  # Need at least 2 years for YoY
                return {
                    "yoy_growth_pct": 0,
                    "recent_trend": "insufficient_data",
                    "is_growing": False
                }

            # Calculate YoY growth (compare last 12 months to previous 12 months)
            sorted_df = df.sort_values('ds')

            # Split into recent vs previous year
            mid_point = len(sorted_df) // 2
            older_half = sorted_df.iloc[:mid_point]['y'].sum()
            recent_half = sorted_df.iloc[mid_point:]['y'].sum()

            yoy_growth_pct = ((recent_half - older_half) / older_half * 100) if older_half > 0 else 0

            # Recent trend (last 6 data points vs previous 6)
            if len(sorted_df) >= 12:
                last_6 = sorted_df.iloc[-6:]['y'].mean()
                prev_6 = sorted_df.iloc[-12:-6]['y'].mean()
                recent_momentum = ((last_6 - prev_6) / prev_6 * 100) if prev_6 > 0 else 0

                if recent_momentum > 5:
                    recent_trend = "accelerating"
                elif recent_momentum < -5:
                    recent_trend = "decelerating"
                else:
                    recent_trend = "stable"
            else:
                recent_momentum = 0
                recent_trend = "stable"

            return {
                "yoy_growth_pct": round(yoy_growth_pct, 2),
                "recent_momentum_pct": round(recent_momentum, 2),
                "recent_trend": recent_trend,
                "is_growing": yoy_growth_pct > 1
            }

        except Exception as e:
            logger.error(f"[FORECAST] Trend analysis error: {e}")
            return {
                "yoy_growth_pct": 0,
                "recent_trend": "error",
                "is_growing": False
            }

    def _calculate_accuracy(
        self,
        actual_df: pd.DataFrame,
        forecast_df: pd.DataFrame
    ) -> Dict:
        """Calculate accuracy metrics for historical predictions."""
        try:
            # Merge actual and forecast
            merged = actual_df.merge(
                forecast_df[['ds', 'yhat']],
                on='ds',
                how='inner'
            )

            if len(merged) == 0:
                return {"error": "No overlapping data for accuracy calculation"}

            # Calculate metrics
            actual = merged['y'].values
            predicted = merged['yhat'].values

            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(actual - predicted))

            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))

            return {
                "mape": round(float(mape), 2),
                "mae": round(float(mae), 2),
                "rmse": round(float(rmse), 2),
                "sample_size": len(merged)
            }

        except Exception as e:
            logger.error(f"[FORECAST] Accuracy calculation error: {e}")
            return {"error": str(e)}

    def _get_freq(self, granularity: str) -> str:
        """Get pandas frequency string from granularity."""
        freq_map = {
            'day': 'D',
            'week': 'W',
            'month': 'MS',  # Month start
            'quarter': 'QS'  # Quarter start
        }
        return freq_map.get(granularity, 'MS')


# Global instance
forecasting_service = ForecastingService()
