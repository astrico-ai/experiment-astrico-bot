# 🚀 Forecasting Feature - Setup & Usage Guide

## ✅ Installation

### 1. Install Dependencies

```bash
cd /Users/sanujphilip/Desktop/anomaly/backend
pip install -r requirements.txt
```

**New packages added:**
- `prophet==1.1.5` - Facebook's forecasting library
- `pandas==2.1.3` - Data manipulation
- `numpy==1.26.2` - Numerical computing
- `scikit-learn==1.3.2` - ML utilities

**Note:** Prophet may take 2-3 minutes to install (it compiles C++ extensions).

### 2. Restart Backend

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 3. Refresh Frontend

Just refresh your browser (Ctrl+R or Cmd+R).

---

## 🎯 How to Use

### **Example 1: Simple Forecast**

**Query:** "What will volume be next month?"

**What happens:**
- LLM calls `forecast_metric(metric="volume", periods_ahead=1, granularity="month")`
- Backend fetches 3 years of historical data
- Prophet trains model and predicts
- Response shows forecast + confidence interval

**Expected response:**
```
📈 **Forecast: VOLUME** (Next 1 month)

**2026-03-01:** 1,250.50 KL
  Range: 1,180.00 - 1,320.00 KL (80% confidence)

**Total Forecast:** 1,250.50 KL

📊 **Model Accuracy:** ±8.5% (based on 36 historical points)
```

---

### **Example 2: Goal Tracking**

**Query:** "Are we on track to hit our FY25 target of 15,000 KL?"

**What happens:**
- Gets current YTD volume
- Forecasts remaining months
- Compares projected final vs target
- Analyzes on-track status

**Expected response:**
```
✅ **Goal Tracking Analysis:**

Current YTD: 10,200.00 KL
Target: 15,000.00 KL
Projected Final: 12,800.00 KL
Expected Achievement: 85.3%

**Status:** At risk: projected 85.3% of target

To hit target, you need to average 2,400.00 KL/month for the remaining 2 months
(vs. current rate of 1,020.00 KL/month).
```

---

### **Example 3: Regional Breakdown**

**Query:** "Forecast volume by region for next quarter"

**What happens:**
- Trains separate model for each region
- Generates 3-month forecast per region
- Shows growth trends

**Expected response:**
```
📈 **Forecast: VOLUME by REGION** (Next 3 months)

**North:**
  - 2026-03-01: 450.00 KL (Range: 420.00 - 480.00)
  - 2026-04-01: 480.00 KL (Range: 445.00 - 515.00)
  - 2026-05-01: 500.00 KL (Range: 460.00 - 540.00)
  Total: 1,430.00 KL

**South:**
  - 2026-03-01: 380.00 KL (Range: 350.00 - 410.00)
  ...
```

---

### **Example 4: Proactive Forecasting**

**Query:** "Show me volume trend for FY25"

**What happens (if enhanced):**
- Shows historical data
- Automatically extends with forecast for remaining months
- User sees complete picture: past + future

---

## 🔍 Testing Checklist

### Phase 1: Basic Forecasting ✅
- [ ] "What will volume be next month?" → Gets single month forecast
- [ ] "Forecast volume for next 3 months" → Gets 3-month forecast
- [ ] "Predict revenue for Q4" → Works with revenue metric
- [ ] Check logs show: `[FORECAST] Starting forecast...` and model training

### Phase 2: Enhanced Features ✅
- [ ] "Are we on track to hit 15,000 KL?" → Goal tracking works
- [ ] "Forecast volume by region" → Regional breakdowns work
- [ ] Confidence intervals show in response
- [ ] Historical accuracy metrics appear

### Phase 3: Edge Cases ✅
- [ ] New product with <10 data points → Shows "insufficient data" error
- [ ] Very long forecast (12+ months) → Works but wider confidence interval
- [ ] Invalid metric → Handles gracefully

---

## 🐛 Troubleshooting

### **Error: "Prophet not found"**
```bash
# Prophet requires specific dependencies on Mac
brew install cmake
pip install prophet
```

### **Error: "Insufficient historical data"**
- Need at least 10 data points for forecasting
- Check if metric has enough history:
```sql
SELECT COUNT(DISTINCT DATE_TRUNC('month', billing_date))
FROM sales_invoices_veedol;
```

### **Forecast seems wrong**
- Check historical data quality (missing months? outliers?)
- Prophet assumes stationarity - works best with consistent patterns
- View logs: `[FORECAST] Model Accuracy: ±X%` to see confidence

### **Too slow (>10 seconds)**
- First run trains model (5-8 seconds)
- Subsequent runs should be faster (model cached)
- Consider reducing historical data window from 3 years to 2 years

---

## 📊 Architecture

```
User Query: "What will volume be next month?"
    ↓
LLM (Gemini) detects forecast intent
    ↓
Calls: forecast_metric(metric="volume", periods_ahead=1)
    ↓
Backend ConversationManager._forecast_metric_tool():
  1. Fetches 3 years historical data via SQL
  2. Calls ForecastingService.forecast_metric()
  3. Prophet trains model on historical data
  4. Prophet generates prediction + confidence interval
  5. Formats results with goal tracking (if target provided)
    ↓
Returns formatted response to LLM
    ↓
LLM formats in natural language
    ↓
User sees forecast
```

---

## 🎓 Understanding Prophet

**What it does:**
- Decomposes time series into: Trend + Seasonality + Holidays
- Automatically detects patterns (weekly, monthly, yearly)
- Provides confidence intervals (uncertainty quantification)

**Best for:**
- Business data with clear seasonality
- Monthly/quarterly forecasts
- Data with some missing points or outliers

**Not ideal for:**
- Very noisy data
- Sudden regime changes
- <1 year of history

---

## 🚀 Next Steps

### **Quick Wins:**
1. Test with real queries
2. Verify accuracy on known data
3. Share with team for feedback

### **Enhancements (Future):**
1. **Visual charts** - Show forecast as line graph
2. **Model caching** - Pre-train models daily for instant predictions
3. **Ensemble models** - Combine Prophet + ARIMA + Linear for better accuracy
4. **Custom seasonality** - Add business-specific patterns (Diwali, holidays)
5. **Scenario analysis** - "What if North grows 10% faster?"

---

## 📞 Support

If you encounter issues:
1. Check backend logs: Look for `[FORECAST]` messages
2. Verify SQL query executes: Check `[SQL]` logs
3. Test Prophet directly:
```python
from app.ml.forecasting import forecasting_service
result = forecasting_service.forecast_metric(
    historical_data=[{"date": "2025-01-01", "value": 100}, ...],
    periods_ahead=3,
    granularity="month"
)
print(result)
```

---

## ✨ You're All Set!

Forecasting is now live. Try asking:
- "What will volume be next month?"
- "Are we on track to hit our target?"
- "Forecast revenue for Q4 by region"

**Have fun predicting the future! 🔮**
