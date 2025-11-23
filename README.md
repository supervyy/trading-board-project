# trading-board-project

### Problem Definition:
**Target**

Prediction of trend direction over next t=[5, 10, 15, 20, 30, 60, 120] minutes for Invesco QQQ ETF (QQQ), using NVIDIA as predictive factor.
For every minute from 2020-01-01 until 2025-06-25 we calculate the target as the normalized linear regression slope of QQQ prices over future window t, while using current features from both QQQ and NVDA as input predictors.

**Input Features**

QQQ Technical Features:
- Normalized VWAP (volume weighted average price) and volume
- Normalized exponential moving average (EMA) over t=[5, 10, 15, 20, 30, 60, 120]
- Slope and second order slope of EMAs
- Short- and mid-term returns (1, 5, 10, 20 minutes)

NVDA Cross-Asset Features:
- Normalized VWAP and volume
- NVDA EMAs over identical time horizons
- NVDA short-term and mid-term returns
- NVDA EMA slopes

QQQ-NVDA Relationship Features:

- Rolling correlations (returns, EMAs, momentum)
- Relative Strength Ratio (QQQ price / NVDA price)
- Return Spread (QQQ return - NVDA return)
- NVDA short-term momentum as lead-indicator

### Procedure Overview:
- Collects minute bars for QQQ and NVDA from 2020-01-01 → 2025-07-24
- Engineers above features for both assets and their relationships for each minute
- Predicts direction of QQQ trend over next 30 minutes (entry-network) using feed-forward neural network ([128, 64] hidden layers, dropout 0.1, ReLU activation)
- Use decision tree (depth=10) with embeddings (hidden layer with 64 neurons) to predict entry points with positive trend direction
- Implement trading strategy in Alpaca, that enters QQQ positions at predicted entry points and holds them for 30 minutes