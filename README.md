# Equity/FX Backtesting Framework

A comprehensive Python backtesting framework for testing trading strategies on equities and FX markets with detailed risk metrics and performance analysis.

## Features

### Core Backtesting Engine
- **Strategy Implementation**: Abstract base class for custom strategy development
- **Position Management**: Long/short position tracking
- **Transaction Costs**: Configurable slippage and commission modeling
- **Trade Logging**: Complete trade history with entry/exit details

### Performance Metrics

#### Risk-Adjusted Returns
- **Sharpe Ratio**: Return per unit of total risk
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: Return per unit of maximum drawdown

#### Drawdown Analysis
- **Maximum Drawdown**: Peak-to-trough decline
- **Drawdown Duration**: Length of drawdown period
- **Recovery Time**: Time to recover from maximum drawdown

#### Trade Statistics
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Trade P&L Distribution**: Statistical analysis of trade outcomes
- **Trade Duration**: Average holding period

## Usage

### Basic Example

```python
from backtest_engine import BacktestEngine, SimpleMovingAverageCrossover
import pandas as pd

# Load your market data
data = pd.read_csv('market_data.csv', index_col=0, parse_dates=True)

# Initialize engine
engine = BacktestEngine(
    initial_capital=100000,
    risk_free_rate=0.02,
    slippage=0.0001,
    commission=0.0001
)

# Run strategy
strategy = SimpleMovingAverageCrossover(fast_period=20, slow_period=50)
results = engine.run_backtest(strategy, data)

# Analyze results
from performance_analyzer import PerformanceAnalyzer
PerformanceAnalyzer.print_summary(results, "My Strategy")
PerformanceAnalyzer.plot_results(results)