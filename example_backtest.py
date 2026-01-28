"""
Example usage of the backtesting framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest_engine import (
    BacktestEngine, 
    SimpleMovingAverageCrossover,
    RSIStrategy
)
from performance_analyzer import PerformanceAnalyzer, StrategyComparison


def generate_sample_data(days: int = 500, initial_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing"""
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movement using geometric Brownian motion
    returns = np.random.normal(0.0005, 0.02, days)
    close_prices = initial_price * np. exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': close_prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': close_prices * (1 + np.random.uniform(0, 0.02, days)),
        'Low': close_prices * (1 - np.random. uniform(0, 0.02, days)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    return data


def main():
    """Run example backtest"""
    
    print("\n" + "="*60)
    print("  EQUITY/FX BACKTESTING FRAMEWORK")
    print("="*60)
    
    # Generate sample data
    print("\nGenerating sample market data...")
    data = generate_sample_data(days=500)
    print(f"Data range: {data.index[0]. date()} to {data.index[-1].date()}")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000,
        risk_free_rate=0.02,
        slippage=0.0001,
        commission=0.0001
    )
    
    results_dict = {}
    
    # Test Strategy 1: SMA Crossover
    print("\n" + "-"*60)
    print("Running SMA Crossover Strategy (20/50)...")
    print("-"*60)
    sma_strategy = SimpleMovingAverageCrossover(fast_period=20, slow_period=50)
    sma_results = engine.run_backtest(sma_strategy, data)
    results_dict['SMA (20/50)'] = sma_results
    PerformanceAnalyzer.print_summary(sma_results, "SMA Crossover (20/50)")
    
    # Test Strategy 2: RSI Extremes
    print("\n" + "-"*60)
    print("Running RSI Strategy...")
    print("-"*60)
    rsi_strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    rsi_results = engine. run_backtest(rsi_strategy, data)
    results_dict['RSI (14)'] = rsi_results
    PerformanceAnalyzer. print_summary(rsi_results, "RSI Strategy")
    
    # Strategy comparison
    print("\n" + "="*60)
    print("  STRATEGY COMPARISON")
    print("="*60)
    comparison = StrategyComparison.compare_strategies(results_dict)
    print("\n", comparison. to_string(index=False))
    
    # Visualizations
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)
    
    print("\nPlotting SMA Strategy results...")
    PerformanceAnalyzer.plot_results(sma_results, "SMA Crossover (20/50)")
    
    print("Plotting SMA Strategy trade analysis...")
    PerformanceAnalyzer.plot_trade_analysis(sma_results)
    
    print("Plotting RSI Strategy results...")
    PerformanceAnalyzer.plot_results(rsi_results, "RSI Strategy")
    
    print("Plotting RSI Strategy trade analysis...")
    PerformanceAnalyzer.plot_trade_analysis(rsi_results)


if __name__ == "__main__":
    main()