"""
Performance Analysis and Visualization Utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from backtest_engine import BacktestResults


class PerformanceAnalyzer:
    """Comprehensive performance analysis and reporting"""
    
    @staticmethod
    def print_summary(results: BacktestResults, strategy_name: str = "Strategy") -> None:
        """Print detailed performance summary"""
        
        print("\n" + "="*60)
        print(f"  BACKTEST SUMMARY:  {strategy_name}")
        print("="*60)
        
        print("\n--- RETURNS ---")
        print(f"Total Return:         {results.performance_metrics['total_return']*100:>10.2f}%")
        print(f"Annual Return:       {results.performance_metrics['annual_return']*100:>10.2f}%")
        print(f"Annual Volatility:   {results.performance_metrics['annual_volatility']*100:>10.2f}%")
        
        print("\n--- RISK-ADJUSTED METRICS ---")
        print(f"Sharpe Ratio:        {results.sharpe_ratio: >10.4f}")
        print(f"Sortino Ratio:       {results.sortino_ratio:>10.4f}")
        print(f"Calmar Ratio:        {results.calmar_ratio:>10.4f}")
        
        print("\n--- DRAWDOWN ANALYSIS ---")
        print(f"Maximum Drawdown:    {results.max_drawdown*100:>10.2f}%")
        print(f"Drawdown Duration:   {PerformanceAnalyzer._calc_dd_duration(results.drawdown_series):>10} days")
        print(f"Recovery Time:       {PerformanceAnalyzer._calc_recovery_time(results.equity_curve):>10} days")
        
        print("\n--- TRADE STATISTICS ---")
        print(f"Total Trades:        {results.performance_metrics['num_trades']: >10.0f}")
        print(f"Win Rate:            {results. win_rate*100:>10.2f}%")
        print(f"Profit Factor:       {results.profit_factor:>10.4f}")
        
        if len(results.trades) > 0:
            print(f"Avg Trade Return:    {results.trades['pnl_pct'].mean()*100:>10.2f}%")
            print(f"Best Trade:           {results.trades['pnl_pct'].max()*100:>10.2f}%")
            print(f"Worst Trade:         {results.trades['pnl_pct'].min()*100:>10.2f}%")
        
        print("\n" + "="*60 + "\n")
    
    @staticmethod
    def plot_results(results: BacktestResults, title: str = "Backtest Results") -> None:
        """Plot equity curve, drawdown, and returns distribution"""
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Equity Curve
        axes[0]. plot(results.equity_curve. index, results.equity_curve. values, 
                     linewidth=2, color='blue', label='Equity Curve')
        axes[0].set_ylabel('Equity ($)', fontsize=11)
        axes[0].set_title(f'{title} - Equity Curve', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Drawdown
        axes[1].fill_between(results.drawdown_series.index, 
                            results.drawdown_series.values * 100, 0,
                            alpha=0.3, color='red', label='Drawdown')
        axes[1].set_ylabel('Drawdown (%)', fontsize=11)
        axes[1].set_title('Drawdown from Peak', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Returns Distribution
        axes[2].hist(results.returns. dropna() * 100, bins=50, 
                     color='green', alpha=0.7, edgecolor='black')
        axes[2].axvline(results.returns.mean() * 100, color='red', 
                       linestyle='--', linewidth=2, label='Mean Return')
        axes[2].set_xlabel('Daily Return (%)', fontsize=11)
        axes[2].set_ylabel('Frequency', fontsize=11)
        axes[2].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_trade_analysis(results: BacktestResults) -> None:
        """Analyze and plot trade results"""
        
        if len(results.trades) == 0:
            print("No trades to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Cumulative P&L
        cumulative_pnl = results. trades['pnl_amount']. cumsum()
        axes[0, 0].plot(range(len(cumulative_pnl)), cumulative_pnl. values, 
                        marker='o', markersize=4, linewidth=1)
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Cumulative P&L ($)')
        axes[0, 0].set_title('Cumulative P&L by Trade')
        axes[0, 0].grid(True, alpha=0.3)
        
        # P&L Distribution
        winning = results.trades[results.trades['pnl_pct'] > 0]['pnl_pct'] * 100
        losing = results. trades[results.trades['pnl_pct'] < 0]['pnl_pct'] * 100
        
        axes[0, 1].hist([winning, losing], bins=20, label=['Wins', 'Losses'],
                        color=['green', 'red'], alpha=0.7)
        axes[0, 1].set_xlabel('Return (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Trade P&L Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Win/Loss Ratio
        win_pct = (len(winning) / len(results.trades)) * 100
        loss_pct = (len(losing) / len(results.trades)) * 100
        axes[1, 0].pie([win_pct, loss_pct], labels=['Wins', 'Losses'],
                       colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Win Rate')
        
        # Trade Duration
        durations = (results.trades['exit_date'] - results.trades['entry_date']).dt.days
        axes[1, 1].hist(durations, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Duration (days)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Trade Duration Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _calc_dd_duration(drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration"""
        in_drawdown = drawdown_series < -0.001
        if not in_drawdown. any():
            return 0
        
        durations = []
        current_duration = 0
        
        for val in in_drawdown: 
            if val: 
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        return max(durations) if durations else 0
    
    @staticmethod
    def _calc_recovery_time(equity_curve:  pd.Series) -> int:
        """Calculate time to recover from maximum drawdown"""
        running_max = equity_curve.expanding().max()
        max_dd_date = (equity_curve - running_max).idxmin()
        
        recovery_equity = running_max.loc[max_dd_date]
        recovered = equity_curve[equity_curve. index > max_dd_date] >= recovery_equity
        
        if recovered.any():
            recovery_date = recovered.idxmax()
            return (recovery_date - max_dd_date).days
        
        return -1  # Not recovered


class StrategyComparison:
    """Compare multiple strategies"""
    
    @staticmethod
    def compare_strategies(results_dict: Dict[str, BacktestResults]) -> pd.DataFrame:
        """Create comparison table of multiple strategies"""
        
        comparison_data = []
        
        for strategy_name, results in results_dict.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{results.performance_metrics['total_return']*100:. 2f}%",
                'Annual Return': f"{results.performance_metrics['annual_return']*100:.2f}%",
                'Sharpe Ratio':  f"{results.sharpe_ratio:.4f}",
                'Sortino Ratio': f"{results. sortino_ratio:.4f}",
                'Max Drawdown': f"{results. max_drawdown*100:.2f}%",
                'Calmar Ratio': f"{results.calmar_ratio:.4f}",
                'Win Rate':  f"{results.win_rate*100:.2f}%",
                'Profit Factor': f"{results.profit_factor:.4f}",
                'Trades': f"{results.performance_metrics['num_trades']:. 0f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df