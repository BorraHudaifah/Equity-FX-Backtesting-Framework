"""
Backtesting Engine for Equities and FX Strategies
Calculates performance metrics:  Sharpe Ratio, Maximum Drawdown, Calmar Ratio, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BacktestResults:
    """Container for backtest results and performance metrics"""
    returns: pd.Series
    equity_curve: pd.Series
    trades: pd.DataFrame
    performance_metrics: Dict[str, float]
    drawdown_series: pd.Series
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    
    def __repr__(self) -> str:
        metrics_str = "\n".join(
            f"  {key}: {value:.4f}" 
            for key, value in self.performance_metrics.items()
        )
        return f"""
BacktestResults:
{metrics_str}
"""


class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals:  1 (long), -1 (short), 0 (neutral)
        """
        pass


class BacktestEngine:
    """Core backtesting engine with comprehensive performance analysis"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.02,
        slippage: float = 0.0001,
        commission:  float = 0.0001
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital:  Starting capital in currency units
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
            slippage:  Slippage as percentage of price (0.0001 = 0.01%)
            commission: Commission as percentage of trade value
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.slippage = slippage
        self.commission = commission
        
    def run_backtest(
        self,
        strategy: TradingStrategy,
        data: pd.DataFrame,
        entry_price_col: str = 'Close',
        position_sizing:  str = 'fixed',
        trade_size: float = 0.95
    ) -> BacktestResults: 
        """
        Run backtest on historical data
        
        Args: 
            strategy: Trading strategy object implementing generate_signals()
            data: DataFrame with OHLCV data, must include entry_price_col
            entry_price_col: Column name to use for entry prices
            position_sizing:  'fixed' (fixed % of capital) or 'dynamic' (Kelly-like)
            trade_size:  Fraction of capital to risk per trade (0.95 = 95%)
            
        Returns:
            BacktestResults object with full analysis
        """
        # Generate signals
        signals = strategy.generate_signals(data. copy())
        
        # Initialize tracking variables
        positions = pd.Series(0.0, index=data.index)
        equity = pd.Series(self.initial_capital, index=data.index)
        trades_list = []
        
        current_position = 0
        entry_price = 0
        entry_date = None
        
        for i in range(1, len(data)):
            signal = signals. iloc[i]
            date = data.index[i]
            price = data[entry_price_col].iloc[i]
            
            # Position changes
            if signal != 0 and current_position == 0:
                # Entry
                current_position = signal
                entry_price = price * (1 + self.slippage * signal)
                entry_date = date
                
            elif signal == 0 and current_position != 0:
                # Exit
                exit_price = price * (1 - self.slippage * abs(current_position))
                
                # Calculate trade P&L
                if current_position > 0:
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price
                
                # Apply commission
                pnl -= self.commission * 2
                
                # Store trade
                trades_list.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': 'LONG' if current_position > 0 else 'SHORT',
                    'pnl_pct': pnl,
                    'pnl_amount': equity.iloc[i-1] * pnl
                })
                
                # Update equity
                equity.iloc[i] = equity.iloc[i-1] * (1 + pnl)
                current_position = 0
            
            positions.iloc[i] = current_position
            
            # Update equity for open positions
            if current_position != 0:
                unrealized_pnl = (price - entry_price) / entry_price
                if current_position < 0:
                    unrealized_pnl = -unrealized_pnl
                equity.iloc[i] = equity.iloc[i-1] * (1 + unrealized_pnl)
        
        # Close final position if open
        if current_position != 0:
            final_price = data[entry_price_col].iloc[-1]
            if current_position > 0:
                pnl = (final_price - entry_price) / entry_price
            else:
                pnl = (entry_price - final_price) / entry_price
            pnl -= self.commission * 2
            equity.iloc[-1] = equity.iloc[-2] * (1 + pnl)
            trades_list.append({
                'entry_date': entry_date,
                'exit_date': data.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'direction':  'LONG' if current_position > 0 else 'SHORT',
                'pnl_pct': pnl,
                'pnl_amount': equity.iloc[-2] * pnl
            })
        
        # Calculate returns and metrics
        returns = equity.pct_change()
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(equity, returns, trades_df)
        
        # Calculate drawdown
        drawdown = self._calculate_drawdown(equity)
        
        return BacktestResults(
            returns=returns,
            equity_curve=equity,
            trades=trades_df,
            performance_metrics=metrics,
            drawdown_series=drawdown,
            max_drawdown=drawdown.min(),
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor']
        )
    
    def _calculate_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        total_return = (equity.iloc[-1] - equity. iloc[0]) / equity.iloc[0]
        num_days = (equity.index[-1] - equity.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(num_days, 1)) - 1
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        sharpe = 0 if np.isnan(sharpe) else sharpe
        
        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        sortino = excess_returns. mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        sortino = 0 if np.isnan(sortino) else sortino
        
        # Maximum Drawdown
        running_max = equity.expanding().max()
        max_dd = ((equity - running_max) / running_max).min()
        
        # Calmar Ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Trade Statistics
        win_rate = 0
        profit_factor = 0
        
        if len(trades) > 0:
            winning_trades = trades[trades['pnl_amount'] > 0]
            losing_trades = trades[trades['pnl_amount'] < 0]
            
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            
            gross_profit = winning_trades['pnl_amount'].sum()
            gross_loss = abs(losing_trades['pnl_amount'].sum())
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio':  calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(trades)
        }
    
    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown from peak"""
        running_max = equity. expanding().max()
        drawdown = (equity - running_max) / running_max
        return drawdown


class SimpleMovingAverageCrossover(TradingStrategy):
    """Simple SMA Crossover strategy"""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data:  pd.DataFrame) -> pd.Series:
        """Generate crossover signals"""
        data['sma_fast'] = data['Close'].rolling(self.fast_period).mean()
        data['sma_slow'] = data['Close'].rolling(self.slow_period).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[data['sma_fast'] > data['sma_slow']] = 1
        signals[data['sma_fast'] < data['sma_slow']] = -1
        
        return signals


class RSIStrategy(TradingStrategy):
    """RSI-based overbought/oversold strategy"""
    
    def __init__(self, period: int = 14, overbought:  int = 70, oversold:  int = 30):
        self.period = period
        self. overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI signals"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1
        signals[rsi > self.overbought] = -1
        
        return signals