# -*- coding: utf-8 -*-
"""
Pairs Trading Bot with Machine Learning
=======================================
A sophisticated pairs trading algorithm that uses cointegration analysis
and machine learning to trade correlated currency pairs on MetaTrader 5.
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import pytz
import pandas_ta as ta
from statsmodels.tsa.stattools import coint
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import MetaTrader5 as mt5

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Trading configuration parameters"""
    # Trading pair
    PRIMARY_SYMBOL = 'GBPUSD'
    SECONDARY_SYMBOL = 'EURUSD'  # or 'BTCUSD' / 'ETHUSD'
    
    # Machine learning settings
    USE_ML = True  # False = use cointegration, True = use ML
    
    # Trading parameters
    LOT_SIZE = 3.0
    PROFIT_TARGET = 20.9
    STOP_LOSS = -198.9
    
    # Signal thresholds
    if USE_ML:
        SHORT_THRESHOLD = -0.1   # Below this = short spread
        LONG_THRESHOLD = 0.1      # Above this = long spread
    else:
        SHORT_THRESHOLD = -9.0e-5
        LONG_THRESHOLD = 9.0e-5
    
    # Maximum allowed spread
    MAX_SPREAD = 10
    
    # MT5 settings
    MAGIC_NUMBER = 234000
    DEVIATION = 20
    
    # Paths (adjust as needed)
    MT5_TERMINAL_PATH = "C:/Program Files/FxPro - MetaTrader 5/terminal64.exe"

# ============================================================================
# MT5 CONNECTION
# ============================================================================

class MT5Connector:
    """Handles MetaTrader 5 connection"""
    
    @staticmethod
    def initialize():
        """Initialize connection to MT5"""
        # Get credentials from environment variables
        account = int(os.getenv('MT5_ACCOUNT', 0))
        password = os.getenv('MT5_PASSWORD', '')
        server = os.getenv('MT5_SERVER', '')
        
        # Initialize MT5
        if not mt5.initialize(Config.MT5_TERMINAL_PATH):
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # Login
        authorized = mt5.login(account, password=password, server=server)
        if not authorized:
            print(f"Login failed: {mt5.last_error()}")
            return False
        
        print("✓ Connected to MetaTrader 5")
        return True
    
    @staticmethod
    def shutdown():
        """Shutdown MT5 connection"""
        mt5.shutdown()
        print("✓ MT5 connection closed")


# ============================================================================
# DATA PROCESSING
# ============================================================================

class ParticleFilter:
    """Particle filter for noise reduction in price data"""
    
    @staticmethod
    def apply(prices, num_particles=20):
        """
        Apply particle filter to price series
        
        Args:
            prices: Array of price values
            num_particles: Number of particles to use
            
        Returns:
            Filtered price series
        """
        def state_space_model(x, v):
            return x + v
        
        # Initialize particles
        initial_x = np.random.uniform(np.average(prices), np.std(prices), num_particles)
        initial_v = np.random.normal(np.average(prices), np.std(prices), num_particles)
        particles = np.vstack([initial_x, initial_v]).T
        
        def propagate_particles(particles):
            v = np.random.normal(0, 1, particles.shape[0])
            x = state_space_model(particles[:, 0], v)
            return np.vstack([x, v]).T
        
        def update_weights(particles, observation):
            likelihood = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (observation - particles[:, 0]) ** 2)
            return likelihood / np.sum(likelihood)
        
        def resample_particles(particles, weights):
            indices = np.random.choice(particles.shape[0], particles.shape[0], p=weights)
            return particles[indices]
        
        filtered_prices = []
        for i in range(len(prices)):
            particles = propagate_particles(particles)
            weights = update_weights(particles, prices[i])
            particles = resample_particles(particles, weights)
            filtered_prices.append(np.average(particles[:, 0], weights=weights))
        
        return filtered_prices


class DataFetcher:
    """Fetches and processes market data from MT5"""
    
    @staticmethod
    def get_rates(symbol, rate_type=1):
        """
        Fetch rates for a symbol
        
        Args:
            symbol: Trading symbol
            rate_type: 1 = tick data, 2 = OHLC bars
            
        Returns:
            DataFrame with processed rates and point value
        """
        point = mt5.symbol_info(symbol).point
        
        # Fetch data based on type
        if rate_type == 2:
            # OHLC bars
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 500)
            rates_frame = pd.DataFrame(rates)
        else:
            # Tick data
            utc_to = datetime.datetime.fromtimestamp(mt5.symbol_info(symbol).time)
            utc_from = utc_to - datetime.timedelta(days=0.01)
            rates = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
            rates_frame = pd.DataFrame(rates)
            rates_frame['close'] = (rates_frame['ask'] + rates_frame['bid']) / 2
        
        # Calculate returns
        rates_frame['log_return'] = np.log(rates_frame['close']).diff()
        rates_frame = rates_frame.dropna()
        
        # Apply particle filter to returns
        returns_list = rates_frame['log_return'].values.tolist()
        rates_frame['returns'] = ParticleFilter.apply(returns_list)
        
        # Format timestamp
        if rate_type == 1:
            rates_frame['time'] = pd.to_datetime(rates_frame['time_msc'], unit='ms')
        else:
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        
        return rates_frame, point


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

class SignalGenerator:
    """Generates trading signals using cointegration or machine learning"""
    
    def __init__(self, use_ml=True):
        self.use_ml = use_ml
        
    def _add_technical_indicators(self, df):
        """Add technical indicators to dataframe"""
        df = df.copy()
        df.ta.bbands(length=20, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.rsi(length=14, append=True)
        df['res'] = df['close'].pct_change(periods=100)
        return df.dropna()
    
    def _train_ml_model(self, X, y):
        """Train machine learning model"""
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, shuffle=False
        )
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # model = xgb.XGBRegressor()
        model = MLPRegressor(solver='lbfgs')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        return y_pred[-1]  # Return last prediction
    
    def generate_signal(self, df1, df2):
        """
        Generate trading signal for the pair
        
        Args:
            df1: Data for first symbol
            df2: Data for second symbol
            
        Returns:
            signal_value, prediction (if applicable)
        """
        if not self.use_ml:
            # Cointegration-based signal
            spread = df1['log_return'] - df2['log_return']
            spread_ma = spread.rolling(100).mean()
            signal = (spread - spread_ma).iloc[-1]
            return signal, 0
        
        else:
            # Machine learning signal
            # Add indicators to both dataframes
            df1_indicators = self._add_technical_indicators(df1)
            df2_indicators = self._add_technical_indicators(df2)
            
            # Prepare features and targets
            df1_features = df1_indicators.drop(['time', 'res'], axis=1)
            df2_features = df2_indicators.drop(['time', 'res'], axis=1)
            
            df1_target = df1_indicators['res']
            df2_target = df2_indicators['res']
            
            # Train models and get predictions
            pred1 = self._train_ml_model(df1_features, df1_target)
            pred2 = self._train_ml_model(df2_features, df2_target)
            
            signal = pred2 - pred1
            return signal, 0


# ============================================================================
# TRADE EXECUTION
# ============================================================================

class TradeExecutor:
    """Handles trade execution and management"""
    
    def __init__(self):
        self.trade_log = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'pred_signal', 'profit', 
            'drawUP', 'drawdown', 'trade_type'
        ])
    
    def place_order(self, symbol, order_type):
        """
        Place a market order
        
        Args:
            symbol: Trading symbol
            order_type: 1 = BUY, 2 = SELL
            
        Returns:
            Order result
        """
        tick = mt5.symbol_info_tick(symbol)
        point = mt5.symbol_info(symbol).point
        
        # Ensure symbol is visible
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        
        if order_type == 1:  # BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": Config.LOT_SIZE,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "sl": 0.0,
                "tp": 0.0,
                "deviation": Config.DEVIATION,
                "magic": Config.MAGIC_NUMBER,
                "comment": "ML Pairs Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        else:  # SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": Config.LOT_SIZE,
                "type": mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "sl": 0.0,
                "tp": 0.0,
                "deviation": Config.DEVIATION,
                "magic": Config.MAGIC_NUMBER,
                "comment": "ML Pairs Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        
        return mt5.order_send(request)
    
    def close_position(self, position):
        """Close an open position"""
        tick = mt5.symbol_info_tick(position.symbol)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if position.type == 1 else tick.bid,
            "deviation": Config.DEVIATION,
            "magic": Config.MAGIC_NUMBER,
            "comment": "Close ML Pairs Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        return mt5.order_send(request)
    
    def log_trade(self, entry_time, exit_time, signal, profit, drawup, drawdown, trade_type):
        """Log trade details"""
        new_row = pd.DataFrame([{
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pred_signal': signal,
            'profit': profit,
            'drawUP': drawup,
            'drawdown': drawdown,
            'trade_type': trade_type
        }])
        self.trade_log = pd.concat([self.trade_log, new_row], ignore_index=True)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"TRADE CLOSED: {trade_type}")
        print(f"Profit: {profit:.2f}")
        print(f"Total P&L: {self.trade_log['profit'].sum():.2f}")
        print(f"{'='*50}\n")


# ============================================================================
# MAIN TRADING LOOP
# ============================================================================

def main():
    """Main trading loop"""
    
    # Initialize components
    if not MT5Connector.initialize():
        return
    
    signal_gen = SignalGenerator(use_ml=Config.USE_ML)
    executor = TradeExecutor()
    
    # Trading state
    entry_time = datetime.datetime.now()
    trade_type = ""
    drawdown = 0
    drawup = 0
    order_code = 0
    
    print(f"\n🚀 Starting Pairs Trading Bot")
    print(f"   Pair: {Config.PRIMARY_SYMBOL} / {Config.SECONDARY_SYMBOL}")
    print(f"   Mode: {'Machine Learning' if Config.USE_ML else 'Cointegration'}")
    print(f"{'='*60}\n")
    
    loop_count = 0
    max_loops = 250
    
    while loop_count < max_loops:
        try:
            # Fetch market data
            df1, point1 = DataFetcher.get_rates(Config.PRIMARY_SYMBOL)
            df2, point2 = DataFetcher.get_rates(Config.SECONDARY_SYMBOL)
            
            # Calculate spreads
            df1['spread'] = df1['bid'] - df1['ask']
            df2['spread'] = df2['bid'] - df2['ask']
            
            # Adjust prices with spread
            df1['ask'] = df1['close'] + ((df1['spread'] + 6) * point1 / 2)
            df1['bid'] = df1['close'] - ((df1['spread'] + 6) * point1 / 2)
            df2['ask'] = df2['close'] + ((df2['spread'] + 6) * point2 / 2)
            df2['bid'] = df2['close'] - ((df2['spread'] + 6) * point2 / 2)
            
            # Merge and align data
            merged = pd.merge(df1, df2, on='time', how='outer')
            merged = merged.sort_values('time').fillna(method='ffill').dropna()
            
            df1_aligned = pd.DataFrame({
                'time': merged['time'],
                'bid': merged['bid_x'],
                'ask': merged['ask_x'],
                'close': merged['close_x'],
                'log_return': merged['log_return_x'],
                'returns': merged['returns_x']
            })
            
            df2_aligned = pd.DataFrame({
                'time': merged['time'],
                'bid': merged['bid_y'],
                'ask': merged['ask_y'],
                'close': merged['close_y'],
                'log_return': merged['log_return_y'],
                'returns': merged['returns_y']
            })
            
            # Generate signal
            signal, _ = signal_gen.generate_signal(df1_aligned, df2_aligned)
            current_spread = mt5.symbol_info(Config.PRIMARY_SYMBOL).spread
            
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Signal: {signal:.8f} | Spread: {current_spread}")
            
            # Check for entry signals
            if len(mt5.positions_get()) == 0:
                # Long signal (spread to widen)
                if signal >= Config.LONG_THRESHOLD and current_spread <= Config.MAX_SPREAD:
                    executor.place_order(Config.SECONDARY_SYMBOL, 1)  # BUY secondary
                    executor.place_order(Config.PRIMARY_SYMBOL, 2)    # SELL primary
                    print("\n✓ ENTRY: Long spread (expecting widening)")
                    entry_time = datetime.datetime.now()
                    trade_type = 'LONG_SPREAD'
                    order_code = 1
                
                # Short signal (spread to narrow)
                elif signal <= Config.SHORT_THRESHOLD and current_spread <= Config.MAX_SPREAD:
                    executor.place_order(Config.SECONDARY_SYMBOL, 2)  # SELL secondary
                    executor.place_order(Config.PRIMARY_SYMBOL, 1)    # BUY primary
                    print("\n✓ ENTRY: Short spread (expecting narrowing)")
                    entry_time = datetime.datetime.now()
                    trade_type = 'SHORT_SPREAD'
                    order_code = 2
            
            # Monitor open positions
            positions = mt5.positions_get()
            if positions:
                # Calculate total profit
                positions_df = pd.DataFrame([p._asdict() for p in positions])
                total_profit = positions_df['profit'].sum()
                
                # Track drawdown/drawup
                if total_profit <= drawdown:
                    drawdown = total_profit
                if total_profit >= drawup:
                    drawup = total_profit
                
                # Check exit conditions
                if total_profit > Config.PROFIT_TARGET or total_profit < Config.STOP_LOSS:
                    for position in positions:
                        executor.close_position(position)
                    
                    executor.log_trade(
                        entry_time, datetime.datetime.now(),
                        signal, total_profit, drawup, drawdown, trade_type
                    )
                    
                    # Reset tracking
                    drawdown = 0
                    drawup = 0
                    loop_count += 1
            
            time.sleep(1)  # Prevent CPU overload
            
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Error in main loop: {e}")
            time.sleep(5)
    
    # Cleanup
    MT5Connector.shutdown()
    print("\n✓ Trading session completed")


if __name__ == "__main__":
    main()
