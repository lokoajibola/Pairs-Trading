# Pairs Trading Bot with Machine Learning

An algorithmic pairs trading bot for MetaTrader 5 that uses **cointegration analysis** and **machine learning** (XGBoost, Neural Networks) to identify and execute mean-reverting spread trades between correlated currency pairs.

## Features

- **Pairs Trading Strategy** – Trades the spread between two correlated instruments (e.g., GBPUSD/EURUSD or BTCUSD/ETHUSD).
- **Machine Learning Integration** – Uses MLPRegressor and XGBoost for signal generation.
- **Particle Filtering** – Implements particle filter for noise reduction in price data.
- **Cointegration Testing** – Statistical test for pair suitability.
- **Technical Indicators** – MACD, RSI, Bollinger Bands via pandas-ta.
- **Automated Trade Management** – Opens and closes positions based on spread deviations.
- **Profit/Loss Tracking** – Logs all trades with entry/exit times, profit, drawdown, and drawup.
- **Spread Analysis** – Monitors spread between instruments in real-time.

## Requirements

- **MetaTrader 5** terminal installed (Windows)
- Python 3.7+
- Required packages:
  ```
  MetaTrader5
  pandas
  numpy
  pytz
  pandas-ta
  statsmodels
  scikit-learn
  xgboost
  scipy
  ```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pairs-trading-ml-bot.git
   cd pairs-trading-ml-bot
   ```

2. **Install Python dependencies**
   ```bash
   pip install MetaTrader5 pandas numpy pytz pandas-ta statsmodels scikit-learn xgboost scipy
   ```

3. **Set up MetaTrader 5**
   - Install MT5 terminal from your broker.
   - Enable automated trading in MT5: **Tools → Options → Expert Advisors** → check "Allow automated trading".
   - Note the installation path.

4. **Configure environment variables**
   - Set up your MT5 credentials as environment variables:
     ```bash
     export MT5_ACCOUNT=your_account_number
     export MT5_PASSWORD=your_password
     export MT5_SERVER=your_broker_server
     ```
   - Or edit the script directly with your credentials.

5. **Customize trading parameters**
   - Adjust `curr1` and `curr2` for your desired pair (default: GBPUSD/EURUSD).
   - Modify `lot` size (default: 3.0).
   - Set profit targets: `fix_profit` (default: 20.9) and `fix_loss` (default: -198.9).

## How It Works

The bot continuously monitors two correlated instruments and executes trades when the spread deviates significantly from its mean:

### 1. Data Collection
- Fetches tick data for both instruments.
- Applies particle filter to reduce noise.
- Calculates returns and technical indicators.

### 2. Signal Generation (Two Modes)

**Mode 0 – Cointegration-based:**
- Tests for cointegration between the two series.
- Calculates spread and rolling averages.
- Generates signals based on z-score deviations.

**Mode 1 – Machine Learning (Default):**
- Calculates technical indicators (Bollinger Bands, MACD, RSI) for both instruments.
- Trains MLPRegressor or XGBoost model on historical data.
- Predicts future returns and generates spread signal.
- Uses StandardScaler for feature normalization.

### 3. Trade Execution

**When spread widens (pred_signal ≤ sig1):**
- SELL the first instrument (curr1)
- BUY the second instrument (curr2)
- Expected: spread will narrow

**When spread narrows (pred_signal ≥ sig2):**
- BUY the first instrument (curr1)
- SELL the second instrument (curr2)
- Expected: spread will widen

### 4. Trade Management
- Monitors combined position profit.
- Tracks drawdown and drawup.
- Closes position when profit target or stop loss is hit.
- Logs all trades with detailed statistics.

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `curr1` | First instrument | GBPUSD |
| `curr2` | Second instrument | EURUSD |
| `ML` | Machine learning mode (0=cointegration, 1=ML) | 1 |
| `lot` | Lot size per leg | 3.0 |
| `fix_profit` | Profit target in account currency | 20.9 |
| `fix_loss` | Stop loss in account currency | -198.9 |
| `sig1` | Lower threshold for short signal | -0.1 |
| `sig2` | Upper threshold for long signal | 0.1 |

## Trade Logging

All trades are saved in `res_check` DataFrame with the following columns:
- `entry_time` – Trade entry timestamp
- `exit_time` – Trade exit timestamp  
- `pred_signal` – Signal value at entry
- `profit` – Realized profit/loss
- `drawUP` – Maximum unrealized profit
- `drawdown` – Maximum unrealized loss
- `trade_type` – 'CLOSE +1' (long spread) or 'WIDEN -1' (short spread)

## Usage

Run the bot from your terminal:
```bash
python pairs_trading_bot.py
```

The bot will:
- Display current signal values
- Execute trades when thresholds are breached
- Monitor open positions continuously
- Log trade results

Press `Ctrl+C` to stop the bot (trades will remain open).

## Machine Learning Details

The ML implementation includes:
- **Feature Engineering**: Returns, technical indicators from both instruments.
- **Model Options**: MLPRegressor (neural network) or XGBoost.
- **Train/Test Split**: 90/10 with no shuffle (time-series appropriate).
- **Standardization**: Features are scaled using StandardScaler.
- **Signal Generation**: Combines predictions from both instruments.

## Risk Disclaimer

**Trading forex and CFDs carries a high level of risk.** This bot is for educational and research purposes only. Pairs trading involves significant risk including:
- Correlation breakdown
- Slippage during volatile markets
- Leverage amplification of losses

Past performance does not guarantee future results. Use at your own risk.

## Performance Metrics

The bot tracks key performance indicators:
- Total profit/loss
- Maximum drawdown
- Maximum run-up
- Win rate (calculated from trade log)

## Customization Tips

1. **Different Pairs**: Modify `curr1` and `curr2` for any MT5 symbols.
2. **Threshold Optimization**: Adjust `sig1`/`sig2` based on pair volatility.
3. **ML Parameters**: Change model type or add features in `get_signal()`.
4. **Timeframe**: Modify `days` parameter in `get_rates()` for different lookback periods.

## Future Enhancements

- Dynamic threshold adjustment based on volatility.
- Multiple pair portfolio management.
- Real-time correlation monitoring.
- Web interface for monitoring.
- Telegram alerts for trade signals.
- Backtesting framework with historical data.

## Troubleshooting

**Common Issues:**

1. **Connection Failed**: Verify MT5 is running and credentials are correct.
2. **No Trades**: Check spread thresholds and pair correlation.
3. **ML Errors**: Ensure sufficient historical data for training.
4. **Position Not Closing**: Verify profit/loss targets are realistic.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Author

**LK** – Initial development and maintenance

---

**Happy Trading!** 📊🤖
