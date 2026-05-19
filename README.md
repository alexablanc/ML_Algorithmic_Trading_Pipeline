# LLM + Dyna-Q Daily Trading Pipeline

Airflow-orchestrated pipeline pulling 60 days of live market data via yfinance. Integrates an OpenAI API call that interprets technical indicators (Bollinger %B, RSI, momentum) to generate a bullish/bearish/neutral signal, which feeds into a Dyna-Q reinforcement learning agent written from scratch. Adapted from a deterministic grid-world formulation to a live stochastic market environment — replacing the deterministic transition function with an empirically learned distribution `P(s' | s, a)` accumulated from real market observations. Encodes a 3,000-state space (10 bins × 10 bins × 10 bins × 3 LLM signal values). Developed as an independent extension of MSCS coursework at Georgia Tech.

---

## Architecture

```
fetch_market_data
      |
generate_llm_signal
      |
execute_q_learner
      |
log_trade_and_metrics
```

| Task | Description |
|---|---|
| `fetch_market_data` | Downloads 60 days of JPM OHLCV data via `yfinance` and computes BB %B, RSI, and Momentum |
| `generate_llm_signal` | Sends indicator values to GPT-4o-mini and receives a directional signal: 0=Bearish, 1=Neutral, 2=Bullish |
| `execute_q_learner` | Discretizes the indicator + LLM state into one of 3,000 states, updates the Q-table with yesterday's reward (Dyna-Q with 200 planning steps), and queries for today's action |
| `log_trade_and_metrics` | Appends the trade to `include/trades_log.csv` and logs cumulative return, Sharpe ratio, and mean daily return to `include/metrics.json` |

---

## File Structure

```
llm_qlearner_trading/
├── dags/
│   └── llm_qlearner_trading_dag.py       # Main DAG definition
├── plugins/
│   └── trading_pipeline/
│       ├── QLearner.py                   # Dyna-Q implementation (available upon request)
│       ├── indicators.py                 # BB %B, RSI, Momentum
│       ├── llm_signal.py                 # OpenAI API signal generation
│       ├── agent.py                      # State discretization + Q-Learner wrapper
│       └── performance.py               # Cumulative return, Sharpe ratio
├── include/
│   ├── trades_log.csv                    # Live trade log (auto-updated daily)
│   ├── metrics.json                      # Latest performance metrics
│   └── qlearner_state.json              # Previous day's state (used to compute reward)
├── .env                                  # API keys (not committed)
├── packages.txt
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- [Astronomer CLI](https://docs.astronomer.io/astro/cli/install-cli) installed
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone the repository

```bash
git clone https://github.com/alexablanc/machine_learning_trading_portfolio_optimization.git
cd llm_qlearner_trading
```

### 2. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key
```

### 3. Start the local Airflow environment

```bash
astro dev start
```

This builds the Docker image, installs dependencies from `requirements.txt`, and starts Airflow at `http://llm-qlearner-trading.localhost`.

### 4. Enable and run the DAG

In the Airflow UI, find `llm_qlearner_trading_pipeline`, toggle it **ON**, and click **Trigger DAG** to run it manually for the first time.

The pipeline will then run automatically at **11:00 AM UTC (4:00 AM PST) on weekdays**, using the previous trading day's closing data.

---

## How It Works

- **State space (3,000 states):** 10 bins for BB %B × 10 bins for RSI × 10 bins for Momentum × 3 LLM signal values.
- **Actions:** 0 = Short (−1000 shares), 1 = Cash (0 shares), 2 = Long (+1000 shares).
- **Reward:** The daily return of JPM, signed by the position held — Long earns +return, Short earns −return, Cash earns 0.
- **Dyna-Q (200 planning steps):** Since real market data arrives only once per day, Dyna-Q simulates 200 additional Bellman updates per real step using the learned transition model `T_count[s, a, s']` and reward model `R_model[s, a, s']`, accelerating Q-table convergence without requiring additional real interactions.
- **Exploration:** Starts at `rar=0.5` (50% random actions), decaying at `radr=0.99` per step. Expect several weeks of daily runs before the policy stabilizes.

---

## Output Files

All output files are written to `include/` and sync automatically to your local project folder.

| File | Description |
|---|---|
| `include/trades_log.csv` | Cumulative log of all trades: Date, Symbol, Order, Price, Target_Position |
| `include/metrics.json` | Latest performance metrics: Cumulative Return, Sharpe Ratio, Mean Daily Return |
| `include/qlearner_state.json` | Previous day's state, action, and closing price (used to compute next reward) |

---

## Notes

- Change the `symbol` variable in the DAG to trade any ticker supported by `yfinance`.
- This is a **paper trading** pipeline — it logs intended trades but does not connect to a brokerage. To execute real trades, integrate the [Alpaca API](https://alpaca.markets/) in the `log_trade_and_metrics` task.
- `QLearner.py` is not included in this repository and is available upon request.
