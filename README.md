# Quantorpolybot

Fully autonomous, self-improving Polymarket trading bot. Runs 100% locally
on Windows or Linux. Async end-to-end. Safe-by-default — never places a
real order unless **two** config flags are explicitly set.

## What it does

1. **Discovers** every active Polymarket market via Gamma API and caches it
   locally.
2. **Ingests** news (RSS), economic data (FRED), forecasts (Metaculus), and
   live Polymarket prices over WebSocket.
3. **Reasons** with a local Ollama LLM for every new feed item: "given this
   news, which market should we trade and at what implied probability?"
4. **Sizes** each trade with fractional Kelly under strict per-position,
   per-category, and daily-loss limits.
5. **Submits** orders through `py-clob-client`, monitors fills, cancels
   stale orders.
6. **Learns** every night:
   - Backtests parameter grids and auto-promotes a better config when it
     beats the current one by >5% (Sharpe or PnL).
   - Asks Ollama to evolve its own grading prompt based on recent failure
     modes, saves new versions, and conservatively activates them.
7. **Surfaces** everything in a FastAPI dashboard at `http://localhost:8000`.

## Quick start

```bash
# 1. Install Python 3.11+ and Ollama (https://ollama.com).
ollama pull mistral
ollama serve         # runs at http://localhost:11434

# 2. Install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

# 3. Configure
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/macOS
# Edit .env if you have a FRED_API_KEY (free from fred.stlouisfed.org).

# 4. Run
python run.py
```

Then open `http://localhost:8000` for the dashboard.

The first launch starts in **DRY RUN** mode. The bot will fetch markets,
ingest feeds, generate signals, and log "DRY RUN - would have placed
order ..." instead of submitting anything.

## Enabling live trading

Live trading is gated by **two independent flags** plus signed CLOB
credentials:

1. In `config/config.yaml`:
   ```yaml
   dry_run: false
   live_trading_enabled: true
   ```
2. In `.env`, set the wallet credentials (use a dedicated wallet you can
   afford to lose):
   ```
   POLY_PRIVATE_KEY=0x...
   POLY_FUNDER_ADDRESS=0x...
   ```
3. `live_trading_enabled` can be flipped from the dashboard's Kill Switch
   button at any time. `dry_run` requires editing the YAML (intentionally
   higher friction).

If either flag is missing or false, the bot logs the would-be order and
keeps every other subsystem (feeds, signals, scheduler, learning) running
exactly as in live mode — that way the optimization and learning modules
gather real data from day one.

## Project layout

```
Quantorpolybot/
  core/
    feeds/          RSS, FRED, Metaculus, Polymarket WebSocket
    markets/        Gamma discovery + read-side cache
    signals/        Candidate matching + Ollama JSON pipeline
    risk/           Hard rules + fractional Kelly + correlation
    execution/      Order engine, py-clob-client wrapper, fill monitor
    state/          Positions, balances, cooldowns, health
    scheduler/      APScheduler async cron jobs
    learning/       Daily prompt evolution
    optimization/   Daily backtest grid search + auto-tune
    backtest/       Replay engine
    utils/          DB, config, logging, hashing, helpers
  dashboard/        FastAPI + Jinja2 UI
  config/           config.yaml + prompts.yaml
  tests/            pytest suite
  logs/             rotating logs + audit JSONL
  run.py            single entrypoint
  polybot.db        auto-created SQLite (WAL mode)
```

## Configuration

All tunables live in `config/config.yaml`. Highlights:

| Section | Key | What it does |
|---|---|---|
| `risk` | `max_position_usd` | Hard cap per single market |
| `risk` | `max_total_exposure_usd` | Cap across all open positions |
| `risk` | `max_daily_loss_usd` | Bot halts new trades when breached |
| `risk` | `min_confidence` | Reject signals below this LLM confidence |
| `risk` | `min_edge` | Reject signals where |implied - mid| is below this |
| `risk` | `cooldown_seconds` | Per-market re-trade cooldown |
| `risk` | `max_correlated_exposure_usd` | Cap per category bucket |
| `kelly` | `fraction` | Fractional Kelly safety multiplier |
| `execution` | `order_timeout_seconds` | Auto-cancel unfilled orders |
| `optimization` | `improvement_threshold` | Sharpe/PnL gain required to auto-update config |

The optimization module rewrites this file in-place when it finds a better
config. The previous version is always backed up to `config.yaml.bak`
before any change. Every override is also logged to the `config_overrides`
table.

## Self-improvement

- **Optimization (`scheduler.optimization_cron`, default 03:05 daily):**
  Replays the last 30 days of stored signals against the historical
  `price_ticks` table. Walks the `optimization.param_grid` (min_confidence,
  min_edge, kelly_fraction). If the best combo beats the current config by
  more than `improvement_threshold` (Sharpe OR PnL), it writes the new
  values to `config.yaml` and reloads them in-process — no restart needed.

- **Learning (`scheduler.learning_cron`, default 03:30 daily):**
  Scores the active prompt against recent labelled outcomes. Asks Ollama
  to propose a revised prompt addressing the most common failure modes.
  The new version is saved to `config/prompts.yaml`. Promotion is
  conservative — it only activates a new prompt when the current baseline
  is negative-EV.

## Database tables

`polybot.db` (auto-created on first run, WAL mode) contains:

- `markets` — local cache of Polymarket markets
- `feed_items` — deduplicated news / data items (URL hash)
- `signals` — every Ollama decision with implied prob, confidence, edge
- `orders` — every order submitted (dry or live)
- `executions` — fills
- `positions` — open and closed
- `price_ticks` — every WebSocket update (used by the backtester)
- `system_log`, `config_overrides`, `health_checks`, `prompt_evals`

## Running the tests

```bash
pip install pytest pytest-asyncio
pytest -v
```

## Troubleshooting

- **Dashboard returns 500 on a fresh DB** — let it run for a minute. The
  market discovery and feed loops need one tick to populate tables.
- **Ollama timeouts** — increase `signals.ollama_timeout_seconds`. The
  `mistral` model on a CPU box can take 30–60s per call.
- **Polymarket WS disconnects** — auto-reconnects with exponential
  backoff. Check the logs/`health_checks` table.
- **Live order rejected as "client not ready"** — `POLY_PRIVATE_KEY` or
  `POLY_FUNDER_ADDRESS` is missing in `.env`.

## Safety reminder

This bot trades real money on a real exchange when `dry_run: false` and
`live_trading_enabled: true`. The risk engine is strict but no risk model
is perfect. Use a dedicated, small-balance wallet. Read the code. Watch
the dashboard. Hit the kill switch if anything looks off.
