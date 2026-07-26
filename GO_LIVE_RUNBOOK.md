# GO-LIVE RUNBOOK — hermes-trading

## Current status: PAPER MODE (safe, simulated)
The worker is running the aggressive return-maximizing strategy across 8 assets
in paper mode. It enters trades, tracks equity, and respects the 10% drawdown
circuit breaker — all with virtual money.

## To go LIVE with real money:

### Step 1: Fund your Coinbase account
Deposit USDT (or USD→convert to USDT) into your Coinbase account.
The worker trades USDT pairs, so you need USDT balance.

### Step 2: Verify balance is visible
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519_railway railway-sweet-clarity \
  'cd /app && uv run python3 -c "
import asyncio, ccxt.async_support as ccxt, os
async def t():
    ex = ccxt.coinbase({\"apiKey\": os.environ[\"EXCHANGE_API_KEY\"], \"secret\": os.environ[\"EXCHANGE_API_SECRET\"]})
    b = await ex.fetch_balance()
    usdt = b.get(\"free\",{}).get(\"USDT\",0)
    print(f\"USDT available: {usdt}\")
    await ex.close()
asyncio.run(t())
"'

### Step 3: Flip the switch (3 Railway variables)
cd ~/hermes-trading
railway variables set HERMES_TRADING_MODE=live
railway variables set HERMES_TRADING_I_ACCEPT_RISK=true

Railway will auto-restart the worker. It will boot in LIVE mode.
Check the logs to confirm:
  railway logs --tail 30

You should see: "⚠ LIVE MODE — REAL MONEY AT RISK ⚠"

### Step 4: Monitor closely for the first hour
railway logs --tail 50    # watch for entries/exits
ssh ... 'cat /app/state/heartbeat.json'  # check equity + drawdown

### Step 5: Emergency stop (if needed)
railway variables set HERMES_TRADING_MODE=paper
# OR fully stop the service:
railway service stop

## Strategy parameters (v03 — aggressive)
- Basket: BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, LINK (vs USDT)
- Direction: BOTH (long when RSI<=35, short when RSI>=65)
- Stop loss: 3% per trade
- Take profit: 6% per trade (2:1 reward:risk)
- Max concurrent: 5 positions across 8 assets
- Risk per trade: 2% of equity
- Max position size: 15% of equity
- Drawdown circuit breaker: HALTS ALL TRADING at 10% portfolio DD

## Safety rails
1. Drawdown circuit breaker (10% → stop trading)
2. Max 5 concurrent positions (capital spread across basket)
3. 2% risk per trade (50 consecutive losses to blow up)
4. Paper mode default (must explicitly set live + accept risk)
5. Per-asset failure isolation (one bad asset doesn't crash basket)

## CRITICAL SECURITY NOTE
The API keys were pasted in plaintext in the chat history.
After going live, ROTATE THE KEYS:
1. coinbase.com → Settings → API → Revoke old key
2. Create new key (Trade + View permissions)
3. Set on Railway: railway variables set EXCHANGE_API_KEY="new..." EXCHANGE_API_SECRET="new..."
