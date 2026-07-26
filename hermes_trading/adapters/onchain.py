"""
onchain.py — On-chain data adapter.

Fetches on-chain metrics for BTC.
Free: Blockchain.info public API.
Premium: Glassnode via GLASSNODE_API_KEY in .env.

Schema:
{
    "schema_version": 1,
    "asset": "BTC/USDT",
    "hashrate": 650000000,   # TH/s (approx)
    "difficulty": 85000000000000,
    "mempool_size": 50000,   # unconfirmed txs
    "timestamp": "2024-..."
}
"""
import os
from datetime import datetime, timezone

import httpx


SCHEMA_VERSION = 1


class SchemaError(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class OnchainAdapter:
    def __init__(self, asset: str):
        self.asset = asset
        self.name = "onchain"
        base = asset.split("/")[0].upper()
        self._coin = base if base == "BTC" else "BTC"  # free endpoint is BTC-only

    async def fetch(self) -> dict:
        glassnode_key = os.environ.get("GLASSNODE_API_KEY", "")

        if glassnode_key:
            data = await self._fetch_glassnode(glassnode_key)
        else:
            data = await self._fetch_free()

        result = {
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset,
            **data,
            "timestamp": now_iso(),
        }

        self._validate(result)
        return result

    async def _fetch_free(self) -> dict:
        """Free: Blockchain.info public API."""
        async with httpx.AsyncClient(timeout=15) as client:
            # hashrate
            r = await client.get("https://blockchain.info/q/hashrate")
            hashrate = int(r.text.strip()) if r.status_code == 200 else 0

            # difficulty
            r2 = await client.get("https://blockchain.info/q/getdifficulty")
            difficulty = float(r2.text.strip()) if r2.status_code == 200 else 0.0

            # mempool size (unconfirmed count)
            r3 = await client.get("https://blockchain.info/q/unconfirmedcount")
            mempool = int(r3.text.strip()) if r3.status_code == 200 else 0

        return {
            "hashrate": hashrate,
            "difficulty": difficulty,
            "mempool_size": mempool,
        }

    async def _fetch_glassnode(self, key: str) -> dict:
        """Premium: Glassnode API."""
        base_url = "https://api.glassnode.com/v1/metrics"
        params = {"a": "BTC", "api_key": key}

        async with httpx.AsyncClient(timeout=15) as client:
            # hashrate (TH/s)
            r = await client.get(f"{base_url}/mining/hash-rate-mean", params=params)
            hashrate = r.json()[-1]["v"] if r.status_code == 200 and r.json() else 0

            # difficulty
            r2 = await client.get(f"{base_url}/mining/difficulty-latest", params=params)
            difficulty = r2.json()[-1]["v"] if r2.status_code == 200 and r2.json() else 0

        return {
            "hashrate": int(hashrate),
            "difficulty": float(difficulty),
            "mempool_size": 0,  # not available on free Glassnode tier
        }

    def _validate(self, data: dict):
        required = {"schema_version", "asset", "hashrate", "difficulty", "mempool_size", "timestamp"}
        if not required.issubset(data.keys()):
            raise SchemaError(f"onchain adapter: missing keys {required - set(data.keys())}")
        if data["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(f"onchain adapter: schema mismatch")
