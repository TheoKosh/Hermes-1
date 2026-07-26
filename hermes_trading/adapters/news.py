"""
news.py — Multi-source news sentiment adapter.

Sources (all free, no API key required):
  1. CryptoCompare news API — crypto-native headlines
  2. CoinGecko trending — what's hot right now
  3. Fear & Greed Index (alternative.me) — market-wide sentiment
  4. RSS feeds (CoinDesk, CoinTelegraph) — breaking news

Premium (optional):
  - NewsAPI.org via NEWS_API_KEY
  - CryptoPanic via CRYPTOPANIC_API_KEY

Schema:
{
    "schema_version": 2,
    "asset": "BTC/USDT",
    "sentiment_score": 0.15,      # -1.0 (bearish) to +1.0 (bullish)
    "headline_count": 42,
    "fear_greed": 52,             # 0-100 market Fear & Greed
    "trending": false,            # is this asset trending?
    "buzz_score": 0.65,           # 0-1 volume-of-coverage intensity
    "top_headlines": ["...", "...", "..."],  # top 3 headlines
    "sources_used": ["cryptocompare", "coingecko", "fear_greed", "rss"],
    "timestamp": "2024-..."
}
"""
import os
import re
from datetime import datetime, timezone

import httpx

SCHEMA_VERSION = 2

# --- Sentiment lexicon (expanded from v1) ---
BULLISH_WORDS = [
    "bull", "surge", "rally", "pump", "breakout", "high", "all-time", "ath",
    "adopt", "approve", "approval", "etf", "gain", "gain%", "up", "positive",
    "soar", "moon", "accumulate", "accumulation", "buy", "long", "support",
    "bounce", "recovery", "reclaim", "outperform", "institutional", "inflow",
    "whale buy", "upgrade", "partnership", "integration", "launch", "milestone",
    "breakthrough", "bullish", "optimis", "fund", "treasury", "reserve",
]
BEARISH_WORDS = [
    "bear", "crash", "dump", "plunge", "fall", "low", "ban", "hack", "exploit",
    "sell", "selling", "fear", "fud", "negative", "lawsuit", "reject", "rejection",
    "delist", "liquidation", "liquidate", "margin call", "short", "resistance",
    "breakdown", "capitulat", "outflow", "withdraw", "rug", "ponzi", "sec",
    "fraud", "investigation", "probe", "crackdown", "regulat", "bearish",
    "downgrade", "selloff", "correction", "risk", "warning", "death cross",
]

# Weight multipliers for bearish words (bad news hits harder/faster than good news)
BEARISH_WEIGHT = 1.3


class SchemaError(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_text(text: str) -> float:
    """Score a text string: returns raw sentiment (unnormalized)."""
    text_lower = text.lower()
    score = 0.0
    for w in BULLISH_WORDS:
        count = text_lower.count(w)
        score += count * 0.15
    for w in BEARISH_WORDS:
        count = text_lower.count(w)
        score -= count * 0.15 * BEARISH_WEIGHT
    return score


def normalize_sentiment(raw_score: float, n_articles: int) -> float:
    """Normalize to [-1, +1]."""
    if n_articles == 0:
        return 0.0
    per_article = raw_score / max(1, n_articles)
    return max(-1.0, min(1.0, per_article))


class NewsAdapter:
    def __init__(self, asset: str):
        self.asset = asset
        self.name = "news"
        self._coin = asset.split("/")[0].upper()
        self._coin_lower = self._coin.lower()

    async def fetch(self) -> dict:
        """Fetch from all available free sources in parallel, combine."""
        sources_used = []
        all_headlines = []
        raw_sentiment = 0.0
        total_articles = 0
        fear_greed = 50  # neutral default
        trending = False
        buzz_score = 0.0

        async with httpx.AsyncClient(timeout=15) as client:
            # Gather all sources concurrently
            import asyncio

            tasks = [
                ("cryptocompare", self._fetch_cryptocompare(client)),
                ("coingecko", self._fetch_coingecko(client)),
                ("fear_greed", self._fetch_fear_greed(client)),
                ("rss", self._fetch_rss(client)),
            ]

            # Optional premium sources
            news_key = os.environ.get("NEWS_API_KEY", "")
            if news_key:
                tasks.append(("newsapi", self._fetch_newsapi(client, news_key)))
            cpanic_key = os.environ.get("CRYPTOPANIC_API_KEY", "")
            if cpanic_key:
                tasks.append(("cryptopanic", self._fetch_cryptopanic(client, cpanic_key)))

            results = await asyncio.gather(
                *[self._safe_fetch(name, coro) for name, coro in tasks],
                return_exceptions=False,
            )

        for name, data in results:
            if data is None:
                continue
            sources_used.append(name)

            if name == "cryptocompare":
                raw_sentiment += data.get("raw_sentiment", 0)
                total_articles += data.get("article_count", 0)
                all_headlines.extend(data.get("headlines", []))
            elif name == "coingecko":
                trending = data.get("trending", False)
            elif name == "fear_greed":
                fear_greed = data.get("value", 50)
            elif name == "rss":
                raw_sentiment += data.get("raw_sentiment", 0)
                total_articles += data.get("article_count", 0)
                all_headlines.extend(data.get("headlines", []))
            elif name == "newsapi":
                raw_sentiment += data.get("raw_sentiment", 0)
                total_articles += data.get("article_count", 0)
                all_headlines.extend(data.get("headlines", []))
            elif name == "cryptopanic":
                raw_sentiment += data.get("raw_sentiment", 0)
                total_articles += data.get("article_count", 0)
                all_headlines.extend(data.get("headlines", []))

        # Combined sentiment
        sentiment = normalize_sentiment(raw_sentiment, max(1, total_articles))

        # Buzz score: how much coverage this asset is getting (0-1)
        # Normalized: 0 articles = 0, 20+ articles = 1.0
        buzz_score = min(1.0, total_articles / 20.0)

        # Top 3 headlines (deduplicated)
        seen = set()
        top_headlines = []
        for h in all_headlines:
            h_clean = h.strip()[:200]
            if h_clean and h_clean not in seen:
                seen.add(h_clean)
                top_headlines.append(h_clean)
            if len(top_headlines) >= 3:
                break

        result = {
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset,
            "sentiment_score": round(sentiment, 4),
            "headline_count": total_articles,
            "fear_greed": fear_greed,
            "trending": trending,
            "buzz_score": round(buzz_score, 3),
            "top_headlines": top_headlines,
            "sources_used": sources_used,
            "timestamp": now_iso(),
        }

        self._validate(result)
        return result

    async def _safe_fetch(self, name: str, coro) -> tuple:
        """Wrap a fetch with error handling — never crash on a single source failure."""
        try:
            data = await coro
            return (name, data)
        except Exception as e:
            return (name, None)

    async def _fetch_cryptocompare(self, client: httpx.AsyncClient) -> dict:
        """CryptoCompare news API (free, no key)."""
        r = await client.get(
            "https://min-api.cryptocompare.com/data/v2/news/",
            params={"categories": self._coin, "lang": "EN"},
        )
        if r.status_code != 200:
            return {"raw_sentiment": 0, "article_count": 0, "headlines": []}

        articles = r.json().get("Data", [])
        headlines = []
        raw = 0.0

        for a in articles[:50]:
            title = a.get("title", "")
            body = a.get("body", "")
            text = title + " " + body
            raw += score_text(text)
            headlines.append(title)

        return {"raw_sentiment": raw, "article_count": len(articles), "headlines": headlines}

    async def _fetch_coingecko(self, client: httpx.AsyncClient) -> dict:
        """CoinGecko trending coins (free, no key)."""
        r = await client.get("https://api.coingecko.com/api/v3/search/trending")
        if r.status_code != 200:
            return {"trending": False}

        trending_coins = r.json().get("coins", [])
        trending_ids = [c.get("item", {}).get("id", "").lower() for c in trending_coins]
        trending_symbols = [c.get("item", {}).get("symbol", "").lower() for c in trending_coins]

        is_trending = (
            self._coin_lower in trending_symbols
            or self._coin_lower in trending_ids
        )
        return {"trending": is_trending}

    async def _fetch_fear_greed(self, client: httpx.AsyncClient) -> dict:
        """Fear & Greed Index from alternative.me (free)."""
        r = await client.get("https://api.alternative.me/fng/?limit=1")
        if r.status_code != 200:
            return {"value": 50}

        data = r.json().get("data", [])
        if data:
            return {"value": int(data[0].get("value", 50))}
        return {"value": 50}

    async def _fetch_rss(self, client: httpx.AsyncClient) -> dict:
        """Fetch and parse RSS feeds from CoinDesk + CoinTelegraph (free)."""
        feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
            "https://cointelegraph.com/rss",
        ]

        headlines = []
        raw = 0.0
        count = 0

        for feed_url in feeds:
            try:
                r = await client.get(feed_url, follow_redirects=True)
                if r.status_code != 200:
                    continue
                # Simple regex XML parse (avoid lxml dependency)
                items = re.findall(r"<item>(.*?)</item>", r.text, re.DOTALL)
                for item in items[:25]:
                    title_match = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>", item, re.DOTALL)
                    title = ""
                    if title_match:
                        title = title_match.group(1) or title_match.group(2) or ""
                    title = title.strip()

                    # Only score if this article mentions our coin
                    if self._coin_lower in title.lower() or self._coin_lower in item.lower():
                        raw += score_text(title + " " + item)
                        headlines.append(title)
                        count += 1
            except Exception:
                continue

        return {"raw_sentiment": raw, "article_count": count, "headlines": headlines}

    async def _fetch_newsapi(self, client: httpx.AsyncClient, key: str) -> dict:
        """Premium: NewsAPI.org."""
        r = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": self._coin,
                "sortBy": "publishedAt",
                "pageSize": 50,
                "language": "en",
                "apiKey": key,
            },
        )
        if r.status_code != 200:
            return {"raw_sentiment": 0, "article_count": 0, "headlines": []}

        articles = r.json().get("articles", [])
        headlines = []
        raw = 0.0

        for a in articles[:50]:
            title = a.get("title", "")
            desc = a.get("description", "")
            raw += score_text(title + " " + desc)
            headlines.append(title)

        return {"raw_sentiment": raw, "article_count": len(articles), "headlines": headlines}

    async def _fetch_cryptopanic(self, client: httpx.AsyncClient, key: str) -> dict:
        """Premium: CryptoPanic news API."""
        r = await client.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": key, "currencies": self._coin, "kind": "news"},
        )
        if r.status_code != 200:
            return {"raw_sentiment": 0, "article_count": 0, "headlines": []}

        articles = r.json().get("results", [])
        headlines = []
        raw = 0.0

        for a in articles[:50]:
            title = a.get("title", "")
            raw += score_text(title)
            headlines.append(title)

        return {"raw_sentiment": raw, "article_count": len(articles), "headlines": headlines}

    def _validate(self, data: dict):
        required = {"schema_version", "asset", "sentiment_score", "headline_count", "timestamp"}
        if not required.issubset(data.keys()):
            raise SchemaError(f"news adapter: missing keys {required - set(data.keys())}")
        if data["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(f"news adapter: schema mismatch (got {data['schema_version']}, expected {SCHEMA_VERSION})")
