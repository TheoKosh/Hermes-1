"""Research database: load, query, and apply academic trading papers.

Papers are stored as JSON in hermes_trading/research_db/. Each entry contains
key findings, strategy rules, performance metrics, and integration notes.
"""

import json
from pathlib import Path
from typing import Optional

RESEARCH_DIR = Path(__file__).parent


def list_papers() -> list[dict]:
    """Return metadata for all papers in the database."""
    papers = []
    for f in sorted(RESEARCH_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        papers.append({
            "id": data["paper"]["id"],
            "title": data["paper"]["title"],
            "authors": data["paper"]["authors"],
            "date": data["paper"]["date"],
            "file": f.name,
        })
    return papers


def load_paper(paper_id: str) -> dict:
    """Load a paper by its ID (e.g., 'ssrn-4631351')."""
    path = RESEARCH_DIR / f"{paper_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"paper not found: {paper_id}")
    return json.loads(path.read_text())


def get_key_findings(paper_id: str) -> list[dict]:
    """Return key findings for a paper."""
    return load_paper(paper_id)["key_findings"]


def get_strategy_rules(paper_id: str) -> dict:
    """Return strategy rules for a paper."""
    return load_paper(paper_id)["strategy_rules"]


def get_performance_metrics(paper_id: str) -> dict:
    """Return performance metrics for a paper."""
    return load_paper(paper_id)["performance_metrics"]


def find_strategies_by_sharpe(min_sharpe: float = 1.0) -> list[dict]:
    """Find all strategies with Sharpe ratio above threshold."""
    results = []
    for f in sorted(RESEARCH_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        metrics = data.get("performance_metrics", {})
        for instrument, m in metrics.items():
            if isinstance(m, dict) and m.get("sharpe_ratio", 0) >= min_sharpe:
                results.append({
                    "paper_id": data["paper"]["id"],
                    "strategy": data["strategy_rules"]["name"],
                    "instrument": instrument,
                    "sharpe": m["sharpe_ratio"],
                    "total_return_pct": m.get("total_return_pct"),
                    "max_drawdown_pct": m.get("max_drawdown_pct"),
                })
    return sorted(results, key=lambda x: -x["sharpe"])


def get_crypto_adaptation_notes(paper_id: str) -> dict:
    """Return crypto-specific adaptation notes for a paper."""
    return load_paper(paper_id).get("relevance_to_crypto", {})
