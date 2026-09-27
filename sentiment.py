"""
sentiment.py
------------
Pulls recent news headlines about a company (default: Apple / AAPL) from
free RSS feeds (no paid API key needed) and scores each headline's
sentiment using a pretrained FinBERT model ("ProsusAI/finbert") from
Hugging Face transformers. Produces a daily average sentiment score in
data/sentiment.csv, where score = P(positive) - P(negative), in [-1, 1].

Free RSS sources tried (no API key required):
    - Yahoo Finance headline RSS for the ticker
    - Google News RSS search for the company name

If none of the network calls succeed (no internet / feeds blocked) or the
FinBERT weights can't be downloaded from huggingface.co, this script falls
back to clearly-labeled offline substitutes so the rest of the pipeline can
still be exercised:
    - synthetic but plausible headlines
    - a small locally-constructed (untrained) BERT sequence-classification
      head, which exercises the exact same tokenize -> model -> softmax
      code path as real FinBERT, just without meaningful learned weights.

Usage:
    python sentiment.py --ticker AAPL --company "Apple"
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

FINBERT_MODEL_NAME = "ProsusAI/finbert"


# ----------------------------------------------------------------------
# Headline collection
# ----------------------------------------------------------------------
def fetch_headlines_rss(ticker: str, company: str, max_items: int = 40):
    """Try a couple of free, no-key-required RSS sources."""
    import feedparser

    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={company}+stock&hl=en-US&gl=US&ceid=US:en",
    ]

    headlines = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            if getattr(feed, "bozo", 0) and not getattr(feed, "entries", None):
                continue
            for entry in feed.entries[:max_items]:
                title = getattr(entry, "title", None)
                published = getattr(entry, "published", None)
                if title:
                    headlines.append({"headline": title, "published": published, "source": url})
        except Exception as e:
            print(f"[sentiment] RSS fetch failed for {url}: {e}")

    return headlines


def synthetic_headlines(company: str, n_days: int = 14, start: str = None, end: str = None):
    """
    Offline fallback: plausible, varied headlines with mixed sentiment.

    If start/end are given, headlines are generated across that whole date
    range (e.g. matching the price history) with a slowly-drifting sentiment
    regime, so the resulting daily sentiment series has realistic day-to-day
    AND longer-term variation for models to actually learn from -- rather
    than a single near-constant value repeated across history.
    """
    templates_pos = [
        f"{company} shares rally as quarterly revenue beats expectations",
        f"Analysts raise price target on {company} after strong earnings",
        f"{company} unveils new product line, investors optimistic",
        f"{company} stock climbs on upbeat guidance for next quarter",
    ]
    templates_neg = [
        f"{company} shares slide amid supply chain concerns",
        f"Regulators open new antitrust probe into {company}",
        f"{company} misses revenue estimates, stock drops",
        f"Analysts downgrade {company} citing weakening demand",
    ]
    templates_neu = [
        f"{company} to report quarterly earnings next week",
        f"{company} announces annual shareholder meeting date",
        f"What to expect from {company}'s upcoming product event",
        f"{company} stock trades flat ahead of earnings season",
    ]

    rng = np.random.default_rng(7)

    if start and end:
        days = pd.bdate_range(start=start, end=end)
    else:
        today = datetime.today()
        days = [today - timedelta(days=i) for i in range(n_days)][::-1]

    # Slowly-drifting underlying "sentiment regime" (mean-reverting random walk
    # in [-1, 1]) so different historical periods have genuinely different
    # average sentiment -- this is what gives the model a real signal to learn.
    n = len(days)
    regime = np.zeros(n)
    for i in range(1, n):
        regime[i] = 0.98 * regime[i - 1] + rng.normal(0, 0.03)
    regime = np.clip(regime, -0.9, 0.9)

    headlines = []
    for i, day in enumerate(days):
        n_today = rng.integers(1, 4)
        # bias the pos/neg/neu mix toward today's regime value
        p_pos = np.clip(0.35 + 0.3 * regime[i], 0.05, 0.85)
        p_neg = np.clip(0.35 - 0.3 * regime[i], 0.05, 0.85)
        p_neu = max(1e-3, 1 - p_pos - p_neg)
        probs = np.array([p_pos, p_neg, p_neu])
        probs = probs / probs.sum()

        for _ in range(n_today):
            bucket = rng.choice(["pos", "neg", "neu"], p=probs)
            pool = {"pos": templates_pos, "neg": templates_neg, "neu": templates_neu}[bucket]
            title = pool[rng.integers(0, len(pool))]
            day_dt = pd.Timestamp(day)
            headlines.append(
                {
                    "headline": title,
                    "published": day_dt.strftime("%a, %d %b %Y %H:%M:%S GMT"),
                    "source": "synthetic (offline fallback)",
                }
            )
    return headlines


# ----------------------------------------------------------------------
# FinBERT scoring
# ----------------------------------------------------------------------
def load_finbert():
    """Load the real pretrained FinBERT model + tokenizer from Hugging Face."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.eval()
    # FinBERT label order is: positive, negative, neutral
    id2label = model.config.id2label
    return tokenizer, model, id2label


def load_offline_fallback_model():
    """
    Build a tiny, randomly-initialized BERT-style sequence classifier
    locally (no download). This exercises the exact same
    tokenize -> forward -> softmax code path as real FinBERT so the
    pipeline can be end-to-end tested without internet access, but the
    resulting scores are NOT meaningful sentiment (random weights).
    """
    from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
    import tempfile

    print("[sentiment] Building a tiny local BERT classifier (untrained) as an offline stand-in for FinBERT.")

    # Minimal WordPiece vocab covering our synthetic headline vocabulary
    vocab_words = set()
    for h in synthetic_headlines("Apple", n_days=1) + synthetic_headlines("AAPL", n_days=1):
        vocab_words.update(h["headline"].lower().replace(",", "").replace("'s", " s").split())
    base_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = base_tokens + sorted(vocab_words)

    tmp_dir = tempfile.mkdtemp()
    vocab_path = os.path.join(tmp_dir, "vocab.txt")
    with open(vocab_path, "w") as f:
        f.write("\n".join(vocab))

    tokenizer = BertTokenizer(vocab_file=vocab_path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=3,
    )
    model = BertForSequenceClassification(config)
    model.eval()
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    return tokenizer, model, id2label


_POS_WORDS = ["rally", "rallies", "beats", "beat", "unveils", "optimistic", "climbs",
              "upbeat", "strong", "raise", "raises"]
_NEG_WORDS = ["slide", "slides", "probe", "misses", "miss", "drops", "downgrade",
              "downgrades", "weakening", "antitrust", "concerns"]


def _lexicon_score(text: str) -> float:
    """Simple deterministic keyword-based sentiment score in [-1, 1]."""
    t = text.lower()
    pos_hits = sum(1 for w in _POS_WORDS if w in t)
    neg_hits = sum(1 for w in _NEG_WORDS if w in t)
    if pos_hits == 0 and neg_hits == 0:
        return 0.0
    return (pos_hits - neg_hits) / (pos_hits + neg_hits)


def score_headlines(headlines, tokenizer, model, id2label, use_lexicon_blend: bool = False):
    """
    Score headlines for sentiment.

    use_lexicon_blend=True is used only for the offline fallback model: an
    UNTRAINED randomly-initialized transformer produces outputs dominated by
    its random bias rather than headline content, so on its own it can't
    reflect real sentiment. Blending in a small deterministic lexicon score
    keeps the exact same tokenize -> model -> softmax code path exercised
    (for pipeline-mechanics testing) while making the resulting numbers
    actually track headline content, so downstream models/demos see
    meaningful variation. Real FinBERT (use_lexicon_blend=False) never uses
    this -- its own learned weights already produce meaningful sentiment.
    """
    import torch

    rows = []
    for h in headlines:
        inputs = tokenizer(h["headline"], return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]

        label_probs = {id2label[i].lower(): p for i, p in enumerate(probs)}
        pos = label_probs.get("positive", 0.0)
        neg = label_probs.get("negative", 0.0)
        model_score = pos - neg  # in [-1, 1]

        if use_lexicon_blend:
            lex_score = _lexicon_score(h["headline"])
            score = 0.15 * model_score + 0.85 * lex_score
        else:
            score = model_score

        rows.append(
            {
                "headline": h["headline"],
                "published": h.get("published"),
                "source": h.get("source"),
                "sentiment_score": score,
                "positive_prob": pos,
                "negative_prob": neg,
            }
        )
    return pd.DataFrame(rows)


def parse_published_to_date(series: pd.Series) -> pd.Series:
    def _parse(x):
        if not x:
            return pd.NaT
        for fmt in (None,):
            try:
                return pd.to_datetime(x, errors="coerce", utc=True)
            except Exception:
                continue
        return pd.NaT

    parsed = series.apply(_parse)
    return parsed.dt.tz_localize(None) if hasattr(parsed, "dt") else parsed


def main():
    parser = argparse.ArgumentParser(description="Score news sentiment for a ticker using FinBERT.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--company", default="Apple")
    parser.add_argument("--start", default=None, help="Start date for synthetic fallback headline range (should match price history)")
    parser.add_argument("--end", default=None, help="End date for synthetic fallback headline range (default: today)")
    parser.add_argument("--out", default="data/sentiment.csv")
    parser.add_argument("--headlines-out", default="data/sentiment_headlines.csv")
    args = parser.parse_args()

    end = args.end or datetime.today().strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1. Headlines
    print(f"[sentiment] Fetching headlines for {args.company} ({args.ticker}) via free RSS feeds ...")
    headlines = fetch_headlines_rss(args.ticker, args.company)
    if headlines:
        print(f"[sentiment] SUCCESS: fetched {len(headlines)} real headlines via RSS.")
        headline_source = "rss (real)"
    else:
        print("[sentiment] RSS fetch returned nothing (likely no internet access). Using synthetic headlines.")
        headlines = synthetic_headlines(args.company, start=args.start, end=end)
        headline_source = "synthetic (offline fallback)"

    # 2. Model
    print(f"[sentiment] Loading FinBERT ({FINBERT_MODEL_NAME}) from Hugging Face ...")
    is_fallback_model = False
    try:
        tokenizer, model, id2label = load_finbert()
        print("[sentiment] SUCCESS: real pretrained FinBERT loaded.")
        model_source = "ProsusAI/finbert (real, pretrained)"
    except Exception as e:
        print(f"[sentiment] Could not download FinBERT: {e}")
        tokenizer, model, id2label = load_offline_fallback_model()
        model_source = "local untrained BERT stand-in + keyword blend (offline fallback -- NOT real FinBERT sentiment)"
        is_fallback_model = True

    # 3. Score
    print("[sentiment] Scoring headlines ...")
    scored = score_headlines(headlines, tokenizer, model, id2label, use_lexicon_blend=is_fallback_model)
    scored["date"] = parse_published_to_date(scored["published"])
    scored["date"] = scored["date"].fillna(pd.Timestamp(datetime.today().date()))
    scored["date"] = scored["date"].dt.date

    scored["headline_source"] = headline_source
    scored["model_source"] = model_source
    scored.to_csv(args.headlines_out, index=False)

    # 4. Daily aggregate
    daily = scored.groupby("date", as_index=False)["sentiment_score"].mean()
    daily.rename(columns={"sentiment_score": "avg_sentiment"}, inplace=True)
    daily["ticker"] = args.ticker
    daily["headline_source"] = headline_source
    daily["model_source"] = model_source
    daily.to_csv(args.out, index=False)

    print(f"[sentiment] Saved {len(scored)} scored headlines -> {args.headlines_out}")
    print(f"[sentiment] Saved {len(daily)} daily sentiment rows -> {args.out}")
    print(daily.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
