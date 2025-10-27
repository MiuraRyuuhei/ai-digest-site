# digest.py  --- GitHub Actions 用：要約 + ベクトル埋め込み + JSON出力
# 収集 → 正規化/重複除去 → AIキーワードで軽フィルタ → 要約 → 埋め込み → JSON
import feedparser, datetime as dt, pytz, re, textwrap, hashlib, json
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ===== タイムゾーン =====
JST = pytz.timezone("Asia/Tokyo")

# ===== AI関連の軽量フィルタ（必要に応じて調整可）=====
AI_KEYWORDS = [
    "ai", "人工知能", "machine learning", "ml", "deep learning", "dl",
    "gpt", "llm", "transformer", "diffusion", "vision-language", "multi-modal",
    "rag", "retrieval", "prompt", "fine-tune", "finetune", "inference",
    "open-source ai", "model release", "benchmark", "alignment"
]
def is_ai_related(entry: dict) -> bool:
    text = f"{entry.get('title','')} {entry.get('rss_summary','')}".lower()
    return any(k in text for k in AI_KEYWORDS)

# ===== 収集するAI系RSS =====
FEEDS = [
    # 既存
    "https://deepmind.google/discover/blog/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://feeds.arxiv.org/rss/cs.LG",
    "https://huggingface.co/blog/feed.xml",
    "https://www.nvidia.com/en-us/about-nvidia/rss/feed.xml",

    # 追記（10+）
    "https://feeds.arxiv.org/rss/cs.AI",      # Artificial Intelligence
    "https://feeds.arxiv.org/rss/cs.CL",      # Computation and Language
    "https://feeds.arxiv.org/rss/cs.CV",      # Computer Vision
    "https://feeds.arxiv.org/rss/stat.ML",    # Statistics / Machine Learning
    "https://hai.stanford.edu/news/rss.xml",          # Stanford HAI
    "https://www.microsoft.com/en-us/research/feed/", # Microsoft Research
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://thegradient.pub/rss/",
    "https://paperswithcode.com/rss"
]

# ===== ユーティリティ =====
def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def fetch_entries(feeds):
    """RSSを巡回して正規化。JST化・重複除去・新しい順・AIキーワードで軽フィルタ。"""
    entries = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
        except Exception:
            continue
        for e in getattr(d, "entries", []):
            if getattr(e, "published_parsed", None):
                pub = dt.datetime(*e.published_parsed[:6], tzinfo=dt.timezone.utc).astimezone(JST)
            elif getattr(e, "updated_parsed", None):
                pub = dt.datetime(*e.updated_parsed[:6], tzinfo=dt.timezone.utc).astimezone(JST)
            else:
                pub = None
            entries.append({
                "title": clean(e.get("title", "")),
                "link": (e.get("link") or "").strip(),
                "published": pub,
                "rss_summary": clean(e.get("summary", "")),
            })

    # 重複除去（link優先）＋新しい順
    seen, uniq = set(), []
    for it in sorted(entries, key=lambda x: x["published"] or dt.datetime(1970,1,1, tzinfo=JST), reverse=True):
        key = it["link"] or hashlib.md5((it["title"] or "").encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key); uniq.append(it)

    # ★AIキーワードで軽く絞り込み（不要なら次行をコメントアウト）
    uniq = [it for it in uniq if is_ai_related(it)]
    return uniq

def within_24h(jst_dt):
    if not jst_dt: return False
    return (dt.datetime.now(JST) - jst_dt).total_seconds() <= 24*3600

def main():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim

    def summarize(text: str) -> str:
        text = (text or "")[:4000]
        if not text: return ""
        try:
            out = summarizer(text, max_length=110, min_length=70, do_sample=False)[0]["summary_text"]
            return clean(out)
        except Exception:
            return textwrap.shorten(text, width=220, placeholder="…")

    raw = fetch_entries(FEEDS)
    today = [e for e in raw if within_24h(e["published"])]
    top10 = today[:10] if len(today) >= 10 else raw[:10]

    texts_for_embed, digest_items = [], []
    for e in top10:
        base = e["rss_summary"] or e["title"]
        summ = summarize(base)
        digest_items.append({
            "title": e["title"],
            "url": e["link"],
            "published": e["published"].strftime("%Y-%m-%d %H:%M") if e["published"] else "",
            "summary": summ
        })
        texts_for_embed.append(f"{e['title']} {summ}")

    # 埋め込み（まとめてエンコード / 正規化済み）
    embs = encoder.encode(texts_for_embed, normalize_embeddings=True)
    for item, vec in zip(digest_items, embs):
        item["emb"] = [float(x) for x in vec]

    now = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    json_out = {"generated_at": f"{now} JST", "items": digest_items}

    Path("ai_digest_latest.json").write_text(
        json.dumps(json_out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("✅ updated ai_digest_latest.json with", len(digest_items), "items")

if __name__ == "__main__":
    main()
