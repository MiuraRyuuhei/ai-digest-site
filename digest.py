# digest.py  --- GitHub Actions 用
# 収集 → 正規化/重複除去 → 軽いAIフィルタ → 要約 → 埋め込み → 2つのJSON出力
import feedparser, datetime as dt, pytz, re, textwrap, hashlib, json, os
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ===== タイムゾーン =====
JST = pytz.timezone("Asia/Tokyo")

# ===== AI関連の軽量フィルタ（必要に応じて調整）=====
AI_KEYWORDS = [
    "ai", "人工知能", "machine learning", "ml", "deep learning", "dl",
    "gpt", "llm", "transformer", "diffusion", "vision-language", "multimodal",
    "rag", "retrieval", "prompt", "fine-tune", "finetune", "inference",
    "open-source ai", "model release", "benchmark", "alignment", "agent", "agents"
]
def is_ai_related(entry: dict) -> bool:
    text = f"{entry.get('title','')} {entry.get('rss_summary','')}".lower()
    return any(k in text for k in AI_KEYWORDS)

# ===== 収集するAI系RSS =====
FEEDS = [
    "https://deepmind.google/discover/blog/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://feeds.arxiv.org/rss/cs.LG",
    "https://huggingface.co/blog/feed.xml",
    "https://www.nvidia.com/en-us/about-nvidia/rss/feed.xml",
    "https://feeds.arxiv.org/rss/cs.AI",
    "https://feeds.arxiv.org/rss/cs.CL",
    "https://feeds.arxiv.org/rss/cs.CV",
    "https://feeds.arxiv.org/rss/stat.ML",
    "https://hai.stanford.edu/news/rss.xml",
    "https://www.microsoft.com/en-us/research/feed/",
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://thegradient.pub/rss/",
    "https://paperswithcode.com/rss",
]

# ===== ユーティリティ =====
def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def host(url: str) -> str:
    try:
        from urllib.parse import urlparse
        h = urlparse(url).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""

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

    # 軽フィルタ（不要ならコメントアウト）
    uniq = [it for it in uniq if is_ai_related(it)]
    return uniq

def within_24h(jst_dt):
    if not jst_dt: return False
    return (dt.datetime.now(JST) - jst_dt).total_seconds() <= 24*3600

# ===== 履歴コーパスのマージ =====
CORPUS_PATH = Path("ai_digest_corpus.json")
CORPUS_LIMIT = 1000  # 最大保持件数

def load_corpus():
    if CORPUS_PATH.exists():
        try:
            return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"items": []}
    return {"items": []}

def save_corpus(corpus):
    # 新しい順に並べ直し＆上限カット
    items = corpus.get("items", [])
    items.sort(key=lambda x: x.get("published",""), reverse=True)
    corpus["items"] = items[:CORPUS_LIMIT]
    CORPUS_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

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

    # 収集
    raw = fetch_entries(FEEDS)
    today = [e for e in raw if within_24h(e["published"])]
    top10_source = today if len(today) >= 10 else raw
    top10 = top10_source[:10]

    # 要約して latest 用の配列を作る
    texts_for_embed, latest_items = [], []
    for e in top10:
        base = e["rss_summary"] or e["title"]
        summ = summarize(base)
        latest_items.append({
            "title": e["title"],
            "url": e["link"],
            "published": e["published"].strftime("%Y-%m-%d %H:%M") if e["published"] else "",
            "summary": summ,
            "source": host(e["link"])
        })
        texts_for_embed.append(f"{e['title']} {summ}")

    # 埋め込み（latest）
    embs = encoder.encode(texts_for_embed, normalize_embeddings=True)
    for item, vec in zip(latest_items, embs):
        item["emb"] = [float(x) for x in vec]

    # latest.json を出力
    now = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    latest_json = {"generated_at": f"{now} JST", "items": latest_items}
    Path("ai_digest_latest.json").write_text(json.dumps(latest_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ wrote ai_digest_latest.json with", len(latest_items), "items")

    # ===== 履歴コーパスを更新（最新10件を取り込み、重複はURLでマージ）=====
    corpus = load_corpus()
    url_to_idx = {it.get("url"): i for i, it in enumerate(corpus.get("items", []))}
    for it in latest_items:
        u = it["url"]
        rec = {
            "title": it["title"],
            "url": u,
            "published": it["published"],
            "summary": it["summary"],
            "emb": it["emb"],
            "source": it["source"]
        }
        if u in url_to_idx:
            corpus["items"][url_to_idx[u]] = rec
        else:
            corpus.setdefault("items", []).append(rec)
    save_corpus(corpus)
    print("✅ merged into ai_digest_corpus.json (total:", len(corpus["items"]), ")")

if __name__ == "__main__":
    main()
