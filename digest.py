# digest.py  --- GitHub Actions 用（pytz使用版）
# 収集 → 正規化/重複除去 → AIキーワードで軽く絞り込み → 要約 → JSON出力
import feedparser, datetime as dt, pytz, re, textwrap, hashlib, json
from pathlib import Path
from transformers import pipeline

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
    # 研究系（arXiv各分野）
    "https://feeds.arxiv.org/rss/cs.AI",      # Artificial Intelligence
    "https://feeds.arxiv.org/rss/cs.CL",      # Computation and Language
    "https://feeds.arxiv.org/rss/cs.CV",      # Computer Vision
    "https://feeds.arxiv.org/rss/stat.ML",    # Statistics / Machine Learning

    # 研究・教育機関
    "https://hai.stanford.edu/news/rss.xml",          # Stanford HAI
    "https://www.microsoft.com/en-us/research/feed/", # Microsoft Research

    # メディア（AIカテゴリ）
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://thegradient.pub/rss/",

    # コミュニティ/リソース
    "https://paperswithcode.com/rss"                  # 論文＋実装リンク
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
            # 公開日時（JST）
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
        seen.add(key)
        uniq.append(it)

    # ★AIキーワードで軽く絞り込み（不要なら次行をコメントアウト）
    uniq = [it for it in uniq if is_ai_related(it)]
    return uniq

def within_24h(jst_dt):
    if not jst_dt:
        return False
    return (dt.datetime.now(JST) - jst_dt).total_seconds() <= 24 * 3600

def main():
    # 軽量な要約モデル（初回にモデルDLあり）
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def summarize(text: str) -> str:
        text = (text or "")[:4000]  # 安全側にトリム
        if not text:
            return ""
        try:
            out = summarizer(text, max_length=110, min_length=70, do_sample=False)[0]["summary_text"]
            return clean(out)
        except Exception:
            # 万一失敗時は素朴に短縮
            return textwrap.shorten(text, width=220, placeholder="…")

    # 収集
    raw = fetch_entries(FEEDS)

    # 24h以内を優先し、足りなければ新着で補完して10件
    today = [e for e in raw if within_24h(e["published"])]
    top10 = today[:10] if len(today) >= 10 else raw[:10]

    # 要約してJSON整形
    digest_items = []
    for e in top10:
        base = e["rss_summary"] or e["title"]
        summ = summarize(base)
        digest_items.append({
            "title": e["title"],
            "url": e["link"],
            "published": e["published"].strftime("%Y-%m-%d %H:%M") if e["published"] else "",
            "summary": summ
        })

    now = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    json_out = {"generated_at": f"{now} JST", "items": digest_items}

    # リポジトリ直下に書き出し（Pagesがそのまま読む）
    Path("ai_digest_latest.json").write_text(
        json.dumps(json_out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("✅ updated ai_digest_latest.json with", len(digest_items), "items")

if __name__ == "__main__":
    main()

