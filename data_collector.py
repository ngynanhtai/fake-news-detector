"""
data_collector.py — Thu thập tin tức thật từ RSS các báo VN lớn

Cách dùng:
    from data_collector import collect_news
    added, skipped = collect_news()          # fetch tất cả nguồn
    added, skipped = collect_news(max_per_feed=20)  # giới hạn mỗi nguồn

Tin được lấy về luôn có nhãn = 1 (Thật) vì đến từ báo uy tín.
Sau khi collect xong, gọi ml_model.retrain() để cập nhật model.
"""

import re
import time
import hashlib
import requests
import feedparser
import pandas as pd
from datetime import datetime

# ── Danh sách RSS nguồn báo uy tín ───────────────────────────────────────────
RSS_SOURCES = [
    # VnExpress
    {"name": "VnExpress - Thời sự",   "url": "https://vnexpress.net/rss/thoi-su.rss",        "label": 1},
    {"name": "VnExpress - Sức khỏe",  "url": "https://vnexpress.net/rss/suc-khoe.rss",       "label": 1},
    {"name": "VnExpress - Pháp luật", "url": "https://vnexpress.net/rss/phap-luat.rss",      "label": 1},
    {"name": "VnExpress - Kinh doanh","url": "https://vnexpress.net/rss/kinh-doanh.rss",     "label": 1},

    # Tuổi Trẻ
    {"name": "Tuổi Trẻ - Thời sự",    "url": "https://tuoitre.vn/rss/thoi-su.rss",           "label": 1},
    {"name": "Tuổi Trẻ - Sức khỏe",   "url": "https://tuoitre.vn/rss/suc-khoe.rss",          "label": 1},
    {"name": "Tuổi Trẻ - Pháp luật",  "url": "https://tuoitre.vn/rss/phap-luat.rss",         "label": 1},

    # Thanh Niên
    {"name": "Thanh Niên - Trang chủ","url": "https://thanhnien.vn/rss/home.rss",             "label": 1},
    {"name": "Thanh Niên - Thời sự",  "url": "https://thanhnien.vn/rss/thoi-su.rss",          "label": 1},

    # Dân Trí
    {"name": "Dân Trí - Xã hội",      "url": "https://dantri.com.vn/rss/xa-hoi.rss",         "label": 1},
    {"name": "Dân Trí - Sức khỏe",    "url": "https://dantri.com.vn/rss/suc-khoe.rss",       "label": 1},
    {"name": "Dân Trí - Pháp luật",   "url": "https://dantri.com.vn/rss/phap-luat.rss",      "label": 1},

    # 24h
    {"name": "24h - Tin tức",          "url": "https://www.24h.com.vn/upload/rss/tintuc247.rss",   "label": 1},
    {"name": "24h - Sức khỏe",         "url": "https://www.24h.com.vn/upload/rss/suc-khoe.rss",    "label": 1},
    {"name": "24h - Pháp luật",        "url": "https://www.24h.com.vn/upload/rss/phap-luat-hinh-su.rss", "label": 1},

    # VTV
    {"name": "VTV - Thời sự",          "url": "https://vtv.vn/trong-nuoc.rss",                "label": 1},

    # Nhân Dân
    {"name": "Nhân Dân - Thời sự",     "url": "https://nhandan.vn/rss/thoi-su.rss",           "label": 1},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

DATA_FILE = "du_lieu_tin_tuc.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _clean_html(text: str) -> str:
    """Loại bỏ HTML tags và khoảng trắng thừa."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_id(tieu_de: str) -> str:
    """Hash tiêu đề để nhận dạng bản trùng."""
    return hashlib.md5(tieu_de.strip().lower().encode("utf-8")).hexdigest()


def _load_existing_ids() -> set:
    """Đọc các hash tiêu đề đã có trong CSV để tránh duplicate."""
    try:
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        return {_make_id(str(t)) for t in df["tieu_de"].dropna()}
    except (FileNotFoundError, KeyError):
        return set()


def _fetch_feed(source: dict, max_items: int) -> list[dict]:
    """
    Fetch một RSS feed, trả về list dict sẵn sàng ghi vào CSV.
    Mỗi dict có keys: tieu_de, noi_dung, nhan, nguon, ngay_lay
    """
    try:
        resp = requests.get(source["url"], headers=HEADERS, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"  ⚠️  {source['name']}: lỗi kết nối — {e}")
        return []

    items = []
    for entry in feed.entries[:max_items]:
        tieu_de  = _clean_html(entry.get("title", "")).strip()
        noi_dung = _clean_html(entry.get("summary", entry.get("description", ""))).strip()

        if not tieu_de:
            continue

        # Nếu không có mô tả, dùng lại tiêu đề
        if not noi_dung:
            noi_dung = tieu_de

        items.append({
            "tieu_de":  tieu_de,
            "noi_dung": noi_dung,
            "nhan":     source["label"],
            "nguon":    source["name"],
            "ngay_lay": datetime.now().strftime("%Y-%m-%d"),
        })

    return items


# ── Hàm chính ────────────────────────────────────────────────────────────────
def collect_news(
    sources: list[dict] | None = None,
    max_per_feed: int = 30,
    delay_seconds: float = 0.5,
) -> tuple[int, int]:
    """
    Fetch RSS từ tất cả nguồn, lọc trùng, ghi vào CSV.

    Args:
        sources       : Danh sách nguồn tùy chỉnh (mặc định = RSS_SOURCES)
        max_per_feed  : Số bài tối đa lấy mỗi feed
        delay_seconds : Nghỉ giữa mỗi request (tránh bị block)

    Returns:
        (added, skipped) : Số bài thêm mới và số bài bỏ qua (trùng)
    """
    if sources is None:
        sources = RSS_SOURCES

    print(f"\n{'='*55}")
    print(f"🌐 Bắt đầu thu thập từ {len(sources)} nguồn RSS...")
    print(f"{'='*55}")

    existing_ids = _load_existing_ids()
    print(f"📂 CSV hiện có: {len(existing_ids)} bài đã ghi trước đó\n")

    new_rows  = []
    added     = 0
    skipped   = 0

    for source in sources:
        print(f"  🔄 {source['name']}...", end=" ", flush=True)
        items = _fetch_feed(source, max_per_feed)

        feed_added = 0
        for item in items:
            uid = _make_id(item["tieu_de"])
            if uid in existing_ids:
                skipped += 1
                continue
            existing_ids.add(uid)
            new_rows.append(item)
            feed_added += 1
            added += 1

        if items:
            print(f"+{feed_added} bài mới (bỏ qua {len(items)-feed_added} trùng)")
        time.sleep(delay_seconds)

    # Ghi vào CSV
    if new_rows:
        df_new = pd.DataFrame(new_rows)

        # Đảm bảo file có đủ cột tieu_de, noi_dung, nhan (cột cũ không bị mất)
        try:
            df_old = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
            df_out = pd.concat([df_old, df_new[["tieu_de", "noi_dung", "nhan"]]], ignore_index=True)
        except FileNotFoundError:
            df_out = df_new[["tieu_de", "noi_dung", "nhan"]]

        df_out.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
        print(f"\n✅ Đã ghi {added} bài mới vào {DATA_FILE}")
        print(f"   Tổng bài trong CSV: {len(df_out)}")
    else:
        print("\n⚠️  Không có bài mới nào được thêm.")

    print(f"{'='*55}")
    print(f"📊 Kết quả: +{added} mới | {skipped} trùng bỏ qua")
    print(f"{'='*55}\n")

    return added, skipped


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thu thập tin tức từ RSS báo VN")
    parser.add_argument("--max", type=int, default=30,
                        help="Số bài tối đa mỗi feed (mặc định: 30)")
    parser.add_argument("--retrain", action="store_true",
                        help="Tự động retrain ML model sau khi collect")
    args = parser.parse_args()

    added, _ = collect_news(max_per_feed=args.max)

    if args.retrain and added > 0:
        print("🤖 Đang retrain ML model với dữ liệu mới...")
        import ml_model
        ml_model.retrain()
        ml_model.reload_model()
        print("✅ Retrain xong!")
