"""
audit_logger.py — Ghi log tương tác và feedback người dùng

Mỗi lần người dùng kiểm tra tin hoặc bấm 👍/👎 đều được ghi vào
audit_log.csv để theo dõi chất lượng hệ thống theo thời gian.
"""

import csv
import os
from datetime import datetime

LOG_FILE = "audit_log.csv"
LOG_FIELDS = ["timestamp", "masked_text", "source", "ml_label", "ml_confidence",
              "final_label", "ai_response_snippet", "feedback"]


def _ensure_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def log_interaction(masked_text: str, ai_response: str, feedback: str | None,
                    source: str = "unknown", ml_label: int = -1,
                    ml_confidence: float = 0.0, final_label: int = -1):
    """
    Ghi một dòng log.

    source      : "ml" | "llm" | "hybrid"
    ml_label    : nhãn ML dự đoán (0/1/-1 nếu không có)
    ml_confidence: độ tự tin của ML
    final_label : nhãn cuối cùng trả cho người dùng
    feedback    : None | "Good" | "Bad"
    """
    _ensure_header()
    row = {
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "masked_text":        masked_text[:200],          # giới hạn độ dài
        "source":             source,
        "ml_label":           ml_label,
        "ml_confidence":      f"{ml_confidence:.3f}",
        "final_label":        final_label,
        "ai_response_snippet": ai_response[:100],
        "feedback":           feedback or "",
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
