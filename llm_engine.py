"""
llm_engine.py — Chỉ gọi LLM khi ML không đủ tự tin
Tối ưu hóa: Bổ sung Few-Shot Context từ ML Database
"""

import re
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma:7b"

# System prompt
SYSTEM_PROMPT = """
Bạn là trợ lý giải thích kết quả nhận diện tin giả.

QUAN TRỌNG:

Nhãn đã được quyết định bởi hệ thống Machine Learning.

Bạn KHÔNG được thay đổi nhãn.

Nhiệm vụ:

1. Giải thích vì sao hệ thống đưa ra kết quả này.
2. Đánh giá rủi ro.
3. Đưa lời khuyên phù hợp.

Trả về JSON:

{
    "ly_do":"...",
    "loi_khuyen":"...",
    "do_nguy_hiem":1
}

do_nguy_hiem từ 1 tới 5.
"""


def mask_personal_info(text: str) -> str:
    # Email
    text = re.sub(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        '[EMAIL_ĐÃ_ẨN]',
        text
    )
    # CCCD (12 số) / CMND (9 số)
    text = re.sub(
        r'(?<!\d)(?:0\d{2}[0-2]\d{8}|\d{9})(?!\d)',
        '[CCCD_ĐÃ_ẨN]',
        text
    )
    # SĐT Việt Nam
    text = re.sub(
        r'(?<!\d)(?:\+84|84|0)(?:3[2-9]|5[25689]|7[06-9]|8[1-9]|9[0-9])\d{7}(?!\d)',
        '[SĐT_ĐÃ_ẨN]',
        text
    )
    # STK ngân hàng
    text = re.sub(
        r'(?i)(?:tài khoản|số tk|stk|chuyển khoản|banking|ngân hàng|vietcombank|vcb|techcombank|bidv|agribank|mbbank|vpbank|acb|sacombank).{0,50}?(?<!\d)(\d{10,16})(?!\d)',
        lambda m: m.group(0).replace(m.group(1), '[STK_ĐÃ_ẨN]'),
        text
    )
    return text


def analyze_news(
    news_text: str,
    label: int,
    confidence: float
) -> tuple[str, str, int]:
    """
    Gọi Ollama LLM để phân tích tin tức.
    ml_context: Chuỗi chứa các mẫu dữ liệu thật/giả lấy từ CSV để làm ví dụ cho LLM (Few-shot)
    """
    masked_text = mask_personal_info(news_text)
    
    # Kết hợp nội dung tin cần check và ngữ cảnh dữ liệu hệ thống đã train
    user_content = ""
    prediction_text = (
    "THAT"
    if label == 1
    else "GIA"
)

    user_content = f"""
    Kết quả từ Machine Learning:

    Label: {prediction_text}

    Confidence: {confidence:.3f}

    Văn bản:

    {masked_text}
    """
    
    user_content += f"Hãy phân tích đoạn tin này cho Cô/Chú:\n\"\"\"\n{masked_text}\n\"\"\""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0  # Đặt bằng 0 để LLM trả ra kết quả nhất quán, không sáng tạo bừa bãi
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result_json = response.json()
        
        # Parse nội dung JSON từ LLM trả về
        content_str = result_json["message"]["content"]
        data = json.loads(content_str)

        ly_do      = data.get("ly_do", "Không rõ lý do.")
        loi_khuyen = data.get("loi_khuyen", "Hãy thận trọng và hỏi ý kiến người thân.")
        nguy_hiem  = max(1, min(5, int(data.get("do_nguy_hiem", 3))))

        if label == 0:
            icon   = "🚨"
            title  = "CẢNH BÁO: Có thể là LỪA ĐẢO hoặc TIN GIẢ!"
            danger = "🔴" * nguy_hiem + "⬜" * (5 - nguy_hiem)
        else:
            icon   = "✅"
            title  = "Tin này có vẻ đáng tin cậy"
            danger = "🟢" + "⬜" * 4

        response_md = (
            f"{icon} **{title}**\n\n"
            f"**Lý do:** {ly_do}\n\n"
            f"**Mức độ nguy hiểm:** {danger}\n\n"
            f"💡 **Lời khuyên:** {loi_khuyen}\n\n"
            f"*Dạ, cháu chỉ là AI — Cô/Chú hãy hỏi thêm người thân hoặc cơ quan chức năng nếu cần.*"
        )
        return response_md, masked_text

    except Exception as e:
        # Fallback khi LLM lỗi hoặc JSON lỗi
        response_md = (
            f"⚠️ **Dạ, cháu đã phân tích xong nhưng định dạng kết quả có chút trục trặc...**\n\n"
            f"Lỗi chi tiết: `{str(e)}`\n\n"
            f"Cô/Chú vui lòng thử lại hoặc hỏi người thân nhé ạ."
        )
        return response_md, masked_text