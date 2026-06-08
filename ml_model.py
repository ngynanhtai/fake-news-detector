"""
ml_model.py — ML model layer (TF-IDF + Logistic Regression)

Trả về:
  - predict(text) -> {"label": 0|1, "confidence": float, "is_certain": bool}
  - retrain()     -> Nạp lại toàn bộ CSV và huấn luyện lại từ đầu
"""

import os
import joblib
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ── Đường dẫn file ──────────────────────────────────────────────────────────
DATA_FILE  = "du_lieu_tin_tuc.csv"
MODEL_FILE = "mo_hinh_ai_tin_gia.pkl"
FEEDBACK_FILE = "feedback_samples.csv"

# Ngưỡng tự tin: nếu < CONFIDENCE_THRESHOLD thì chuyển sang LLM
CONFIDENCE_THRESHOLD = 0.85


# ── Tạo dữ liệu mẫu nếu chưa có ─────────────────────────────────────────────
def _tao_du_lieu_mau():
    if os.path.exists(DATA_FILE):
        return
    print(f"⚠️  Chưa có {DATA_FILE}. Đang tạo dữ liệu mẫu...")
    mau = {
        "tieu_de": [
            "Báo động: Ăn tỏi chữa bách bệnh, ung thư cũng khỏi!",
            "Bộ Y tế khuyến cáo tiêm vắc xin phòng cúm mùa",
            "CẤP BÁCH: Chuyển tiền ngay vào quỹ hỗ trợ người cao tuổi để nhận x3 tiền lãi!!!",
            "Hà Nội: Dự báo thời tiết cuối tuần có mưa dông",
            "Phát hiện thuốc tiên chữa dứt điểm tiểu đường chỉ trong 1 đêm",
            "Cách nhận biết các tin nhắn lừa đảo qua mạng xã hội",
            "Bí mật động trời: Nước chanh nóng tiêu diệt virus 100%!",
            "Khởi tố nhóm đối tượng lừa đảo chiếm đoạt tài sản qua mạng",
            "MIỄN PHÍ: Nhận ngay 500k từ ngân hàng, bấm link để nhận!!!",
            "Hội thảo sức khỏe người cao tuổi tại Trung tâm Y tế Quận 1",
        ],
        "noi_dung": [
            "Bài thuốc bí truyền từ tỏi đen giúp chữa mọi loại bệnh tật...",
            "Để phòng ngừa dịch bệnh mùa đông xuân, người dân nên chủ động...",
            "Cơ hội duy nhất trong ngày! Bấm vào link và chuyển khoản ngay để...",
            "Trung tâm Khí tượng Thủy văn cho biết cuối tuần này nhiệt độ giảm...",
            "Không cần đến bệnh viện, chỉ cần uống loại lá này là hết bệnh...",
            "Theo các chuyên gia an ninh mạng, tuyệt đối không bấm vào link lạ...",
            "Chia sẻ ngay kẻo lỡ! Chanh nóng pha mật ong diệt sạch virus...",
            "Công an đã bắt giữ nhóm đối tượng chuyên gọi điện giả danh...",
            "Chương trình ưu đãi đặc biệt chỉ hôm nay, chuyển khoản xác nhận...",
            "Hội thảo miễn phí với sự tham gia của các bác sĩ đầu ngành...",
        ],
        "nhan": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # 1=Thật, 0=Giả/Lừa đảo
    }
    pd.DataFrame(mau).to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    print("✅ Đã tạo xong dữ liệu mẫu.\n")

# ── Load Training Data ──────────────────────────────────────────────────
def load_training_data():
    """
    Gộp:
    - dataset gốc
    - feedback từ người dùng

    Trả về DataFrame chuẩn:
    [tieu_de, noi_dung, nhan]
    """

    datasets = []

    # Dataset gốc
    if os.path.exists(DATA_FILE):
        try:
            df_main = pd.read_csv(
                DATA_FILE,
                encoding="utf-8-sig"
            )

            df_main = df_main[
                ["tieu_de", "noi_dung", "nhan"]
            ]

            datasets.append(df_main)

        except Exception as e:
            print("Lỗi đọc DATA_FILE:", e)

    # Feedback dataset
    if os.path.exists(FEEDBACK_FILE):
        try:
            df_feedback = pd.read_csv(
                FEEDBACK_FILE,
                encoding="utf-8-sig"
            )

            if len(df_feedback) > 0:

                df_feedback = df_feedback.rename(
                    columns={
                        "text": "noi_dung",
                        "label": "nhan"
                    }
                )

                df_feedback["tieu_de"] = "Feedback User"

                df_feedback = df_feedback[
                    ["tieu_de", "noi_dung", "nhan"]
                ]

                datasets.append(df_feedback)

        except Exception as e:
            print("Lỗi đọc FEEDBACK_FILE:", e)

    if not datasets:
        return pd.DataFrame(
            columns=[
                "tieu_de",
                "noi_dung",
                "nhan"
            ]
        )

    df = pd.concat(
        datasets,
        ignore_index=True
    )

    # Loại trùng
    df = df.drop_duplicates(
        subset=["noi_dung"],
        keep="last"
    )

    return df

# ── Huấn luyện và lưu model ──────────────────────────────────────────────────
def retrain():
    """
    Huấn luyện lại mô hình toán học từ file CSV dữ liệu.
    Trả về: (pipeline_model, accuracy_score)
    """
    _tao_du_lieu_mau()  # Đảm bảo file tồn tại
    
    if not os.path.exists(DATA_FILE):
        return None, 0.0
        
    df = load_training_data()
    if len(df) < 5:  # Nếu dữ liệu quá ít thì không chia tập test, mặc định accuracy thấp
        X = df['noi_dung']
        y = df['nhan']
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_FILE)
        return pipeline, 0.50  # Mặc định 50% khi quá ít dữ liệu
        
    # Chia dữ liệu huấn luyện thành 80% Train / 20% Test để đo Accuracy độc lập
    X_train, X_test, y_train, y_test = train_test_split(
        df['noi_dung'], df['nhan'], test_size=0.2, random_state=42, stratify=df['nhan']
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    
    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)
    
    # Tính điểm độ chính xác (Accuracy) dựa trên tập Test
    y_pred = pipeline.predict(X_test)

    accuracy = float(
        pipeline.score(
            X_test,
            y_test
        )
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    with open(
        "model_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "dataset_size": len(df),
                "test_size": len(X_test),
                "fake_news": int((df["nhan"] == 0).sum()),
                "real_news": int((df["nhan"] == 1).sum()),
                "fake_detected": int(tn),
                "fake_missed": int(fp),
                "real_detected": int(fn),
                "real_misclassified": int(tp)
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    
    # Lưu mô hình đã tối ưu
    joblib.dump(pipeline, MODEL_FILE)
    print(f"🤖 Đã Huấn luyện lại ML Model. Tập Test Accuracy: {accuracy*100:.1f}%")
    
    return pipeline, accuracy


# ── Load model (lazy) ────────────────────────────────────────────────────────
_cached_model: Pipeline | None = None


def _load_model() -> Pipeline | None:
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if os.path.exists(MODEL_FILE):
        _cached_model = joblib.load(MODEL_FILE)
        return _cached_model
    # Chưa có model → tự train lần đầu
    _cached_model = retrain()
    return _cached_model


def reload_model():
    """Gọi sau khi retrain() để app.py nhận model mới nhất."""
    global _cached_model
    _cached_model = None
    return _load_model()

def get_metrics():

    default_metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "dataset_size": 0,
        "test_size": 0
    }

    if not os.path.exists("model_metrics.json"):
        return default_metrics

    try:
        with open(
            "model_metrics.json",
            "r",
            encoding="utf-8"
        ) as f:
            return {
                **default_metrics,
                **json.load(f)
            }

    except Exception:
        return default_metrics

# ── Dự đoán chính ────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    """
    Trả về dict:
      label       : 0 (Giả/Lừa đảo) | 1 (Có vẻ thật)
      confidence  : float 0–1  (xác suất class được chọn)
      is_certain  : bool  (True nếu confidence >= CONFIDENCE_THRESHOLD)
      proba_fake  : float  (xác suất là giả)
      proba_real  : float  (xác suất là thật)
    """
    model = _load_model()
    if model is None:
        return {"label": -1, "confidence": 0.0, "is_certain": False,
                "proba_fake": 0.5, "proba_real": 0.5, "error": "Model chưa sẵn sàng"}

    probas = model.predict_proba([text])[0]
    classes = model.classes_           # thường là [0, 1]

    prob_dict = dict(zip(classes, probas))
    proba_real = prob_dict.get(1, 0.0)
    proba_fake = prob_dict.get(0, 0.0)

    label      = int(model.predict([text])[0])
    confidence = float(max(probas))

    return {
        "label": label,
        "confidence": confidence,
        "is_certain": confidence >= CONFIDENCE_THRESHOLD,
        "proba_fake": proba_fake,
        "proba_real": proba_real,
    }


# ── Ghi nhãn mới từ feedback ─────────────────────────────────────────────────
def save_labeled_sample(tieu_de: str, noi_dung: str, nhan: int):
    """
    Thêm một dòng vào CSV.
    nhan: 1 = thật (feedback 👍 với kết quả ML=1 hoặc 👎 với kết quả ML=0)
          0 = giả
    """
    _tao_du_lieu_mau()  # Đảm bảo file tồn tại
    df_new = pd.DataFrame([{"tieu_de": tieu_de, "noi_dung": noi_dung, "nhan": nhan}])
    df_new.to_csv(DATA_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    print(f"💾 Đã lưu mẫu mới (nhãn={nhan}). Tổng: {len(pd.read_csv(DATA_FILE))} mẫu.")


# ── Save feedback file ─────────────────────────────────────────────────
def save_feedback_sample(
    text: str,
    corrected_label: int
):
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "label": corrected_label
    }])

    if os.path.exists(FEEDBACK_FILE):
        row.to_csv(
            FEEDBACK_FILE,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig"
        )
    else:
        row.to_csv(
            FEEDBACK_FILE,
            index=False,
            encoding="utf-8-sig"
        )
