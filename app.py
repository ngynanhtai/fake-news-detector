"""
app.py — Giao diện Streamlit: Hybrid ML + LLM với vòng feedback → retrain
"""

import os
import pandas as pd
import streamlit as st
import ml_model
import json
from llm_engine import analyze_news
from audit_logger import log_interaction
from data_collector import collect_news, RSS_SOURCES

# ── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kiểm Tra Tin Tức Cho Người Cao Tuổi",
    page_icon="🛡️",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.big-font { font-size:20px!important; font-weight:bold; }

/* Viền trái cho khu vực kết quả — dùng border-left trên container */
[data-result-box="warn"] > div:first-child  { border-left: 5px solid #e53935; padding-left: 14px; }
[data-result-box="safe"] > div:first-child  { border-left: 5px solid #43a047; padding-left: 14px; }

.badge-ml  { background:#e3f2fd; color:#1565c0; padding:2px 10px;
             border-radius:12px; font-size:13px; font-weight:600; }
.badge-llm { background:#fce4ec; color:#880e4f; padding:2px 10px;
             border-radius:12px; font-size:13px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("result_md",     None),
    ("masked_text",   None),
    ("source",        None),
    ("ml_result",     None),
    ("final_label",   -1),
    ("feedback_done", False),
    ("raw_input",     ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Hàm xử lý feedback ───────────────────────────────────────────────────────
def _handle_feedback(feedback: str):
    final_label = st.session_state.final_label
    correct_label = final_label if feedback == "Good" else (
        1 - final_label if final_label in (0, 1) else 0
    )

    raw_text = st.session_state.raw_input

    with st.spinner("🔄 Đang cập nhật hệ thống học máy..."):
        ml_model.save_feedback_sample(
            raw_text,
            correct_label
        )

    log_interaction(
        masked_text=st.session_state.masked_text or "",
        ai_response=st.session_state.result_md or "",
        feedback=feedback,
        source=st.session_state.source or "unknown",
        ml_label=st.session_state.ml_result.get("label", -1) if st.session_state.ml_result else -1,
        ml_confidence=st.session_state.ml_result.get("confidence", 0.0) if st.session_state.ml_result else 0.0,
        final_label=correct_label,
    )

    st.session_state.feedback_done = True
    st.rerun()


# ── Tiêu đề ───────────────────────────────────────────────────────────────────
st.title("🛡️ Trợ Lý AI Nhận Diện Tin Lừa Đảo")
st.markdown(
    '<p class="big-font">Chào Cô/Chú! Dán tin nhắn lạ vào đây để cháu kiểm tra giúp nhé:</p>',
    unsafe_allow_html=True,
)

noi_dung = st.text_area("Nhập nội dung tin tức cần kiểm tra", height=150, placeholder="Ví dụ: Bấm vào link để nhận 1 tỷ đồng...", label_visibility="collapsed")

# ── Nút kiểm tra ─────────────────────────────────────────────────────────────
if st.button("🔍 Kiểm Tra Ngay", use_container_width=True):
    if not noi_dung.strip():
        st.warning("Vui lòng nhập nội dung trước khi kiểm tra!")
    else:
        st.session_state.feedback_done = False
        st.session_state.raw_input     = noi_dung

        with st.spinner("⏳ Cháu đang đọc và phân tích tin này..."):
            ml_result = ml_model.predict(noi_dung)
            st.session_state.ml_result = ml_result

            label = ml_result["label"]
            confidence = ml_result["confidence"]

            label_name = (
                "REAL"
                if label == 1
                else "FAKE"
            )

            print(
                f"""
            ================ ML RESULT ================
            Label       : {label_name}
            Confidence  : {ml_result['confidence']:.4f}
            Fake Prob   : {ml_result['proba_fake']:.4f}
            Real Prob   : {ml_result['proba_real']:.4f}
            LLM Needed  : {not ml_result['is_certain']}
            ===========================================
            """
            )

            llm_response, masked_text = analyze_news(
                noi_dung,
                label,
                confidence
            )

            st.session_state.result_md = llm_response
            st.session_state.masked_text = masked_text
            st.session_state.source = "hybrid"
            st.session_state.final_label = label
            st.session_state.ml_result = ml_result

# ── Hiển thị kết quả ─────────────────────────────────────────────────────────
if st.session_state.result_md:
    st.markdown("---")

    # Badge nguồn
    if st.session_state.source == "hybrid":
        st.markdown('<span class="badge-ml">⚡ Phân tích bởi "Hybrid ML + LLM"</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-llm">🧠 Phân tích bởi LLM (sâu hơn)</span>',
                    unsafe_allow_html=True)

    st.write("")  # khoảng cách nhỏ

    # Dùng st.container() thay vì raw <div> để markdown render đúng
    is_warn = st.session_state.final_label == 0
    result_container = st.container(border=True)
    with result_container:
        if is_warn:
            st.error(st.session_state.result_md)
        else:
            st.success(st.session_state.result_md)

    # ── Feedback ─────────────────────────────────────────────────────────────
    st.markdown("---")
    if not st.session_state.feedback_done:
        st.markdown("**Đánh giá này có đúng không, Cô/Chú?**")
        st.caption("Phản hồi của Cô/Chú giúp hệ thống học thêm và chính xác hơn theo thời gian.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Đúng rồi / Rất hữu ích", use_container_width=True):
                _handle_feedback("Good")
        with col2:
            if st.button("👎 Sai rồi / Cần xem lại", use_container_width=True):
                _handle_feedback("Bad")
    else:
        st.success("✨ Dạ, cháu đã ghi nhận phản hồi của Cô/Chú. Cảm ơn rất nhiều ạ!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*🔒 Bảo mật: Số điện thoại, số tài khoản ngân hàng được tự động ẩn "
    "trước khi gửi đi phân tích.*"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Sức khỏe Mô hình Máy học (ML)")
    
    # 1. Tính tổng số mẫu thực tế trong database CSV
    total_samples = 0
    if os.path.exists(ml_model.DATA_FILE):
        try:
            df_status = pd.read_csv(ml_model.DATA_FILE, encoding="utf-8-sig")
            total_samples = len(df_status)
        except:
            pass

    # 2. Lấy giá trị chính xác và mới nhất từ file metric ra hiển thị
    metrics = ml_model.get_metrics()

    current_accuracy = metrics["accuracy"]
    current_precision = metrics["precision"]
    current_recall = metrics["recall"]
    current_f1 = metrics["f1"]

    # 3. Hiển thị thông số dạng số và thanh Progress Bar
    st.metric(
        label="Độ chính xác hiện tại (Accuracy)", 
        value=f"{current_accuracy * 100:.1f} %",
        delta=f"+{total_samples} mẫu tin tổng hợp" if total_samples > 0 else None
    )
    st.metric(
        "Precision",
        f"{current_precision * 100:.1f}%"
    )

    st.metric(
        "Recall",
        f"{current_recall * 100:.1f}%"
    )

    st.metric(
        "F1 Score",
        f"{current_f1 * 100:.1f}%"
    )
    
    st.progress(int(current_accuracy * 100))
    
    # 4. Cảnh báo trạng thái điều hướng hệ thống
    if current_accuracy < 0.85:
        st.warning(
            f"⚠️ **Hiện trạng:** Độ chính xác thuật toán toán học chưa đạt ngưỡng an toàn (>=85%). "
            f"Hệ thống sẽ chuyển giao dữ liệu sang **LLM phân tích** để bảo vệ Cô/Chú."
        )
    else:
        st.success("✅ **Hiện trạng:** Mô hình toán học đã đủ mạnh để tự động phân tách phần lớn tin tức mà không cần gọi LLM.")

    st.markdown(f"Tổng số mẫu trong DB: `{total_samples}`")
    st.markdown("---")

    st.header("📊 Thống kê hệ thống")

    if os.path.exists(ml_model.DATA_FILE):
        try:
            df = pd.read_csv(ml_model.DATA_FILE, encoding="utf-8-sig")
            st.metric("Tổng mẫu huấn luyện", len(df))
            st.metric("Mẫu tin giả (0)", int((df["nhan"] == 0).sum()))
            st.metric("Mẫu tin thật (1)", int((df["nhan"] == 1).sum()))
        except Exception:
            st.write("Chưa có dữ liệu.")
    else:
        st.write("Chưa có file dữ liệu.")

    st.divider()

    if os.path.exists("audit_log.csv"):
        try:
            df_log = pd.read_csv("audit_log.csv", encoding="utf-8-sig")
            st.metric("Tổng lượt kiểm tra", len(df_log))
            st.metric("Feedback 👍", int((df_log["feedback"] == "Good").sum()))
            st.metric("Feedback 👎", int((df_log["feedback"] == "Bad").sum()))
        except Exception:
            pass

    st.divider()
    if st.button("🔄 Retrain thủ công", use_container_width=True):
        with st.spinner("Đang huấn luyện lại..."):
            ml_model.retrain()
            ml_model.reload_model()
        st.success("✅ Xong!")

    st.divider()
    st.subheader("🌐 Thu thập tin tức")
    st.caption("Lấy tin thật từ báo uy tín để bổ sung dữ liệu huấn luyện.")

    max_per_feed = st.slider("Số bài / nguồn", min_value=5, max_value=50, value=20, step=5)

    # Cho phép chọn nguồn
    all_source_names = [s["name"] for s in RSS_SOURCES]
    selected_names = st.multiselect(
        "Chọn nguồn RSS",
        options=all_source_names,
        default=all_source_names[:6],   # mặc định chọn 6 nguồn đầu
    )

    if st.button("📥 Thu thập ngay", use_container_width=True):
        selected_sources = [s for s in RSS_SOURCES if s["name"] in selected_names]
        if not selected_sources:
            st.warning("Vui lòng chọn ít nhất một nguồn.")
        else:
            progress = st.progress(0, text="Đang kết nối...")
            log_area  = st.empty()
            logs      = []

            for i, source in enumerate(selected_sources):
                progress.progress(
                    int((i + 1) / len(selected_sources) * 100),
                    text=f"🔄 {source['name']}",
                )
                logs.append(f"🔄 {source['name']}...")
                log_area.text("\n".join(logs[-6:]))  # hiện 6 dòng gần nhất

            # Chạy collect thật
            added, skipped = collect_news(
                sources=selected_sources,
                max_per_feed=max_per_feed,
            )

            progress.empty()
            log_area.empty()

            if added > 0:
                st.success(f"✅ Thêm {added} bài mới | Bỏ qua {skipped} trùng")
                # Tự động retrain
                with st.spinner("🤖 Đang retrain lại ML model..."):
                    model_updated, acc_updated = ml_model.retrain()
                    ml_model.reload_model()
                    metrics = ml_model.get_metrics()

                    st.session_state["metrics"] = metrics
                st.success("🤖 Retrain xong! Model đã được cập nhật.")
            else:
                st.info(f"Không có bài mới (bỏ qua {skipped} bài trùng).")

            st.rerun()