# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import zxingcpp
import numpy as np
import pandas as pd
import math
from PIL import Image

# ==================== 引入粘贴组件库 ====================
try:
    from streamlit_paste_button import paste_image_button
except ImportError:
    st.error("请先安装依赖库: pip install streamlit-paste-button")
    st.stop()

# ==================== 0. 页面配置与 CSS 样式优化 ====================

st.set_page_config(page_title="PDF417 扫码专家", page_icon="💳", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding: 1rem 0.5rem;
        }
        div[data-testid="stCameraInput"] {
            width: 100% !important;
        }
        div[data-testid="stCameraInput"] video {
            border-radius: 12px !important;
            width: 100% !important;
            object-fit: cover;
        }
        button[data-baseweb="tab"] div {
            font-size: 1.1em !important;
            padding: 1em !important;
        }
        div.stButton > button:first-child {
            width: 100%;
        }
        /* 优化 Zint 文本框展示 */
        textarea {
            font-family: monospace !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== 1. 核心算法区 ====================

def get_zint_escaped_str(raw_bytes):
    """
    生成 Zint 专用转义文本。
    严格保留原始空格，将不可见字符转换为 \\xHH 格式。
    """
    escaped_output = ""
    for byte in raw_bytes:
        # ASCII 32 是空格，32-126 是常规可见字符
        if 32 <= byte <= 126:
            char = chr(byte)
            # 转义反斜杠本身，防止冲突
            if char == "\\":
                escaped_output += "\\\\"
            else:
                escaped_output += char
        else:
            # 使用双反斜杠确保输出文本为 \xHH 格式，避免 Python 自身的转义报错
            escaped_output += f"\\x{byte:02X}"
    return escaped_output

def get_hex_dump_str(raw_bytes):
    """生成易读的 HEX 数据视图"""
    output = []
    output.append(f"📦 数据长度: {len(raw_bytes)} 字节")
    output.append("-" * 50)
    
    try:
        hex_str = raw_bytes.hex().upper()
    except AttributeError:
        hex_str = raw_bytes.encode('utf-8').hex().upper()

    for i in range(0, len(hex_str), 32):
        chunk = hex_str[i:i+32]
        ascii_chunk = ""
        for j in range(0, len(chunk), 2):
            try:
                byte_val = int(chunk[j:j+2], 16)
                ascii_chunk += chr(byte_val) if 32 <= byte_val <= 126 else "."
            except ValueError:
                ascii_chunk += "?"
        output.append(f"{chunk.ljust(32)} | {ascii_chunk}")
    return "\n".join(output)

def preprocess_image_candidates(img):
    candidates = [("原图", img)]
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    candidates.append(("灰度", gray))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    candidates.append(("CLAHE", enhanced))
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    candidates.append(("锐化", sharpened))
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("二值(OTSU)", binary))
    return candidates

def try_decode(image):
    try:
        results = zxingcpp.read_barcodes(image)
        for result in results:
            if result.format == zxingcpp.BarcodeFormat.PDF417:
                return True, result
    except Exception:
        pass
    return False, None

def smart_scan_logic(original_img):
    base_candidates = preprocess_image_candidates(original_img)
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(base_candidates) * 3
    step = 0
    found_result = None

    for mode_name, img_candidate in base_candidates:
        transforms = [
            ("正常", lambda x: x),
            ("旋转90°", lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)),
            ("放大1.5x", lambda x: cv2.resize(x, None, fx=1.5, fy=1.5)),
        ]
        for trans_name, trans_func in transforms:
            step += 1
            progress_bar.progress(min(step / total_steps, 0.95))
            status_text.caption(f"正在分析: {mode_name} / {trans_name}...")
            try:
                processed_img = trans_func(img_candidate)
                success, result = try_decode(processed_img)
                if success:
                    found_result = result
                    status_text.success(f"✅ 识别成功! (模式: {mode_name} - {trans_name})")
                    progress_bar.progress(1.0)
                    break
            except Exception:
                continue
        if found_result: break
    if not found_result:
        status_text.error("❌ 未识别。请靠近一点，确保光线充足且对焦清晰。")
        progress_bar.empty()
    return found_result

def calculate_pdf417_params(byte_len):
    if byte_len <= 0: return pd.DataFrame()
    estimated_data_cw = math.ceil(byte_len / 1.8) 
    ecc_cw = 64  
    total_cw = estimated_data_cw + ecc_cw
    data = []
    for cols in range(9, 21):
        rows = math.ceil(total_cw / cols)
        if rows < 3 or rows > 90: continue
        width_units = (cols + 4) * 17
        height_units = rows * 3 
        ratio = width_units / height_units
        note = ""
        if cols == 17: note = "⭐ AAMVA 标准"
        elif 11 <= cols <= 13: note = "🔹 窄版 (NY/CA风格)"
        if 3.0 <= ratio <= 5.0: note += " | 完美比例"
        data.append({
            "列数 (Cols)": cols,
            "推算行数 (Rows)": rows,
            "估算宽高比 (W/H)": f"{ratio:.1f}",
            "类型备注": note
        })
    return pd.DataFrame(data)

# ==================== 2. 网页界面区 ====================

st.title("💳 PDF417 扫码专家")

tab1, tab2 = st.tabs(["📸 网页小窗 (快速)", "📱 全屏拍照 / 粘贴"])

target_image = None
data_source = None

with tab1:
    st.caption("适用于光线好、条码清晰的简单场景。请横屏使用。")
    camera_file = st.camera_input("请对准条码", label_visibility="collapsed")
    if camera_file:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)
        data_source = "网页相机"

with tab2:
    st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; margin-bottom: 20px;">
            <h4 style="margin: 0; color: #2e7d32; font-size: 1.1rem;">🚀 多种上传方式：</h4>
            <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #333;">
                1. <b>粘贴图片</b>：截图后点击“粘贴”按钮。<br>
                2. <b>全屏拍照</b>：点击“浏览文件” -> 选择“拍照/相机”。
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_paste, col_upload = st.columns([1, 2])
    with col_paste:
        paste_result = paste_image_button(
            label="📋 粘贴剪贴板图片",
            background_color="#FF4B4B",
            hover_background_color="#FF0000",
            text_color="#FFFFFF",
            errors="ignore"
        )
    with col_upload:
        upload_file = st.file_uploader("启动全屏相机", type=["jpg", "png", "jpeg", "heic"], label_visibility="collapsed")

    if paste_result.image_data is not None:
        target_image = cv2.cvtColor(np.array(paste_result.image_data), cv2.COLOR_RGB2BGR)
        data_source = "剪贴板粘贴"
    elif upload_file:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)
        data_source = "文件上传"

if target_image is not None:
    st.divider()
    with st.expander(f"查看当前处理图片 ({data_source})", expanded=False):
        st_display_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        st.image(st_display_img, use_column_width=True)

    result = smart_scan_logic(target_image)
    
    if result:
        st.success("🎉 解码成功！")
        raw_data = result.bytes if result.bytes else result.text.encode('latin-1', errors='ignore')
        
        # === 新增展示：Zint 专用转义文本 ===
        st.subheader("🚀 Zint 专用转义文本 (保留空格)")
        zint_escaped = get_zint_escaped_str(raw_data)
        st.text_area(label="可以直接复制的内容", value=zint_escaped, height=150, label_visibility="collapsed")
        st.caption("注：已自动将特殊控制符转换为 \\xHH 格式，并保留了所有原始空格。")

        data_type = "二进制 (Bytes)" if isinstance(result.bytes, bytes) else "文本 (Text)"
        st.info(f"数据类型: **{data_type}** | 字节长度: **{len(raw_data)}** bytes")
        
        if result.text and data_type == "文本 (Text)":
            st.subheader("📝 文本内容")
            st.code(result.text, language="text")
        elif data_type == "二进制 (Bytes)":
            st.subheader("📝 尝试解码文本 (Latin-1)")
            try:
                st.code(raw_data.decode('latin-1'), language="text")
            except:
                st.code("无法以 Latin-1 解码", language="text")

        with st.expander("查看底层 HEX 数据 (点击展开)", expanded=False):
            hex_str = get_hex_dump_str(raw_data)
            st.code(hex_str, language="text")

        st.subheader("📐 PDF417 参数逆向计算 (AAMVA)")
        df_params = calculate_pdf417_params(len(raw_data))
        col_summary, col_table_content = st.columns([1, 2])

        with col_summary:
            st.markdown(f"**分析长度:** `{len(raw_data)} bytes`")
            st.markdown(f"**ECC 安全等级:** `Level 5 (64 Codewords)`")
            best_row = df_params[df_params['列数 (Cols)'] == 17]
            if not best_row.empty:
                st.success(f"💡 AAMVA 推荐: **Cols=17, Rows={best_row.iloc[0]['推算行数 (Rows)']}**")

        with col_table_content:
            st.dataframe(df_params, use_container_width=True, hide_index=True)

        st.divider()
        if st.button("🔄 扫描下一张", type="primary"):
            st.rerun()
