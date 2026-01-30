# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import zxingcpp
import numpy as np
import pandas as pd
import math
from PIL import Image

# ==================== 1. 依赖库检查 ====================
try:
    from streamlit_paste_button import paste_image_button
except ImportError:
    st.error("请先安装依赖库: pip install streamlit-paste-button")
    st.stop()

# ==================== 2. 页面配置与 CSS 样式 ====================
st.set_page_config(page_title="PDF417 扫码专家", page_icon="💳", layout="wide")

st.markdown("""
    <style>
        /* 标题居中 */
        .main h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        
        /* 表单内容左对齐 */
        .stMarkdown, .stCaption, .stCodeBlock, div[data-testid="stExpander"] {
            text-align: left !important;
        }

        /* 极简间距优化 */
        .block-container {
            padding: 1rem 1rem;
        }

        /* 相机组件全宽 */
        div[data-testid="stCameraInput"] {
            width: 100% !important;
        }

        /* Zint 专用区域：取消手动缩放，优化视觉 */
        textarea {
            resize: none !important;
            font-family: 'Courier New', monospace !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== 3. 核心算法逻辑 ====================

def get_zint_escaped_str(raw_bytes):
    """
    生成 Zint 专用转义文本。
    严格保留原始空格，将不可见字符转换为 \\xHH 格式。
    """
    escaped_output = ""
    for byte in raw_bytes:
        # ASCII 32 是空格，32-126 是可见字符
        if 32 <= byte <= 126:
            char = chr(byte)
            # 转义反斜杠本身，防止 Zint 解析错误
            if char == "\\":
                escaped_output += "\\\\"
            else:
                escaped_output += char
        else:
            # 使用双反斜杠确保 Python 字符串不报错，输出为纯文本 \xHH
            escaped_output += f"\\x{byte:02X}"
    return escaped_output

def get_hex_dump_str(raw_bytes):
    """生成易读的 HEX 数据视图"""
    output = []
    output.append(f"📦 数据长度: {len(raw_bytes)} 字节")
    output.append("-" * 50)
    hex_str = raw_bytes.hex().upper()
    for i in range(0, len(hex_str), 32):
        chunk = hex_str[i:i+32]
        ascii_chunk = ""
        for j in range(0, len(chunk), 2):
            try:
                byte_val = int(chunk[j:j+2], 16)
                ascii_chunk += chr(byte_val) if 32 <= byte_val <= 126 else "."
            except: ascii_chunk += "."
        output.append(f"{chunk.ljust(32)} | {ascii_chunk}")
    return "\n".join(output)

def try_decode(image):
    try:
        results = zxingcpp.read_barcodes(image)
        for result in results:
            if result.format == zxingcpp.BarcodeFormat.PDF417:
                return True, result
    except: pass
    return False, None

def smart_scan_logic(img):
    """增强版解码逻辑"""
    # 尝试原图、灰度、锐化等候选项
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    candidates = [img, gray, clahe]
    
    found_result = None
    for cand in candidates:
        # 尝试不同角度和缩放
        for scale in [1.0, 1.5]:
            processed = cv2.resize(cand, None, fx=scale, fy=scale) if scale != 1.0 else cand
            success, res = try_decode(processed)
            if success: return res
    return None

# ==================== 4. UI 界面实现 ====================

st.title("💳 PDF417 扫码专家")

tab1, tab2 = st.tabs(["📸 网页相机", "📱 全屏上传 / 粘贴"])

target_image = None

with tab1:
    camera_file = st.camera_input("扫描条码", label_visibility="collapsed")
    if camera_file:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)

with tab2:
    col_p, col_u = st.columns([1, 2])
    with col_p:
        # 粘贴按钮
        paste_result = paste_image_button(
            label="📋 粘贴图片", 
            background_color="#212529", 
            hover_background_color="#495057"
        )
    with col_u:
        upload_file = st.file_uploader("全屏拍照上传", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if paste_result.image_data is not None:
        target_image = cv2.cvtColor(np.array(paste_result.image_data), cv2.COLOR_RGB2BGR)
    elif upload_file:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)

if target_image is not None:
    st.divider()
    with st.spinner("正在分析条码数据..."):
        result = smart_scan_logic(target_image)
    
    if result:
        st.success("🎉 识别成功")
        # 提取原始字节
        raw_data = result.bytes if result.bytes else result.text.encode('latin-1', errors='ignore')
        
        # --- Zint 专用文本展示区 ---
        st.subheader("🚀 Zint 专用转义文本 (保留空格)")
        zint_text = get_zint_escaped_str(raw_data)
        
        # 动态计算高度以消除滚动条：假设每 80 字符一行，每行约 25px
        calc_height = max(100, (len(zint_text) // 80 + 2) * 25)
        
        st.text_area(
            label="可以直接复制到生成器的内容：",
            value=zint_text,
            height=calc_height,
            label_visibility="collapsed"
        )
        st.caption("注：已自动将特殊控制符转换为 \\xHH 格式，并保留了所有原始空格。")

        # --- 其他详细信息 ---
        with st.expander("查看底层 HEX 数据与长度分析", expanded=True):
            st.code(get_hex_dump_str(raw_data), language="text")
            st.markdown(f"**总数据长度:** `{len(raw_data)} 字节`")

        if st.button("🔄 扫描下一张", type="primary"):
            st.rerun()
    else:
        st.error("❌ 未能识别条码，请确保对焦清晰并靠近条码。")

# 侧边栏说明（可选）
with st.sidebar:
    st.header("关于 Zint 转义")
    st.info("犹他州 (UT) 驾照条码通常包含不可见字符（如 RS, GS, CR）。本工具将其转义为 Zint 引擎可识别的标准格式。")
