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
        /* 标题居中 */
        .main h1 { text-align: center; color: #333; }
        
        .block-container { padding: 1rem 0.5rem; }
        
        /* 强制 st.code 区域取消滚动条，高度自适应 */
        code {
            white-space: pre-wrap !important;
            word-break: break-all !important;
        }
        pre {
            white-space: pre-wrap !important;
            overflow: visible !important;
            height: auto !important;
        }

        /* 相机与按钮样式优化 */
        div[data-testid="stCameraInput"] { width: 100% !important; }
        div.stButton > button:first-child { width: 100%; }
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
            if char == "\\":
                escaped_output += "\\\\"  # 转义反斜杠本身
            else:
                escaped_output += char
        else:
            # 双反斜杠避免 Unicode 转义报错
            escaped_output += f"\\x{byte:02X}"
    return escaped_output

def get_hex_dump_str(raw_bytes):
    """生成 HEX 数据视图"""
    output = []
    output.append(f"📦 数据长度: {len(raw_bytes)} 字节")
    output.append("-" * 50)
    hex_str = raw_bytes.hex().upper()
    for i in range(0, len(hex_str), 32):
        chunk = hex_str[i:i+32]
        ascii_chunk = "".join([chr(int(chunk[j:j+2], 16)) if 32 <= int(chunk[j:j+2], 16) <= 126 else "." for j in range(0, len(chunk), 2)])
        output.append(f"{chunk.ljust(32)} | {ascii_chunk}")
    return "\n".join(output)

def smart_scan_logic(img):
    """增强版解码逻辑"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    for cand in [img, gray, clahe]:
        for scale in [1.0, 1.5]:
            p = cv2.resize(cand, None, fx=scale, fy=scale) if scale != 1.0 else cand
            results = zxingcpp.read_barcodes(p)
            for res in results:
                if res.format == zxingcpp.BarcodeFormat.PDF417:
                    return res
    return None

# ==================== 2. 网页界面区 ====================

st.title("💳 PDF417 扫码专家")

tab1, tab2 = st.tabs(["📸 网页相机", "📱 全屏上传 / 粘贴"])

target_image = None
with tab1:
    camera_file = st.camera_input("扫描条码", label_visibility="collapsed")
    if camera_file:
        target_image = cv2.imdecode(np.asarray(bytearray(camera_file.read()), dtype=np.uint8), 1)

with tab2:
    col_p, col_u = st.columns([1, 2])
    with col_p:
        paste_result = paste_image_button(label="📋 粘贴剪贴板图片", background_color="#FF4B4B")
    with col_u:
        upload_file = st.file_uploader("拍照上传", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if paste_result.image_data is not None:
        target_image = cv2.cvtColor(np.array(paste_result.image_data), cv2.COLOR_RGB2BGR)
    elif upload_file:
        target_image = cv2.imdecode(np.asarray(bytearray(upload_file.read()), dtype=np.uint8), 1)

if target_image is not None:
    st.divider()
    result = smart_scan_logic(target_image)
    
    if result:
        st.success("🎉 识别成功")
        raw_data = result.bytes if result.bytes else result.text.encode('latin-1', errors='ignore')
        
        # --- Zint 专用展示区 (自带右上角复制按钮) ---
        st.subheader("🚀 Zint 专用转义文本 (保留空格)")
        zint_escaped = get_zint_escaped_str(raw_data)
        
        # 使用 st.code 会自动在右上角生成复制按钮
        st.code(zint_escaped, language="text")
        st.caption("注：点击右上角按钮即可快速复制。已保留所有原始空格并转义控制符。")

        # --- 原始数据展示 ---
        st.info(f"📊 字节长度: **{len(raw_data)}** bytes")
        with st.expander("查看底层 HEX 数据 (点击展开)", expanded=False):
            st.code(get_hex_dump_str(raw_data), language="text")

        if st.button("🔄 扫描下一张", type="primary"):
            st.rerun()
    else:
        st.error("❌ 未识别。请确保对焦清晰并靠近条码。")
