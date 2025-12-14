# -*- coding: utf-8 -*-
import cv2
import zxingcpp
import math
import numpy as np
import streamlit as st
import pandas as pd
from io import BytesIO

# --- 1. 图像预处理与识别核心函数 (基于原代码修改) ---

def preprocess_image(img):
    """
    模拟 HAX/专业扫描器的预处理逻辑：
    1. 转灰度
    2. 放大 (让密集条码更容易识别)
    3. 二值化/自适应二值化 (去除背景干扰)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 尝试1: 普通灰度
    yield gray, "Level 1: 灰度原图"
    
    # 尝试2: 放大 2 倍 (针对高密度 PDF417)
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    yield scaled, "Level 2: 2X 放大灰度"
    
    # 尝试3: 二值化 (去除背景花纹干扰)
    _, binary = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)
    yield binary, "Level 3: 2X 放大 + 普通二值化"
    
    # 尝试4: 自适应二值化 (针对光照不均)
    adaptive = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    yield adaptive, "Level 4: 2X 放大 + 自适应二值化"

def get_barcode_data(img_bytes):
    """尝试读取条码，如果失败则返回 None"""
    # 将上传的 BytesIO 对象转换为 OpenCV 图像
    nparr = np.frombuffer(img_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, "❌ 错误: 图片无法加载"

    # 1. 先尝试直接读取原图
    results = zxingcpp.read_barcodes(img)
    for res in results:
        if res.format == zxingcpp.BarcodeFormat.PDF417:
            return res.bytes, "✅ 原图模式下成功识别"

    # 2. 如果失败，进入“增强模式” (HAX 逻辑)
    for i, (processed_img, desc) in enumerate(preprocess_image(img)):
        results = zxingcpp.read_barcodes(processed_img)
        for res in results:
            if res.format == zxingcpp.BarcodeFormat.PDF417:
                return res.bytes, f"✅ 增强模式 ({desc}) 下成功识别"
                
    return None, "❌ 最终失败: 所有增强算法均无法读取该 PDF417。"

# --- 2. 参数逆向计算核心函数 (生成 DataFrame) ---

def calculate_pdf417_params(byte_len):
    """
    根据字节长度，计算所有可能的 PDF417 行列组合，并估算宽高比。
    """
    if byte_len <= 0:
        return pd.DataFrame()

    # AAMVA 标准估算逻辑
    # 1.8 bytes ≈ 1 data codeword (平均值)
    estimated_data_cw = math.ceil(byte_len / 1.8) 
    ecc_cw = 64  # Level 5 Security (AAMVA Standard)
    total_cw = estimated_data_cw + ecc_cw
    
    data = []
    possible_cols = range(9, 21) # AAMVA/DL 常用范围
    
    for cols in possible_cols:
        rows = math.ceil(total_cw / cols)
        
        if rows < 3 or rows > 90: # 规范限制
            continue
            
        # 宽高比估算 (假设行高与模块宽度的比值为 3)
        width_units = (cols + 4) * 17
        height_units = rows * 3 
        ratio = width_units / height_units

        # 备注逻辑
        note = ""
        if cols == 17: note = "⭐ AAMVA 标准"
        elif 16 <= cols <= 18: note = "✅ 常见宽版"
        elif 11 <= cols <= 13: note = "🔹 窄版 (NY/CA风格)"
        
        if 3.0 <= ratio <= 5.0: note += " | 完美比例"
        elif ratio > 6.0: note += " | 扁长条码"
        elif ratio < 2.5: note += " | 正方条码"
        
        data.append({
            "列数 (Cols)": cols,
            "推算行数 (Rows)": rows,
            "估算宽高比 (W/H)": f"{ratio:.1f}",
            "类型备注": note
        })
    
    return pd.DataFrame(data)

# --- 3. Streamlit UI 界面 ---

st.set_page_config(layout="wide", page_title="PDF417 条码分析工具")

# 初始化 session_state
if 'last_scan_bytes' not in st.session_state:
    st.session_state['last_scan_bytes'] = 0

st.title("🆔 PDF417 条码分析与逆向工具")

# --- 扫码区域 ---
st.subheader("📸 1. 扫码提取数据长度 (HAX 增强模式)")

uploaded_file = st.file_uploader("上传 PDF417 条码图片 (如身份证背面扫描件)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 复制文件流以供多次读取或显示
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    # 显示图片
    st.image(file_bytes.getvalue(), caption='上传的图片', use_column_width=True)
    
    st.markdown("---")
    
    # 进行识别
    with st.spinner("正在尝试多级图像增强识别..."):
        # 必须重新定位文件流到开始，因为 st.image 可能已读取了一部分
        file_bytes.seek(0) 
        raw_bytes, status_msg = get_barcode_data(file_bytes)
        
    st.info(status_msg)
    
    if raw_bytes:
        data_len = len(raw_bytes)
        st.success(f"🎉 成功提取数据! 原始字节长度: {data_len} bytes")
        # 将长度保存到 session_state 供计算器使用
        st.session_state['last_scan_bytes'] = data_len 
        
        # 可以在此处显示解码后的文本（如果需要）
        # st.code(raw_bytes.decode('latin-1', errors='ignore'), language='text')

# --- 参数逆向计算器区域 ---
st.divider()
st.subheader("📐 2. 参数逆向计算器")
st.caption("基于 Level 5 (64 CW ECC) 和 AAMVA 1.8 bytes/CW 估算。")

# 自动填入或手动输入
default_len = st.session_state['last_scan_bytes']

with st.expander("点击输入数据长度并查看结果表", expanded=True):
    col_input, col_info = st.columns([1, 2])
    
    with col_input:
        byte_input = st.number_input(
            "原始数据字节长度 (Raw Data Length)", 
            min_value=0, 
            value=default_len,
            step=1,
            key="byte_input_key",
            help="输入 HAX 工具或扫码器读出的原始数据字节数。"
        )

    if byte_input > 0:
        df_result = calculate_pdf417_params(byte_input)
        
        with col_info:
            st.markdown(f"**分析长度:** `{byte_input} bytes` | **总码字估算 (CW):** `{math.ceil(byte_input / 1.8) + 64}`")
            
            best_row = df_result[df_result['列数 (Cols)'] == 17]
            if not best_row.empty:
                rec_rows = best_row.iloc[0]['推算行数 (Rows)']
                st.success(f"💡 AAMVA 推荐设置: **Cols=17, Rows={rec_rows}** (最标准的制作参数)")
            else:
                st.warning("数据量较大，请参考下方表格寻找最佳比例。")

        # 结果表格展示
        st.dataframe(
            df_result,
            use_container_width=True,
            hide_index=True,
            column_config={
                "列数 (Cols)": st.column_config.NumberColumn(format="%d"),
                "推算行数 (Rows)": st.column_config.NumberColumn(format="%d"),
                "估算宽高比 (W/H)": st.column_config.TextColumn("W/H 比例"),
                "类型备注": st.column_config.TextColumn("备注"),
            }
        )
    else:
        with col_info:
             st.info("请上传图片或手动输入数据长度（Raw Data Length）来计算最佳行列参数。")
