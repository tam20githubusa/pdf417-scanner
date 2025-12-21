# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import zxingcpp
import numpy as np
import pandas as pd
import math
from PIL import Image

# ==================== 新增：引入粘贴组件库 ====================
# 如果报错，请先运行: pip install streamlit-paste-button
try:
    from streamlit_paste_button import paste_image_button
except ImportError:
    st.error("请先安装依赖库: pip install streamlit-paste-button")
    st.stop()

# ==================== 0. 页面配置与 CSS 样式优化 ====================

st.set_page_config(page_title="PDF417 扫码专家", page_icon="💳", layout="wide")

# 注入 CSS：强制去除边距，放大相机，优化提示框
st.markdown("""
    <style>
        /* 1. 极大幅度减少页面四周的留白 */
        .block-container {
            padding: 1rem 0.5rem;
        }
        
        /* 2. 强制网页相机组件占满 100% 宽度 */
        div[data-testid="stCameraInput"] {
            width: 100% !important;
        }
        div[data-testid="stCameraInput"] video {
            border-radius: 12px !important;
            width: 100% !important;
            object-fit: cover;
        }

        /* 3. 加大 Tab 标签文字，更容易点 */
        button[data-baseweb="tab"] div {
            font-size: 1.1em !important;
            padding: 1em !important;
        }
        
        /* 4. 优化粘贴按钮样式 */
        div.stButton > button:first-child {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== 1. 核心算法区 ====================

def get_hex_dump_str(raw_bytes):
    """生成易读的 HEX 数据视图"""
    output = []
    output.append(f"📦 数据长度: {len(raw_bytes)} 字节")
    output.append("-" * 50)
    
    try:
        hex_str = raw_bytes.hex().upper()
    except AttributeError:
        # 如果 zxingcpp 返回的是 text (str)，则需要先编码为 bytes
        hex_str = raw_bytes.encode('utf-8').hex().upper()

    for i in range(0, len(hex_str), 32):
        chunk = hex_str[i:i+32]
        ascii_chunk = ""
        for j in range(0, len(chunk), 2):
            try:
                byte_val = int(chunk[j:j+2], 16)
                ascii_chunk += chr(byte_val) if 32 <= byte_val <= 126 else "."
            except ValueError:
                ascii_chunk += "?" # 处理末尾不完整的字节
        output.append(f"{chunk.ljust(32)} | {ascii_chunk}")
    return "\n".join(output)

def preprocess_image_candidates(img):
    """生成图像候选项"""
    candidates = []
    candidates.append(("原图", img))
    
    # 转换为灰度图 (zxingcpp 需要)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # 经典增强算法
    candidates.append(("灰度", gray))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    candidates.append(("CLAHE", enhanced)) # 局部对比度增强
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    candidates.append(("锐化", sharpened)) # 锐化
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("二值(OTSU)", binary)) # 大津二值化
    return candidates

def try_decode(image):
    """尝试解码"""
    try:
        results = zxingcpp.read_barcodes(image)
        for result in results:
            if result.format == zxingcpp.BarcodeFormat.PDF417:
                return True, result
    except Exception:
        pass
    return False, None

def smart_scan_logic(original_img):
    """智能扫描主逻辑 (HAX 增强版)"""
    base_candidates = preprocess_image_candidates(original_img)
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(base_candidates) * 4
    step = 0
    found_result = None

    for mode_name, img_candidate in base_candidates:
        # 常见条码方向和密度问题
        transforms = [
            ("正常", lambda x: x),
            ("旋转90°", lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)),
            ("放大1.5x", lambda x: cv2.resize(x, None, fx=1.5, fy=1.5)),
            # 缩小对 PDF417 效果不好，但保留一个快速尝试
            # ("缩小0.5x", lambda x: cv2.resize(x, (x.shape[1]//2, x.shape[0]//2))) 
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

# --- PDF417 参数逆向计算 ---

def calculate_pdf417_params(byte_len):
    """
    根据字节长度，计算所有可能的 PDF417 行列组合，并估算宽高比。
    """
    if byte_len <= 0:
        return pd.DataFrame()

    # AAMVA 标准估算逻辑 (北美驾照/ID标准)
    estimated_data_cw = math.ceil(byte_len / 1.8) 
    ecc_cw = 64  # Level 5 Security (AAMVA Standard)
    total_cw = estimated_data_cw + ecc_cw
    
    data = []
    possible_cols = range(9, 21) # 常用列数范围
    
    for cols in possible_cols:
        rows = math.ceil(total_cw / cols)
        
        if rows < 3 or rows > 90: # 规范限制
            continue
            
        # 宽高比估算 (W/H)，假设行高/模块宽度 = 3 (常见于ID卡)
        width_units = (cols + 4) * 17
        height_units = rows * 3 
        ratio = width_units / height_units

        # 备注逻辑
        note = ""
        if cols == 17: note = "⭐ AAMVA 标准"
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

# ==================== 2. 网页界面区 ====================

st.title("💳 PDF417 扫码专家")

# 使用 tabs 进行模式切换
tab1, tab2 = st.tabs(["📸 网页小窗 (快速)", "📱 全屏拍照 / 粘贴"])

target_image = None
raw_data = None
data_source = None

# --- Tab 1: 网页相机 ---
with tab1:
    st.caption("适用于光线好、条码清晰的简单场景。请横屏使用。")
    camera_file = st.camera_input("请对准条码", label_visibility="collapsed")
    if camera_file:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)
        data_source = "网页相机"

# --- Tab 2: 全屏拍照 / 粘贴 (核心修改点) ---
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

    # 布局：将粘贴按钮和上传组件分开，避免拥挤
    col_paste, col_upload = st.columns([1, 2])
    
    with col_paste:
        # === 新增功能：粘贴剪贴板图片 ===
        paste_result = paste_image_button(
            label="📋 粘贴剪贴板图片",
            background_color="#FF4B4B",
            hover_background_color="#FF0000",
            text_color="#FFFFFF",
            errors="ignore"
        )
    
    with col_upload:
        upload_file = st.file_uploader("启动全屏相机", type=["jpg", "png", "jpeg", "heic"], label_visibility="collapsed")
    
    # 逻辑判断优先级：如果点击了粘贴，优先使用粘贴的图片
    if paste_result.image_data is not None:
        with st.spinner("正在处理剪贴板图片..."):
            # paste_result.image_data 是 PIL Image 对象
            pil_image = paste_result.image_data
            # 将 PIL 转换为 OpenCV 格式 (RGB -> BGR)
            target_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            data_source = "剪贴板粘贴"
            
    elif upload_file:
        with st.spinner("正在上传高清原图并解码..."):
            file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
            target_image = cv2.imdecode(file_bytes, 1)
            data_source = "文件上传"

# --- 处理结果展示 ---
if target_image is not None:
    st.divider()
    
    # 显示一下当前使用的图片（可选，方便用户确认）
    with st.expander(f"查看当前处理图片 ({data_source})", expanded=False):
        # 转换 BGR -> RGB 以便 st.image 正确显示
        st_display_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        st.image(st_display_img, use_column_width=True)

    result = smart_scan_logic(target_image)
    
    if result:
        st.success("🎉 解码成功！")
        raw_data = result.bytes if result.bytes else result.text.encode('latin-1', errors='ignore')
        
        # 确定数据类型
        data_type = "二进制 (Bytes)" if isinstance(result.bytes, bytes) else "文本 (Text)"
        
        # 1. 结果概览
        st.info(f"数据类型: **{data_type}** | 字节长度: **{len(raw_data)}** bytes")
        
        # 2. 文本内容（如果存在）
        if result.text and data_type == "文本 (Text)":
            st.subheader("📝 文本内容")
            st.code(result.text, language="text")
        elif data_type == "二进制 (Bytes)":
            st.subheader("📝 尝试解码文本 (Latin-1)")
            try:
                 st.code(result.bytes.decode('latin-1'), language="text")
            except Exception:
                 st.code("无法以 Latin-1 解码", language="text")

        # 3. HEX 数据
        with st.expander("查看底层 HEX 数据 (点击展开)", expanded=False):
            hex_str = get_hex_dump_str(raw_data)
            st.code(hex_str, language="text")

        # 4. 参数逆向计算器
        st.subheader("📐 PDF417 参数逆向计算 (AAMVA)")
        byte_len = len(raw_data)
        df_params = calculate_pdf417_params(byte_len)
        
        col_summary, col_table_content = st.columns([1, 2]) # 更改列名

        with col_summary:
            st.markdown(f"**分析长度:** `{byte_len} bytes`")
            st.markdown(f"**ECC 安全等级:** `Level 5 (64 Codewords)`")
            
            best_row = df_params[df_params['列数 (Cols)'] == 17]
            if not best_row.empty:
                rec_rows = best_row.iloc[0]['推算行数 (Rows)']
                st.success(f"💡 AAMVA 推荐: **Cols=17, Rows={rec_rows}**")

        with col_table_content:
            # 创建 Row，放置表格标题和下载按钮
            col_header, col_button = st.columns([4, 1])
            
            with col_header:
                st.markdown("##### 推算行列组合结果 (数据表)")

            with col_button:
                # 使用 st.download_button 模拟复制功能
                csv_data = df_params.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 导出 CSV",
                    data=csv_data,
                    file_name='pdf417_params.csv',
                    mime='text/csv',
                    help="点击下载表格数据为 CSV 文件，方便复制到其他地方。"
                )
            
            # 显示 DataFrame
            st.dataframe(
                df_params,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "估算宽高比 (W/H)": st.column_config.TextColumn("W/H 比例"),
                    "类型备注": st.column_config.TextColumn("备注"),
                }
            )

        # 5. 重开按钮
        st.divider()
        if st.button("🔄 扫描下一张", type="primary"):
            st.rerun()
