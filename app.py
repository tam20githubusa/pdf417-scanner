# -*- coding: utf-8 -*-
import cv2
import zxingcpp
import math
import numpy as np
import streamlit as st
import pandas as pd
import threading
from io import BytesIO

# 引入 WebRTC 组件
from streamlit_webrtc import webrtc_stream, VideoTransformerBase, WebRtcMode
import av 

# --- 1. 核心计算函数 (参数逆向计算) ---

def calculate_pdf417_params(byte_len):
    """
    根据字节长度，计算所有可能的 PDF417 行列组合，并估算宽高比。
    - AAMVA 标准: 1.8 bytes ≈ 1 data codeword
    - ECC: Level 5 (64 Codewords)
    """
    if byte_len <= 0:
        return pd.DataFrame()

    estimated_data_cw = math.ceil(byte_len / 1.8) 
    ecc_cw = 64
    total_cw = estimated_data_cw + ecc_cw
    
    data = []
    possible_cols = range(9, 21)
    
    for cols in possible_cols:
        rows = math.ceil(total_cw / cols)
        
        if rows < 3 or rows > 90:
            continue
            
        # 宽高比估算 (W/H)，假设行高/模块宽度 = 3
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

# --- 2. 文件上传模式下的图像识别 (HAX 增强逻辑) ---

def preprocess_image(img):
    """多级图像预处理，用于增强识别率"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    yield gray, "Level 1: 灰度原图"
    
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    yield scaled, "Level 2: 2X 放大灰度"
    
    _, binary = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)
    yield binary, "Level 3: 2X 放大 + 普通二值化"
    
    adaptive = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    yield adaptive, "Level 4: 2X 放大 + 自适应二值化"

def get_barcode_data(img_bytes):
    """尝试读取条码，返回数据字节和状态信息"""
    nparr = np.frombuffer(img_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, "❌ 错误: 图片无法加载"

    # 1. 尝试直接读取原图
    results = zxingcpp.read_barcodes(img)
    for res in results:
        if res.format == zxingcpp.BarcodeFormat.PDF417:
            return res.bytes, "✅ 原图模式下成功识别"

    # 2. 尝试增强模式
    for i, (processed_img, desc) in enumerate(preprocess_image(img)):
        results = zxingcpp.read_barcodes(processed_img)
        for res in results:
            if res.format == zxingcpp.BarcodeFormat.PDF417:
                return res.bytes, f"✅ 增强模式 ({desc}) 下成功识别"
                
    return None, "❌ 最终失败: 所有增强算法均无法读取该 PDF417。"

# --- 3. 实时摄像头视频处理器 (WebRTC) ---

class BarcodeScanner(VideoTransformerBase):
    def __init__(self, callback):
        self.callback = callback
        self.scanned_data = None
        self.lock = threading.Lock()
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        results = zxingcpp.read_barcodes(gray)
        
        for res in results:
            if res.format == zxingcpp.BarcodeFormat.PDF417:
                data_bytes = res.bytes
                position = res.position
                
                # 如果是新的数据，则回调给主应用
                if self.scanned_data is None or data_bytes != self.scanned_data:
                    self.scanned_data = data_bytes
                    self.callback(data_bytes) # 触发 Streamlit 状态更新
                
                # 绘制定位框
                points = [(p.x, p.y) for p in position.points]
                rect = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [rect], True, (0, 255, 0), 3)
                
                # 绘制文本
                text = f"PDF417: {len(data_bytes)} Bytes"
                cv2.putText(img, text, (points[0][0], points[0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                break
        
        return img

# --- 回调函数：将扫描结果存入 Session State ---
def barcode_scanned_callback(data_bytes):
    """当条码被成功扫描时调用此函数，更新 Session State"""
    data_len = len(data_bytes)
    # 使用 lock 确保线程安全
    with st.session_state['lock']:
        st.session_state['last_scan_bytes'] = data_len
        st.session_state['scanned_result'] = f"✅ 实时扫描成功! 数据长度: {data_len}"
        st.session_state['rerun_flag'] = True # 设置标记，通知主应用需要刷新

# --- 4. Streamlit UI 界面 ---

st.set_page_config(layout="wide", page_title="PDF417 条码分析工具")

# 初始化 session_state
if 'last_scan_bytes' not in st.session_state:
    st.session_state['last_scan_bytes'] = 0
if 'scanned_result' not in st.session_state:
    st.session_state['scanned_result'] = "请选择一种扫描模式开始"
if 'lock' not in st.session_state:
    st.session_state['lock'] = threading.Lock()
if 'rerun_flag' not in st.session_state:
    st.session_state['rerun_flag'] = False

st.title("🆔 PDF417 条码分析与逆向工具")
st.markdown("---")


# --- 模式选择与扫描 ---

scan_mode = st.radio(
    "选择扫描模式", 
    ('实时相机扫描 (WebRTC)', '上传图片文件'),
    horizontal=True
)

data_extracted = False

if scan_mode == '实时相机扫描 (WebRTC)':
    st.subheader("📸 实时相机扫描模式")
    
    # 1. 设置 WebRTC 流
    ctx = webrtc_stream(
        key="pdf417-scanner",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: BarcodeScanner(barcode_scanned_callback), 
        async_transform=True,
    )
    
    # 2. 显示扫描状态和结果
    st.info(st.session_state['scanned_result'])
    
    # 如果扫描成功，且需要刷新，则强制刷新 UI
    if st.session_state['rerun_flag']:
        st.session_state['rerun_flag'] = False
        st.experimental_rerun()


elif scan_mode == '上传图片文件':
    st.subheader("⬆️ 图片文件上传模式 (HAX 增强识别)")
    
    uploaded_file = st.file_uploader(
        "上传 PDF417 条码图片 (如身份证背面扫描件)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = BytesIO(uploaded_file.getvalue())
        
        col_img, col_status = st.columns([1, 1])
        with col_img:
            st.image(file_bytes.getvalue(), caption='上传的图片', use_column_width=True)
        
        with col_status:
            with st.spinner("正在尝试多级图像增强识别..."):
                file_bytes.seek(0)
                raw_bytes, status_msg = get_barcode_data(file_bytes)
                
            st.info(status_msg)
            
            if raw_bytes:
                data_len = len(raw_bytes)
                st.success(f"🎉 成功提取数据! 原始字节长度: {data_len} bytes")
                # 保存到 session_state 供计算器使用
                st.session_state['last_scan_bytes'] = data_len
                st.session_state['scanned_result'] = f"✅ 文件扫描成功! 数据长度: {data_len}"
                data_extracted = True
            else:
                st.error("无法识别 PDF417。请尝试使用实时相机或更清晰的图片。")


# --- 参数逆向计算器区域 ---
st.divider()
st.subheader("📐 2. 参数逆向计算器")
st.caption("基于 Level 5 (64 CW ECC) 和 AAMVA 1.8 bytes/CW 估算。")

# 自动填入或手动输入
default_len = st.session_state['last_scan_bytes']

with st.expander("展开计算器并输入数据长度", expanded=(default_len > 0)):
    col_input, col_info = st.columns([1, 2])
    
    with col_input:
        byte_input = st.number_input(
            "原始数据字节长度 (Raw Data Length)", 
            min_value=0, 
            value=default_len,
            step=1,
            key="byte_input_calc",
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
                st.warning("数据量较大或过小，未找到标准 17 列方案。")

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
             st.info("请在上方进行扫描或手动输入长度来计算最佳行列参数。")
