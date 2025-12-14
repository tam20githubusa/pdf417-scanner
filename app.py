import streamlit as st
import cv2
import zxingcpp
import numpy as np
from PIL import Image
import time

# ==================== 1. 核心算法区 (你的增强逻辑) ====================

def get_hex_dump_str(raw_bytes):
    """生成漂亮的 HEX 视图字符串"""
    output = []
    output.append(f"数据长度: {len(raw_bytes)} 字节")
    output.append("-" * 40)
    
    hex_str = raw_bytes.hex().upper()
    for i in range(0, len(hex_str), 32):
        chunk = hex_str[i:i+32]
        ascii_chunk = ""
        for j in range(0, len(chunk), 2):
            byte_val = int(chunk[j:j+2], 16)
            ascii_chunk += chr(byte_val) if 32 <= byte_val <= 126 else "."
        output.append(f"{chunk.ljust(32)} | {ascii_chunk}")
    return "\n".join(output)

def preprocess_image_candidates(img):
    """生成图像候选项：原图、灰度、增强、锐化、二值化"""
    candidates = []
    candidates.append(("原图", img))
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    candidates.append(("灰度", gray))

    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    candidates.append(("对比度增强", enhanced))

    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    candidates.append(("锐化", sharpened))

    # 二值化
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("二值化", binary))
    
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
    """智能扫描主逻辑：包含旋转和缩放尝试"""
    base_candidates = preprocess_image_candidates(original_img)
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(base_candidates) * 4
    step = 0

    for mode_name, img_candidate in base_candidates:
        # 变换策略：正常 -> 旋转90 -> 放大 -> 缩小
        transforms = [
            ("正常", lambda x: x),
            ("旋转90°", lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)),
            ("放大1.5x", lambda x: cv2.resize(x, None, fx=1.5, fy=1.5)),
            ("缩小0.5x", lambda x: cv2.resize(x, (x.shape[1]//2, x.shape[0]//2)))
        ]

        for trans_name, trans_func in transforms:
            step += 1
            progress_bar.progress(min(step / total_steps, 0.95))
            status_text.text(f"正在分析: {mode_name} + {trans_name}...")
            
            try:
                processed_img = trans_func(img_candidate)
                success, result = try_decode(processed_img)
                
                if success:
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ 成功! (模式: {mode_name} - {trans_name})")
                    return result
            except:
                continue
                
    status_text.error("❌ 未识别。请尝试靠近拍摄，确保光线充足且无反光。")
    return None

# ==================== 2. 网页界面区 (手机端优化) ====================

st.set_page_config(page_title="PDF417 解析器", layout="centered")

st.title("💳 PDF417 强力解码")
st.info("后端使用 OpenCV + ZXingCpp 引擎，支持模糊/低光环境增强。")

# 选项卡：提供两种方式
tab1, tab2 = st.tabs(["📸 直接拍照", "📂 上传原图"])

target_image = None

with tab1:
    st.write("点击下方按钮直接调用相机：")
    camera_file = st.camera_input("拍照区域", label_visibility="collapsed")
    if camera_file:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)

with tab2:
    st.write("如果直接拍照无法识别，请使用系统相机拍一张高清图上传：")
    upload_file = st.file_uploader("选择图片", type=["jpg", "png", "jpeg"])
    if upload_file:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)

# 开始处理
if target_image is not None:
    st.divider()
    result = smart_scan_logic(target_image)
    
    if result:
        # 成功后的展示区
        st.success("解码成功！")
        
        with st.expander("查看原始 HEX 数据", expanded=True):
            hex_str = get_hex_dump_str(result.bytes)
            st.code(hex_str, language="text")
            
        if result.text:
            st.subheader("解析文本")
            st.text_area("内容", result.text, height=150)
