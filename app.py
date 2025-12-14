import streamlit as st
import cv2
import zxingcpp
import numpy as np
from PIL import Image

# ==================== 0. 页面配置与 CSS 样式优化 ====================

st.set_page_config(page_title="PDF417 扫码专家", page_icon="💳", layout="wide")

# 注入 CSS：强制去除边距，放大相机
st.markdown("""
    <style>
        /* 1. 极大幅度减少页面四周的留白 (手机端关键) */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        
        /* 2. 强制相机组件占满 100% 宽度 */
        div[data-testid="stCameraInput"] {
            width: 100% !important;
        }

        /* 3. 调整视频流的显示样式 */
        video {
            border-radius: 12px !important; /* 圆角看起来更像原生 App */
            width: 100% !important;
            object-fit: cover; /* 充满容器 */
        }
        
        /* 4. 优化按钮样式，让手机上更容易点 */
        button {
            height: 3em !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== 1. 核心算法区 (图像增强与解码) ====================

def get_hex_dump_str(raw_bytes):
    """生成易读的 HEX 数据视图"""
    output = []
    output.append(f"📦 数据长度: {len(raw_bytes)} 字节")
    output.append("-" * 35)
    
    hex_str = raw_bytes.hex().upper()
    for i in range(0, len(hex_str), 32):
        chunk = hex_str[i:i+32]
        ascii_chunk = ""
        for j in range(0, len(chunk), 2):
            byte_val = int(chunk[j:j+2], 16)
            ascii_chunk += chr(byte_val) if 32 <= byte_val <= 126 else "."
        # 手机端简化显示，避免换行混乱
        output.append(f"{chunk[:16]}... | {ascii_chunk}")
    return "\n".join(output)

def preprocess_image_candidates(img):
    """生成图像候选项：原图、灰度、增强、锐化、二值化"""
    candidates = []
    candidates.append(("原图", img))
    
    # 确保转为灰度
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    candidates.append(("灰度", gray))

    # CLAHE 对比度增强 (应对光线不足)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    candidates.append(("增强", enhanced))

    # 锐化 (应对模糊)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    candidates.append(("锐化", sharpened))

    # 二值化 (应对低对比度)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("二值", binary))
    
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
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(base_candidates) * 4
    step = 0

    found_result = None

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
            status_text.caption(f"正在分析: {mode_name} / {trans_name}...")
            
            try:
                processed_img = trans_func(img_candidate)
                success, result = try_decode(processed_img)
                
                if success:
                    found_result = result
                    status_text.success(f"✅ 识别成功! (模式: {mode_name} - {trans_name})")
                    progress_bar.progress(1.0)
                    break
            except:
                continue
        
        if found_result:
            break
            
    if not found_result:
        status_text.error("❌ 未识别。请靠近一点，或尝试'上传原图'模式。")
        progress_bar.empty()
        
    return found_result

# ==================== 2. 网页界面区 ====================

st.title("💳 PDF417 扫码专家")

# 选项卡
tab1, tab2 = st.tabs(["📸 直接拍照 (Web)", "📂 上传原图 (高清)"])

target_image = None

# --- Tab 1: 网页相机 ---
with tab1:
    st.info("💡 提示：请将手机**横屏**以获得最大视野。")
    # key=None 强制刷新，help 提示
    camera_file = st.camera_input("请对准条码", label_visibility="hidden")
    if camera_file:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)

# --- Tab 2: 文件上传 ---
with tab2:
    st.write("如果直接拍照看不清，请点下面按钮选**“拍照”**：")
    upload_file = st.file_uploader("选择图片/拍照", type=["jpg", "png", "jpeg", "heic"])
    if upload_file:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, 1)

# --- 处理结果展示 ---
if target_image is not None:
    st.divider()
    
    # 执行智能扫描
    result = smart_scan_logic(target_image)
    
    if result:
        st.success("🎉 解码成功！")
        
        # 1. 文本内容
        if result.text:
            st.subheader("📝 文本内容")
            st.code(result.text, language="text")
        
        # 2. 原始 HEX 数据
        with st.expander("查看底层 HEX 数据 (点击展开)", expanded=False):
            hex_str = get_hex_dump_str(result.bytes)
            st.code(hex_str, language="text")
            
        # 3. 重新开始按钮
        if st.button("🔄 扫描下一张"):
            st.rerun()
