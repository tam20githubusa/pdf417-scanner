# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import zxingcpp
import numpy as np
import pandas as pd
import math
from PIL import Image

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

# --- 新增：PDF417 参数逆向计算 ---

def calculate_pdf417_params(byte_len):
    """
    根据字节长度，计算所有可能的 PDF417 行列组合，并估算宽高比。
    """
    if byte_len <= 0:
        return pd.DataFrame()

    estimated_data_cw = math.ceil(byte_len / 1.8) 
    ecc_cw = 64  # Level 5 Security (AAMVA Standard)
    total_cw = estimated_data_cw + ecc_cw
    
    data = []
    possible_cols = range(9, 21)
    
    for cols in possible_cols:
        rows = math.ceil(total_cw / cols)
        
        if rows < 3 or rows > 90:
            continue
            
        width_units = (cols + 4) * 17
        height_units = rows * 3 
        ratio = width_units / height_units

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

# --- 新增：AAMVA 数据解析函数 ---

def parse_aamva_data(raw_bytes):
    """
    解析 AAMVA D20 标准的原始字节数据，提取关键字段。
    """
    try:
        # AAMVA 使用 ASCII 或 Latin-1 编码
        data_str = raw_bytes.decode('latin-1', errors='ignore') 
    except Exception:
        return {"Error": "无法将数据解码为 ASCII/Latin-1 文本。"}

    # 定义字段分隔符 (RS: 1E) 和记录分隔符 (LF: 0A, CR: 0D)
    
    # 查找主数据段 (以 DL, ID, 或 DB 开头)
    segments = data_str.split('\x1e') 
    
    # 字段代码到描述的映射 (只列出关键字段和您提到的字段)
    fields_map = {
        "DCS": "姓氏 (Last Name)",
        "DDEN": "名 (First Name)",
        "DAC": "中间名 (Middle Name)",
        "DDG": "签发日期 (Issue Date)",
        "DBD": "出生日期 (DOB)",
        "DBA": "到期日期 (Expiry Date)",
        "DCD": "驾照/证件号码 (License No.)", # 关键字段
        "DBC": "性别 (Gender Code)",
        "DAU": "地址 (Street)",
        "DAI": "城市 (City)",
        "DAJ": "州/省 (Jurisdiction)",
        "DCF": "国家/地区 (Country)",
        "DCK": "身高/体重 (CK)",
        # ZFZFA - ZFK 是州自定义字段，通常用于冗余数据
        "ZFJ": "自定义号 (ZFJ)"
    }
    
    parsed_data = {}
    
    # 查找主数据段
    main_segment_found = False
    for segment in segments:
        if segment.startswith('DL') or segment.startswith('ID'):
            main_segment_found = True
            data_content = segment[segment.find('Z')+1:] # 从 'Z' 之后开始解析数据
            break
            
    if not main_segment_found:
        return {"Error": "未找到 DL/ID 主数据段。"}
        
    # 解析逻辑：寻找 3 或 4 个大写字母的代码
    current_pos = 0
    while current_pos < len(data_content):
        match_found = False
        
        # 查找下一个字段代码（3或4个大写字母）
        for code in fields_map.keys():
            if data_content.startswith(code, current_pos):
                field_code = code
                field_description = fields_map[field_code]
                
                # 寻找下一个字段代码的起始位置作为当前值的结束
                next_field_pos = len(data_content)
                
                # 查找下一个字段代码的位置 (可以是任何一个已知的代码)
                for next_code in fields_map.keys():
                    pos = data_content.find(next_code, current_pos + len(field_code))
                    if pos != -1 and pos < next_field_pos:
                         next_field_pos = pos
                
                value = data_content[current_pos + len(field_code): next_field_pos]
                
                # 清理值中的分隔符 (\n, \r)
                parsed_data[field_description] = value.replace('\n', '').replace('\r', '').strip()
                current_pos = next_field_pos
                match_found = True
                break
        
        if not match_found:
            current_pos += 1 # 找不到字段时跳过，避免死循环
        
        if current_pos >= len(data_content):
            break
            
    return parsed_data

# ==================== 2. 网页界面区 ====================

st.title("💳 PDF417 扫码专家")

# 使用 tabs 进行模式切换
tab1, tab2 = st.tabs(["📸 网页小窗 (快速)", "📱 全屏拍照 (高清推荐)"])

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

# --- Tab 2: 全屏拍照 (核心修改点) ---
with tab2:
    st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; margin-bottom: 20px;">
            <h4 style="margin: 0; color: #2e7d32; font-size: 1.1rem;">🚀 最佳识别方案：</h4>
            <p style="margin: 10px 0 0 0; font-size: 1rem; color: #333;">
                点击下方按钮，在弹出的菜单中选择 <b>“拍照”</b> 或 <b>“相机”</b>。<br>
                这将启动你的<b>系统原生相机</b>，享受<b>全屏、高清、手动对焦</b>体验！
            </p>
        </div>
    """, unsafe_allow_html=True)

    upload_file = st.file_uploader("启动全屏相机", type=["jpg", "png", "jpeg", "heic"], label_visibility="collapsed")
    
    if upload_file:
        with st.spinner("正在上传高清原图并解码..."):
            file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
            target_image = cv2.imdecode(file_bytes, 1)
            data_source = "文件上传"

# --- 处理结果展示 ---
if target_image is not None:
    st.divider()
    result = smart_scan_logic(target_image)
    
    if result:
        st.success("🎉 解码成功！")
        raw_data = result.bytes if result.bytes else result.text.encode('latin-1', errors='ignore')
        
        # 确定数据类型
        data_type = "二进制 (Bytes)" if isinstance(result.bytes, bytes) else "文本 (Text)"
        
        # 1. 结果概览
        st.info(f"数据类型: **{data_type}** | 字节长度: **{len(raw_data)}** bytes")
        
        # 2. 结构化解析 (新增区域)
        if data_type == "二进制 (Bytes)" and len(raw_data) > 100:
            st.subheader("📋 结构化数据解析 (AAMVA)")
            parsed_data = parse_aamva_data(raw_data)
            
            if "Error" in parsed_data:
                 st.error(f"解析失败: {parsed_data['Error']}")
            else:
                 # 使用 Pandas DataFrame 展示解析结果，更美观
                 df_parsed = pd.DataFrame(parsed_data.items(), columns=["字段", "值"])
                 
                 # 确保许可证号和姓名放在最前面
                 df_parsed = df_parsed.sort_values(by="字段", key=lambda x: x.map({'驾照/证件号码 (License No.)': 0, '姓氏 (Last Name)': 1}), ascending=True, ignore_index=True)
                 
                 st.dataframe(df_parsed, use_container_width=True, hide_index=True)
                 
                 # --- 演示您想要的格式 (DAQ123456 驾照/身份证号 123456) ---
                 license_no = parsed_data.get('驾照/证件号码 (License No.)', 'N/A')
                 st.markdown(f"**快速查看:** **{license_no}** 对应 **驾照/证件号码**")

        # 3. HEX 数据
        with st.expander("查看底层 HEX 数据 (点击展开)", expanded=False):
            hex_str = get_hex_dump_str(raw_data)
            st.code(hex_str, language="text")

        # 4. 参数逆向计算器 (含导出 CSV)
        st.subheader("📐 PDF417 参数逆向计算 (AAMVA)")
        byte_len = len(raw_data)
        df_params = calculate_pdf417_params(byte_len)
        
        col_summary, col_table_content = st.columns([1, 2])

        with col_summary:
            st.markdown(f"**分析长度:** `{byte_len} bytes`")
            st.markdown(f"**ECC 安全等级:** `Level 5 (64 Codewords)`")
            
            best_row = df_params[df_params['列数 (Cols)'] == 17]
            if not best_row.empty:
                rec_rows = best_row.iloc[0]['推算行数 (Rows)']
                st.success(f"💡 AAMVA 推荐: **Cols=17, Rows={rec_rows}**")

        with col_table_content:
            col_header, col_button = st.columns([4, 1])
            
            with col_header:
                st.markdown("##### 推算行列组合结果 (数据表)")

            with col_button:
                csv_data = df_params.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 导出 CSV",
                    data=csv_data,
                    file_name='pdf417_params.csv',
                    mime='text/csv',
                    help="点击下载表格数据为 CSV 文件，方便复制到其他地方。"
                )
            
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
