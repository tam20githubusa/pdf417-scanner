# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
import io 
import math
import pandas as pd
import base64
import os
import subprocess

# --- 引入外部库 ---
try:
    from pdf417 import encode, render_image
except ImportError:
    st.warning("警告：PDF417 编码库 (pdf417) 未安装。条码图像功能将使用占位符。请运行 `pip install pdf417`。")
    def encode(*args, **kwargs): return []
    def render_image(*args, **kwargs): return Image.new('RGB', (400, 100), color='white')


# ==================== 0. 配置与 51 州 IIN 映射 (最终版) ====================

# 州代码到 IIN 和版本信息的映射 (AAMVA V09/D20-2020 兼容)
JURISDICTION_MAP = {
    # 东北地区 (Northeast)
    "ME": {"name": "Maine - 缅因州", "iin": "636021", "jver": "01", "race": "W"},
    "VT": {"name": "Vermont - 佛蒙特州", "iin": "636044", "jver": "01", "race": "W"},
    "NH": {"name": "New Hampshire - 新罕布什尔州", "iin": "636029", "jver": "01", "race": "W"},
    "MA": {"name": "Massachusetts - 马萨诸塞州", "iin": "636022", "jver": "01", "race": "W"},
    "RI": {"name": "Rhode Island - 罗德岛州", "iin": "636039", "jver": "01", "race": "W"},
    "CT": {"name": "Connecticut - 康涅狄格州", "iin": "636003", "jver": "01", "race": "W"},
    "NY": {"name": "New York - 纽约州", "iin": "636034", "jver": "01", "race": "W"},
    "NJ": {"name": "New Jersey - 新泽西州", "iin": "636030", "jver": "01", "race": "W"},
    "PA": {"name": "Pennsylvania - 宾夕法尼亚州", "iin": "636038", "jver": "01", "race": "W"},
    # 中西部地区 (Midwest)
    "OH": {"name": "Ohio - 俄亥俄州", "iin": "636035", "jver": "01", "race": "W"},
    "IN": {"name": "Indiana - 印第安纳州", "iin": "636014", "jver": "01", "race": "W"},
    "IL": {"name": "Illinois - 伊利诺伊州", "iin": "636013", "jver": "01", "race": "W"},
    "MI": {"name": "Michigan - 密歇根州", "iin": "636023", "jver": "01", "race": "W"},
    "WI": {"name": "Wisconsin - 威斯康星州", "iin": "636047", "jver": "01", "race": "W"},
    "MN": {"name": "Minnesota - 明尼苏达州", "iin": "636024", "jver": "01", "race": "W"},
    "IA": {"name": "Iowa - 爱荷华州", "iin": "636015", "jver": "01", "race": "W"},
    "MO": {"name": "Missouri - 密苏里州", "iin": "636025", "jver": "01", "race": "W"},
    "ND": {"name": "North Dakota - 北达科他州", "iin": "636033", "jver": "01", "race": "W"},
    "SD": {"name": "South Dakota - 南达科他州", "iin": "636042", "jver": "01", "race": "W"},
    "NE": {"name": "Nebraska - 内布拉斯加州", "iin": "636028", "jver": "01", "race": "W"},
    "KS": {"name": "Kansas - 堪萨斯州", "iin": "636016", "jver": "01", "race": "W"},
    # 南部地区 (South)
    "DE": {"name": "Delaware - 特拉华州", "iin": "636004", "jver": "01", "race": "W"},
    "MD": {"name": "Maryland - 马里兰州", "iin": "636020", "jver": "01", "race": "W"},
    "VA": {"name": "Virginia - 弗吉尼亚州", "iin": "636046", "jver": "01", "race": "W"},
    "WV": {"name": "West Virginia - 西弗吉尼亚州", "iin": "636048", "jver": "01", "race": "W"},
    "NC": {"name": "North Carolina - 北卡罗来纳州", "iin": "636032", "jver": "01", "race": "W"},
    "SC": {"name": "South Carolina - 南卡罗来纳州", "iin": "636041", "jver": "01", "race": "W"},
    "GA": {"name": "Georgia - 佐治亚州", "iin": "636008", "jver": "01", "race": "W"},
    "FL": {"name": "Florida - 佛罗里达州", "iin": "636005", "jver": "01", "race": "W"},
    "KY": {"name": "Kentucky - 肯塔基州", "iin": "636017", "jver": "01", "race": "W"},
    "TN": {"name": "Tennessee - 田纳西州", "iin": "636040", "jver": "01", "race": "W"},
    "AL": {"name": "Alabama - 阿拉巴马州", "iin": "636001", "jver": "01", "race": "W"},
    "MS": {"name": "Mississippi - 密西西比州", "iin": "636026", "jver": "01", "race": "W"},
    "AR": {"name": "Arkansas - 阿肯色州", "iin": "636002", "jver": "01", "race": "W"},
    "LA": {"name": "Louisiana - 路易斯安那州", "iin": "636019", "jver": "01", "race": "W"},
    "OK": {"name": "Oklahoma - 俄克拉荷马州", "iin": "636036", "jver": "01", "race": "W"},
    "TX": {"name": "Texas - 德克萨斯州", "iin": "636043", "jver": "01", "race": "W"},
    # 西部地区 (West)
    "MT": {"name": "Montana - 蒙大拿州", "iin": "636027", "jver": "01", "race": "W"},
    "ID": {"name": "Idaho - 爱达荷州", "iin": "636012", "jver": "01", "race": "W"},
    "WY": {"name": "Wyoming - 怀俄明州", "iin": "636049", "jver": "01", "race": "W"},
    "CO": {"name": "Colorado - 科罗拉多州", "iin": "636020", "jver": "01", "race": "CLW"}, # 特殊的 DCL 码
    "UT": {"name": "Utah - 犹他州", "iin": "636045", "jver": "01", "race": "W"},
    "AZ": {"name": "Arizona - 亚利桑那州", "iin": "636006", "jver": "01", "race": "W"},
    "NM": {"name": "New Mexico - 新墨西哥州", "iin": "636031", "jver": "01", "race": "W"},
    "AK": {"name": "Alaska - 阿拉斯加州", "iin": "636000", "jver": "00", "race": "W"},
    "WA": {"name": "Washington - 华盛顿州", "iin": "636045", "jver": "00", "race": "W"},
    "OR": {"name": "Oregon - 俄勒冈州", "iin": "636037", "jver": "01", "race": "W"},
    "CA": {"name": "California - 加利福尼亚州", "iin": "636000", "jver": "00", "race": "W"},
    "NV": {"name": "Nevada - 内华达州", "iin": "636032", "jver": "01", "race": "W"},
    "HI": {"name": "Hawaii - 夏威夷州", "iin": "636009", "jver": "01", "race": "W"},
    # 地区 (Territories/DC)
    "DC": {"name": "District of Columbia - 华盛顿特区", "iin": "636007", "jver": "01", "race": "W"},
}

st.set_page_config(page_title="AAMVA PDF417 生成专家", page_icon="💳", layout="wide")

# 注入 CSS：优化布局
st.markdown("""
    <style>
        .block-container { padding: 1rem 1rem; }
        [data-testid="stTextInput"] { width: 100%; }
        .stButton>button { width: 100%; }
        .stSelectbox { width: 100%; }
    </style>
""", unsafe_allow_html=True)


# ==================== 1. 核心辅助函数 ====================

def get_hex_dump_str(raw_bytes):
    """生成易读的 HEX 数据视图"""
    output = []
    output.append(f"📦 数据长度: {len(raw_bytes)} 字节")
    output.append("-" * 50)
    
    if isinstance(raw_bytes, str):
        raw_bytes = raw_bytes.encode('latin-1', errors='ignore')

    hex_str = raw_bytes.hex().upper()

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

def clean_date_input(date_str):
    """清理日期输入，移除分隔符"""
    return date_str.replace("/", "").replace("-", "").strip().upper()

def convert_height_to_inches_ui(height_str):
    """将身高 (如 510) 转换为 AAMVA 要求的 3 位总英寸 (如 070)"""
    height_str = height_str.strip()
    if not height_str or not height_str.isdigit(): return "000"
    
    if len(height_str) < 3: 
        total_inches = int(height_str)
    else:
        try:
            inches_part = int(height_str[-2:])
            feet_part = int(height_str[:-2])
            total_inches = (feet_part * 12) + inches_part
        except ValueError:
             return f"{int(height_str):03d}"
             
    return f"{total_inches:03d}"


# ==================== 2. AAMVA 生成核心逻辑 ====================

def generate_aamva_data_core(inputs):
    """根据 Streamlit 输入字典，生成 AAMVA PDF417 原始数据流 (CO 格式模板)"""
    
    # 1. 获取州配置
    jurisdiction_code = inputs['jurisdiction_code']
    config = JURISDICTION_MAP.get(jurisdiction_code)
    
    iin = config['iin']
    jurisdiction_version = config['jver']
    
    # 2. 清洗输入数据 (转换为大写，清理空格)
    first_name = inputs['first_name'].strip().upper()
    middle_name = inputs['middle_name'].strip().upper() if inputs['middle_name'] else "NONE"
    last_name = inputs['last_name'].strip().upper()
    address = inputs['address'].strip().upper()
    city = inputs['city'].strip().upper()
    
    # 邮编处理
    zip_code = inputs['zip_input'].replace("-", "").strip().upper()
    if len(zip_code) == 5: zip_code += "0000"
    
    # 日期处理
    dob = clean_date_input(inputs['dob'])
    exp_date = clean_date_input(inputs['exp_date'])
    iss_date = clean_date_input(inputs['iss_date'])
    rev_date = clean_date_input(inputs['rev_date'])

    # 证件详情
    dl_number = inputs['dl_number'].strip().upper()
    class_code = inputs['class_code'].strip().upper()
    rest_code = inputs['rest_code'].strip().upper() if inputs['rest_code'] else "NONE"
    end_code = inputs['end_code'].strip().upper() if inputs['end_code'] else "NONE"
    dd_code = inputs['dd_code'].strip().upper()
    audit_code = inputs['audit_code'].strip().upper()
    dda_code = inputs['dda_code'].strip().upper()
    
    # 物理特征
    sex = inputs['sex'].strip()
    height = convert_height_to_inches_ui(inputs['height_input'])
    weight = inputs['weight'].strip().upper()
    eyes = inputs['eyes'].strip().upper()
    hair = inputs['hair'].strip().upper()
    race = inputs['race'].strip().upper() if inputs['race'] else config['race']
    
    # --- 3. 构造子文件 DL (AAMVA V09 核心结构) ---
    aamva_version = "09"
    num_entries = "02" # 使用 CO 验证的 2 个子文件 (DL+ZC) 结构

    # 字段顺序和格式参照 CO/TX 经验证的结构进行泛化
    subfile_dl_body = (
        f"DL"                                    
        f"DAQ{dl_number}\x0a"                      
        f"DCS{last_name}\x0a"                      
        f"DDEN{first_name}\x0a"                    
        f"DAC{middle_name}\x0a"                    
        f"DDFN\x0a"                                
        f"DAD\x0a"                                 
        f"DDGN\x0a"                                
        f"DCA{class_code}\x0a"                     
        f"DCB{rest_code}\x0a"                      
        f"DCD{end_code}\x0a"                       
        f"DBD{iss_date}\x0a"                       
        f"DBB{dob}\x0a"
        f"DBA{exp_date}\x0a"
        f"DBC{sex}\
