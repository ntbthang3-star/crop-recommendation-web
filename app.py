"""
Đất & Cây - Hệ thống gợi ý cây trồng theo điều kiện đất và khí hậu.
Mô hình: MyRandomForest tự cài đặt trong thuat_toan.py (không dùng scikit-learn).
Chạy:  streamlit run app.py
"""
import os, io, re, base64, unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from thuat_toan import MyRandomForest

BASE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Đất & Cây", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

FEATS = ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]
SOIL = ["N", "P", "K", "pH"]
LBL = {"N": "N - Đạm (Nitrogen)", "P": "P - Lân (Phosphorus)", "K": "K - Kali (Potassium)",
       "temperature": "Nhiệt độ (°C)", "humidity": "Độ ẩm không khí (%)", "pH": "pH - Độ chua của đất",
       "rainfall": "Lượng mưa (mm)"}
SHORT = {"N": "N", "P": "P", "K": "K", "temperature": "Nhiệt độ", "humidity": "Độ ẩm", "pH": "pH", "rainfall": "Lượng mưa"}
LIMIT = {"N": (0.0, 300.0, 1.0), "P": (0.0, 300.0, 1.0), "K": (0.0, 400.0, 1.0), "temperature": (0.0, 50.0, 0.5),
         "humidity": (0.0, 100.0, 1.0), "pH": (0.0, 14.0, 0.1), "rainfall": (0.0, 600.0, 1.0)}

# key trong dataset: (tên Việt, tên khoa học, nhóm, emoji, thời gian, ghi chú)
CROPS = {
    "rice": ("Lúa", "Oryza sativa", "Ngũ cốc", "90–120 ngày", "Cây lương thực chính, ưa nóng ẩm và nhiều nước."),
    "maize": ("Ngô", "Zea mays", "Ngũ cốc", "80–110 ngày", "Cây ngắn ngày, nhạy cảm với úng và hạn lúc trổ cờ."),
    "chickpea": ("Đậu gà", "Cicer arietinum", "Đậu đỗ", "90–120 ngày", "Chịu hạn tốt, hợp mùa khô mát."),
    "kidneybeans": ("Đậu đỏ", "Phaseolus vulgaris", "Đậu đỗ", "60–120 ngày", "Cần đất tơi xốp, không chịu được úng."),
    "pigeonpeas": ("Đậu triều", "Cajanus cajan", "Đậu đỗ", "120–180 ngày", "Cây chịu hạn, rễ sâu, cải tạo đất tốt."),
    "mothbeans": ("Đậu mỡ", "Vigna aconitifolia", "Đậu đỗ", "60–90 ngày", "Rất chịu nóng và khô, hợp vùng ít mưa."),
    "mungbean": ("Đậu xanh", "Vigna radiata", "Đậu đỗ", "60–75 ngày", "Cây ngắn ngày, hợp xen canh sau lúa."),
    "blackgram": ("Đậu đen", "Vigna mungo", "Đậu đỗ", "70–90 ngày", "Ưa ấm, ẩm vừa, cố định đạm cho đất."),
    "lentil": ("Đậu lăng", "Lens culinaris", "Đậu đỗ", "80–120 ngày", "Ưa khí hậu mát, khô ráo."),
    "pomegranate": ("Lựu", "Punica granatum", "Cây ăn quả", "2–3 năm cho quả đầu", "Chịu hạn khá, cần nắng nhiều để quả chín đẹp."),
    "banana": ("Chuối", "Musa spp.", "Cây ăn quả", "9–12 tháng thu hoạch", "Cần ẩm cao, nhiều nước và nhiều kali."),
    "mango": ("Xoài", "Mangifera indica", "Cây ăn quả", "3–5 năm cho quả đầu", "Ưa nóng, cần mùa khô ngắn để ra hoa tập trung."),
    "grapes": ("Nho", "Vitis vinifera", "Cây ăn quả", "2–3 năm cho quả", "Cần nắng, thoát nước tốt và giàn leo."),
    "apple": ("Táo", "Malus domestica", "Cây ăn quả", "2–5 năm cho quả", "Cần khí hậu mát, có giai đoạn lạnh để ra hoa."),
    "orange": ("Cam", "Citrus sinensis", "Cây ăn quả", "2–3 năm cho quả đầu", "Ưa ấm, đất thoát nước, tưới đều."),
    "papaya": ("Đu đủ", "Carica papaya", "Cây ăn quả", "8–10 tháng", "Lớn nhanh, sợ úng, cần ấm quanh năm."),
    "coconut": ("Dừa", "Cocos nucifera", "Cây ăn quả", "5–7 năm cho quả đầu", "Hợp vùng ven biển, ẩm cao, mưa nhiều."),
    "watermelon": ("Dưa hấu", "Citrullus lanatus", "Dưa", "60–90 ngày", "Cần nắng, đất cát pha thoát nước, ít mưa lúc chín."),
    "muskmelon": ("Dưa lê", "Cucumis melo", "Dưa", "70–90 ngày", "Ưa nắng ấm, tránh ẩm thấp gây bệnh lá."),
    "cotton": ("Bông", "Gossypium spp.", "Cây công nghiệp", "150–180 ngày", "Cần nắng nhiều, mùa thu hoạch cần khô ráo."),
    "jute": ("Đay", "Corchorus spp.", "Cây công nghiệp", "100–120 ngày", "Ưa nóng ẩm, mưa nhiều, đất phù sa."),
    "coffee": ("Cà phê", "Coffea spp.", "Cây công nghiệp", "3–4 năm cho quả đầu", "Cần mưa đủ, mùa khô ngắn để phân hóa mầm hoa."),
}
GROUPS = ["Ngũ cốc", "Đậu đỗ", "Cây ăn quả", "Dưa", "Cây công nghiệp"]
PERENNIAL = {k for k, v in CROPS.items() if v[2] == "Cây ăn quả"} | {"coffee"}
GROUP_SEASON = {"Ngũ cốc": "Trồng theo vụ; chọn vụ có đủ nắng ấm và nguồn nước.",
                "Đậu đỗ": "Thường trồng xen hoặc nối vụ sau lúa; tránh mùa mưa lớn lúc thu hoạch.",
                "Cây ăn quả": "Cây lâu năm; trồng đầu mùa mưa để cây ít phải tưới, thu hoạch theo mùa quả.",
                "Dưa": "Trồng vụ khô nắng, đủ ánh sáng để quả ngọt; tránh mưa nhiều lúc quả chín.",
                "Cây công nghiệp": "Chọn thời điểm đầu mùa mưa (cây lâu năm) hoặc đầu vụ nắng ấm (cây hằng năm)."}

STEPS_ANNUAL = [("Làm đất & bón lót", "Cày bừa tơi xốp, làm sạch cỏ, bón lót phân hữu cơ hoai mục và lân."),
                ("Gieo trồng", "Chọn giống tốt, xử lý hạt, gieo đúng mật độ và độ sâu theo giống."),
                ("Bón thúc & tưới", "Chia phân đạm, kali làm 2–3 lần; giữ ẩm đều, nhất là lúc ra hoa."),
                ("Chăm sóc & phòng bệnh", "Làm cỏ, vun gốc, theo dõi sâu bệnh định kỳ, ưu tiên biện pháp sinh học."),
                ("Thu hoạch & bảo quản", "Thu khi đạt độ chín, phơi sấy hoặc bảo quản nơi khô thoáng.")]
STEPS_PERENNIAL = [("Làm đất & đào hố", "Đào hố rộng, bón lót phân hữu cơ và lân, để đất nghỉ ít nhất 2–4 tuần."),
                   ("Trồng cây giống", "Chọn cây giống sạch bệnh, trồng đầu mùa mưa, che bóng và tưới giữ ẩm giai đoạn đầu."),
                   ("Bón phân & tưới", "Bón theo giai đoạn: kiến thiết cơ bản, trước ra hoa, nuôi quả; tưới giữ ẩm đều."),
                   ("Tỉa cành & phòng bệnh", "Tạo tán, tỉa cành thông thoáng; theo dõi sâu bệnh, vệ sinh vườn sau thu hoạch."),
                   ("Thu hoạch & chăm sóc sau thu", "Thu đúng độ chín, cắt tỉa, bón phục hồi để chuẩn bị mùa sau.")]

ADV = {"N": ("Đạm thấp hơn mức cây thường ưa. Bón thêm phân đạm hoặc phân hữu cơ hoai mục, chia làm nhiều lần.",
             "Đạm cao hơn mức điển hình. Giảm hoặc bỏ phân đạm, tránh bón thúc nhiều khiến cây vươn lá, đổ ngã và nhiều sâu bệnh."),
       "P": ("Lân thấp. Bón lót phân lân (super lân, lân nung chảy) kèm phân chuồng hoai.",
             "Lân cao hơn mức điển hình. Có thể gây thiếu kẽm, sắt; tạm ngừng bón lân."),
       "K": ("Kali thấp. Bổ sung phân kali hoặc tro bếp, nhất là giai đoạn nuôi quả, hạt.",
             "Kali cao. Giảm phân kali, theo dõi cân đối canxi và magiê để cây không thiếu chất khác."),
       "temperature": ("Nhiệt độ thấp hơn cây ưa. Chọn mùa vụ ấm hơn, dùng màng phủ hoặc che chắn gió lạnh.",
                       "Nhiệt độ cao hơn cây ưa. Tránh mùa nóng nhất, che lưới giảm nắng, tưới vào sáng sớm hoặc chiều mát."),
       "humidity": ("Không khí khô hơn cây ưa. Tưới phun sương, phủ gốc giữ ẩm, trồng cây chắn gió.",
                    "Độ ẩm cao dễ phát sinh nấm bệnh. Trồng thưa, tỉa cành thông thoáng, vệ sinh vườn."),
       "pH": ("Đất chua hơn cây ưa. Bón vôi hoặc dolomite, kết hợp phân hữu cơ để nâng pH.",
              "Đất kiềm hơn cây ưa. Dùng phân hữu cơ, phân chua sinh lý hoặc lưu huỳnh, tưới đủ nước để rửa bớt muối."),
       "rainfall": ("Lượng mưa thấp hơn nhu cầu. Cần tưới bổ sung (nhỏ giọt, phun), phủ gốc giữ ẩm.",
                    "Lượng mưa cao hơn nhu cầu. Làm luống cao, đào rãnh thoát nước, tránh úng gốc.")}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&display=swap');
:root { --paper:#F6F1E7; --ink:#1E2B22; --forest:#24402E; --moss:#5E7F4A; --clay:#B5651D; --line:#DDD3BE; --muted:#6B7466; }
html, body, [class*="css"], .stMarkdown, button, input, label { font-family: 'Be Vietnam Pro', sans-serif !important; }
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], header[data-testid="stHeader"], footer { display: none !important; }
.block-container { max-width: 1160px; padding-top: 1rem; padding-bottom: 3rem; }
h1, h2, h3, h4, .serif { font-family: 'Fraunces', Georgia, serif !important; color: var(--forest); font-weight: 600 !important; letter-spacing: -.01em; }
div[role="radiogroup"] { gap: .45rem; }
div[role="radiogroup"] label { background: #fff; border: 1px solid var(--line); border-radius: 99px; padding: .25rem .95rem; cursor: pointer; margin: 0; }
div[role="radiogroup"] label > div:first-child { display: none; }
div[role="radiogroup"] label:has(input:checked) { background: var(--forest); border-color: var(--forest); }
div[role="radiogroup"] label:has(input:checked) * { color: #fff !important; }
.stButton > button { width: 100%; border-radius: 10px; border: 1px solid var(--line); background: #fff; color: var(--forest); font-weight: 600; font-size: .84rem; padding: .4rem .6rem; transition: all .15s; }
.stButton > button:hover { border-color: var(--forest); background: var(--forest); color: #fff; }
.stButton > button[kind="primary"] { background: var(--clay); border-color: var(--clay); color: #fff; }
.stButton > button[kind="primary"]:hover { background: #944f13; border-color: #944f13; }
.brand { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 700; color: var(--forest); padding-top: .1rem; }
.brand span { color: var(--clay); }
.eyebrow { font-size: .72rem; letter-spacing: .16em; font-weight: 600; text-transform: uppercase; color: var(--clay); }
.hero { position: relative; border-radius: 24px; overflow: hidden; min-height: 400px; margin: .6rem 0 .9rem; background-size: cover; background-position: center; }
.hero .ov { position: absolute; inset: 0; background: linear-gradient(95deg, rgba(14,32,20,.92) 0%, rgba(14,32,20,.65) 48%, rgba(14,32,20,.1) 100%); }
.hero .tx { position: relative; padding: 3.4rem 3rem; max-width: 640px; color: #fff; }
.hero .eyebrow { color: #F0D9A6; }
.hero h1 { color: #fff; font-size: 3.2rem; line-height: 1.06; margin: .7rem 0 1rem; }
.hero p { color: #E2ECDD; font-size: 1rem; line-height: 1.65; margin: 0; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); margin: 1.6rem 0; }
.stats div { padding: 1.1rem .5rem; text-align: center; border-right: 1px solid var(--line); }
.stats div:last-child { border-right: 0; }
.stats b { display: block; font-family: 'Fraunces', serif; font-size: 2.1rem; color: var(--forest); font-weight: 600; }
.stats span { font-size: .8rem; color: var(--muted); }
.sec { font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 600; color: var(--forest); text-align: center; margin: .2rem 0 .3rem; }
.sub { text-align: center; color: var(--muted); font-size: .92rem; margin-bottom: 1.4rem; }
.center { text-align: center; margin-top: 2.4rem; }
.step { border-top: 2px solid var(--forest); padding-top: .8rem; height: 100%; }
.step .n { font-family: 'Fraunces', serif; font-size: 2.2rem; color: var(--clay); line-height: 1; }
.step h4 { margin: .5rem 0 .3rem; font-size: 1.15rem; }
.step p { margin: 0; color: #4d5a50; font-size: .88rem; line-height: 1.6; }
.pcard { position: relative; border-radius: 18px; overflow: hidden; height: 270px; background-size: cover; background-position: center; box-shadow: 0 2px 10px rgba(30,43,34,.12); margin-bottom: .45rem; }
.pcard.sm { height: 150px; }
.pcard .ov { position: absolute; inset: 0; background: linear-gradient(180deg, rgba(10,24,14,0) 38%, rgba(10,24,14,.86) 100%); }
.pcard .tg { position: absolute; top: 12px; left: 12px; font-size: .65rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 600; color: #fff; background: rgba(14,32,20,.55); backdrop-filter: blur(4px); padding: .2rem .6rem; border-radius: 99px; }
.pcard .tx { position: absolute; left: 16px; right: 16px; bottom: 14px; color: #fff; }
.pcard .nm { font-family: 'Fraunces', serif; font-size: 1.45rem; font-weight: 600; line-height: 1.15; }
.pcard.sm .nm { font-size: 1.15rem; }
.pcard .la { font-style: italic; font-size: .76rem; color: #D9E6D2; }
.pcard .mt { font-size: .74rem; color: #E9EFE4; margin-top: .3rem; }
.pcard .pp { position: absolute; top: 10px; right: 12px; font-weight: 700; color: #fff; background: var(--clay); border-radius: 99px; padding: .1rem .6rem; font-size: .8rem; }
.band { background: var(--forest); border-radius: 24px; padding: 2.4rem 2rem; color: #fff; margin: 2.2rem 0; }
.band .sec { color: #fff; } .band .sub { color: #C9D9C5; } .band .eyebrow { color: #F0B873; }
.band .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
.band .k { border-left: 2px solid #F0B873; padding-left: 1rem; }
.band .k b { display: block; font-family: 'Fraunces', serif; font-size: 2.3rem; font-weight: 600; } .band .k span { font-size: .8rem; color: #C9D9C5; }
.kpi2 { border-top: 2px solid var(--forest); padding-top: .6rem; }
.kpi2 b { display: block; font-family: 'Fraunces', serif; font-size: 1.9rem; color: var(--forest); font-weight: 600; } .kpi2 span { font-size: .78rem; color: var(--muted); }
.cta { position: relative; border-radius: 24px; overflow: hidden; background-size: cover; background-position: center; margin-top: 2rem; }
.cta .ov { position: absolute; inset: 0; background: rgba(20,46,28,.78); }
.cta .tx { position: relative; padding: 2.6rem 2.4rem; color: #fff; }
.cta h2 { color: #fff; margin: 0 0 .3rem; font-size: 2rem; } .cta p { margin: 0; color: #D8E8D4; }
.rescard { position: relative; border-radius: 22px; overflow: hidden; height: 290px; background-size: cover; background-position: center; box-shadow: 0 4px 18px rgba(30,43,34,.2); }
.rescard .ov { position: absolute; inset: 0; background: linear-gradient(180deg, rgba(10,24,14,.05) 30%, rgba(10,24,14,.9) 100%); }
.rescard .tx { position: absolute; left: 22px; right: 22px; bottom: 18px; color: #fff; }
.rescard .rk { font-size: .7rem; letter-spacing: .16em; color: #F0D9A6; font-weight: 600; }
.rescard h2 { color: #fff; font-size: 2.5rem; margin: .1rem 0; line-height: 1.1; }
.rescard .la { font-style: italic; color: #D9E6D2; font-size: .88rem; }
.pill { position: absolute; top: 14px; right: 14px; border-radius: 99px; padding: .25rem .85rem; font-size: .8rem; font-weight: 700; }
.pill.hi { background: #D6EBC8; color: #1f4d2a; } .pill.mid { background: #FBE7B5; color: #735305; } .pill.lo { background: #F8D0CC; color: #8a2a22; }
.warn { background: #FFF3D6; border-left: 3px solid #E0A93B; border-radius: 8px; padding: .6rem .9rem; font-size: .84rem; color: #6d5a1c; margin: .5rem 0; }
.bar { display: flex; align-items: center; gap: .6rem; margin: .45rem 0; font-size: .86rem; color: #33453a; }
.bar .th { width: 30px; height: 30px; border-radius: 50%; flex: none; background-size: cover; background-position: center; background-color: #cdd9c0; }
.bar .l { width: 70px; flex: none; font-weight: 600; } .bar .t { flex: 1; background: #E8E0CC; border-radius: 6px; height: 12px; overflow: hidden; }
.bar .f { height: 100%; background: var(--moss); border-radius: 6px; } .bar .v { width: 44px; text-align: right; font-weight: 700; }
.strip { margin: .5rem 0 .95rem; } .strip .sl { font-size: .78rem; font-weight: 600; color: #33453a; margin-bottom: .35rem; }
.strip .out { color: #b53a2a; } .strip .ok { color: var(--moss); }
.strip .tr { position: relative; height: 10px; background: #E5DDC9; border-radius: 6px; }
.strip .rg { position: absolute; top: 0; height: 10px; background: #A9C98F; border-radius: 6px; }
.strip .md { position: absolute; top: -3px; width: 2px; height: 16px; background: var(--forest); }
.strip .dot { position: absolute; top: -1px; width: 12px; height: 12px; margin-left: -6px; border-radius: 50%; border: 2px solid #fff; box-shadow: 0 0 0 1px #0003; }
.strip .dot.bad { background: #c0392b; } .strip .dot.good { background: var(--forest); }
.strip .ax { display: flex; justify-content: space-between; font-size: .68rem; color: #9a9a88; margin-top: .25rem; }
.adv { border-left: 3px solid #d9534f; background: #FBEDEA; border-radius: 8px; padding: .65rem .9rem; margin: .45rem 0; font-size: .86rem; color: #5a2d28; }
.adv.okk { border-left-color: var(--moss); background: #EAF2E2; color: #264d2c; }
.banner { position: relative; border-radius: 22px; overflow: hidden; min-height: 300px; background-size: cover; background-position: center; margin-bottom: 1rem; }
.banner .ov { position: absolute; inset: 0; background: linear-gradient(90deg, rgba(10,24,14,.88) 0%, rgba(10,24,14,.35) 70%, rgba(10,24,14,.1) 100%); }
.banner .tx { position: relative; padding: 2.4rem 2.4rem; color: #fff; max-width: 620px; }
.banner h1 { color: #fff; margin: .3rem 0 0; font-size: 3rem; } .banner .la { font-style: italic; color: #D9E6D2; }
.chip { display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3); border-radius: 99px; padding: .15rem .7rem; font-size: .76rem; margin: .6rem .35rem .4rem 0; }
.banner p { color: #E6EFE1; margin: .2rem 0 0; }
.info { background: #fff; border: 1px solid var(--line); border-top: 3px solid var(--clay); border-radius: 12px; padding: .9rem 1rem; height: 100%; font-size: .88rem; color: #3f4a40; line-height: 1.55; }
.info b { display: block; font-size: .68rem; letter-spacing: .12em; color: var(--clay); margin-bottom: .3rem; }
.sstep { display: flex; gap: .9rem; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; margin: .5rem 0; font-size: .88rem; color: #3f4a40; }
.sstep .n { font-family: 'Fraunces', serif; font-size: 1.8rem; color: var(--clay); line-height: 1; flex: none; width: 1.6rem; }
.sstep b { color: var(--forest); }
</style>
"""

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
ALIAS = {"maize": ["corn", "bap"], "kidneybeans": ["kidney", "daudo"], "pigeonpeas": ["pigeon", "dautrieu"],
         "mothbeans": ["moth", "daumo"], "mungbean": ["mung", "dauxanh"], "blackgram": ["black", "dauden"],
         "chickpea": ["daugà", "dauga"], "lentil": ["daulang"], "grapes": ["grape"], "muskmelon": ["melon", "cantaloupe", "duale"],
         "watermelon": ["duahau"], "pomegranate": ["luu"], "rice": ["gao", "paddy"], "orange": ["cam"]}



def norm(s):
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def find_images():
    files = {}
    for root, dirs, fns in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in (".venv", "venv", "__pycache__", ".git", "node_modules")]
        for fn in fns:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMG_EXT:
                files.setdefault(norm(stem), os.path.join(root, fn))
    out = {}
    for k, v in CROPS.items():
        cands = [norm(k), norm(v[0])] + [norm(a) for a in ALIAS.get(k, [])]
        hit = next((files[c] for c in cands if c in files), None)
        if not hit:
            hit = next((p for st_, p in files.items() for c in cands if len(c) >= 4 and c in st_), None)
        if hit:
            out[k] = hit
    return out


@st.cache_data(show_spinner=False)
def photo(path, w, h):
    from PIL import Image, ImageOps
    im = ImageOps.fit(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), (w, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


IMGS = find_images()

def home_bg(w=1400, h=560):
    """Lấy ảnh nền trang chủ từ thư mục 22 ảnh."""
    folder = os.path.join(BASE, "22 ảnh")

    if os.path.isdir(folder):
        for fn in os.listdir(folder):
            stem, ext = os.path.splitext(fn)

            if norm(stem) == norm("ảnh nền") and ext.lower() in IMG_EXT:
                p = os.path.join(folder, fn)
                try:
                    return f"background-image:url('{photo(p, w, h)}')"
                except Exception:
                    break

    return "background-image:linear-gradient(135deg,#355a3d,#86a566)"


def bg(k, w=640, h=480):
    """Thuộc tính style nền bằng ảnh thật của cây k (không có ảnh thì dùng nền xanh)."""
    p = IMGS.get(k)
    if p:
        try:
            return f"background-image:url('{photo(p, w, h)}')"
        except Exception:
            pass
    return "background-image:linear-gradient(135deg,#355a3d,#86a566)"


def H(html):
    st.markdown(html, unsafe_allow_html=True)


def vn(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def pc(x, d=1):
    return vn(x * 100, d) + "%"


# ======================================================================
# DỮ LIỆU + MÔ HÌNH
# ======================================================================
def evaluate(m, X, y):
    P = m.predict_proba(X)
    cls = m.classes_
    top = np.argsort(-P, axis=1)
    pred = cls[top[:, 0]]
    top3 = float(np.mean([yy in cls[t[:3]] for yy, t in zip(y, top)]))
    f1 = {}
    for c in cls:
        tp = np.sum((pred == c) & (y == c))
        p = tp / max(np.sum(pred == c), 1)
        r = tp / max(np.sum(y == c), 1)
        f1[c] = float(2 * p * r / (p + r)) if p + r else 0.0
    return {"acc": float(np.mean(pred == y)), "top3": top3, "f1": f1, "macro": float(np.mean(list(f1.values())))}


@st.cache_resource(show_spinner="Đang huấn luyện Random Forest tự cài đặt (lần đầu khoảng 10–20 giây)...")
def load():
    tr = pd.read_csv(os.path.join(BASE, "train.csv"))
    te = pd.read_csv(os.path.join(BASE, "test.csv"))
    tr, te = tr.rename(columns={"Crop": "label"}), te.rename(columns={"Crop": "label"})
    full = MyRandomForest(n_estimators=100, vote="soft", random_state=42).fit(tr[FEATS], tr["label"])
    soil = MyRandomForest(n_estimators=100, vote="soft", random_state=42).fit(tr[SOIL], tr["label"])
    q = tr.groupby("label")[FEATS].quantile([0.1, 0.5, 0.9])
    rng = {k: {f: (float(q.loc[(k, 0.1), f]), float(q.loc[(k, 0.5), f]), float(q.loc[(k, 0.9), f])) for f in FEATS}
           for k in tr["label"].unique()}
    ax = {f: (float(np.floor(tr[f].min())), float(np.ceil(tr[f].max()))) for f in FEATS}
    y = te["label"].to_numpy()
    return {"tr": tr, "te": te, "full": full, "soil": soil, "rng": rng, "ax": ax, "med": tr[FEATS].median(),
            "ev_full": evaluate(full, te[FEATS], y), "ev_soil": evaluate(soil, te[SOIL], y)}


try:
    D = load()
except Exception as e:
    st.error(f"Không đọc được train.csv / test.csv cạnh app.py hoặc huấn luyện lỗi: {e}")
    st.stop()

NAME2KEY = {v[0]: k for k, v in CROPS.items()}
vi = lambda k: CROPS[k][0]


# ======================================================================
# THÀNH PHẦN GIAO DIỆN
# ======================================================================
def go(page, crop=None):
    st.session_state.page = page
    if crop:
        st.session_state.crop = crop
        st.session_state.last_crop = crop


def crop_card(k, prefix):
    n, la, g, days, _ = CROPS[k]
    a, _, b = D["rng"][k]["temperature"]
    H(f"<div class='pcard' style=\"{bg(k)}\"><div class='ov'></div><span class='tg'>{g}</span><div class='tx'>"
      f"<div class='nm'>{n}</div><div class='la'>{la}</div><div class='mt'>{days} · {a:.0f}–{b:.0f} °C</div></div></div>")
    st.button("Xem cách trồng →", key=f"{prefix}_{k}", on_click=go, args=("Cách trồng", k))


def strip(f, k, v=None):
    lo, hi = D["ax"][f]
    a, m, b = D["rng"][k][f]
    p = lambda x: max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))
    tag, mark = "", ""
    if v is not None:
        out = v < a or v > b
        tag = "<span class='out'>(NGOÀI khoảng điển hình)</span>" if out else "<span class='ok'>(trong khoảng điển hình)</span>"
        mark = f"<i class='dot {'bad' if out else 'good'}' style='left:{p(v):.1f}%'></i>"
    return (f"<div class='strip'><div class='sl'>{LBL[f]} {tag}</div><div class='tr'>"
            f"<b class='rg' style='left:{p(a):.1f}%;width:{max(p(b) - p(a), 1):.1f}%'></b>"
            f"<b class='md' style='left:{p(m):.1f}%'></b>{mark}</div>"
            f"<div class='ax'><span>{lo:g}</span><span>{hi:g}</span></div></div>")


def compare(k, vals):
    feats = list(vals) if vals else FEATS
    H("".join(strip(f, k, vals.get(f) if vals else None) for f in feats))
    H("<div style='font-size:.72rem;color:#8a8a7a'>Dải xanh: khoảng điển hình (10%–90% mẫu của cây này) · vạch đậm: trung vị"
      + (" · chấm tròn: giá trị của bạn." if vals else ".") + "</div>")
    if not vals:
        return
    st.markdown(f"##### Điều kiện của bạn cần điều chỉnh gì để trồng {vi(k).lower()}?")
    n_bad = 0
    for f, v in vals.items():
        a, _, b = D["rng"][k][f]
        if v < a or v > b:
            n_bad += 1
            H(f"<div class='adv'><b>{SHORT[f]} = {vn(v)}</b> (khoảng điển hình {vn(a)} – {vn(b)}). {ADV[f][0 if v < a else 1]}</div>")
    if not n_bad:
        H(f"<div class='adv okk'>Tất cả chỉ số bạn nhập đều nằm trong khoảng điển hình của {vi(k).lower()}.</div>")


def faq(items):
    for q, a in items:
        with st.expander(q):
            st.write(a)


# ======================================================================
# ĐIỀU HƯỚNG
# ======================================================================
PAGES = ["Trang chủ", "Gợi ý cây trồng", "Thư viện cây trồng", "Cách trồng", "Về hệ thống"]
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("last_crop", "rice")
H(CSS)
c1, c2 = st.columns([1, 4])
c1.markdown("<div class='brand'>Đất <span>&amp;</span> Cây</div>", unsafe_allow_html=True)
c2.radio("Điều hướng", PAGES, key="page", horizontal=True, label_visibility="collapsed")
page = st.session_state.page
EF, ES = D["ev_full"], D["ev_soil"]
n_all = len(D["tr"]) + len(D["te"])
HERO = next((k for k in ["rice", "coffee", "banana", "mango"] if k in IMGS), next(iter(IMGS), None))

# ======================================================================
# TRANG CHỦ
# ======================================================================
if page == "Trang chủ":
    H(f"<div class='hero' style=\"{bg(HERO, 1400, 560)}\"><div class='ov'></div><div class='tx'><div class='eyebrow'>Chào mừng đến với Đất &amp; Cây</div>"
      "<h1>Chọn đúng cây,<br>trúng mùa vàng</h1><p>Nhập chỉ số đất (N, P, K, pH) và khí hậu (nhiệt độ, độ ẩm, lượng mưa), hệ thống gợi ý loại cây phù hợp nhất "
      "và hướng dẫn cách trồng từ chuẩn bị đất đến thu hoạch.</p></div></div>")
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Bắt đầu gợi ý", type="primary", on_click=go, args=("Gợi ý cây trồng",))
    b2.button("Xem thư viện cây", on_click=go, args=("Thư viện cây trồng",))
    H("<div class='stats'>" + "".join(f"<div><b>{v}</b><span>{t}</span></div>" for v, t in
      [(f"{n_all:,}".replace(",", "."), "mẫu dữ liệu đất và khí hậu"), (str(len(CROPS)), "loại cây trồng"),
       (pc(EF["acc"]), "gợi ý số 1 đúng trên dữ liệu kiểm tra"), (str(len(CROPS)), "cẩm nang trồng chi tiết")]) + "</div>")
    H("<div class='center'><div class='eyebrow'>Đơn giản</div><div class='sec'>Chỉ 3 bước để biết nên trồng gì</div>"
      "<div class='sub'>Không cần kiến thức chuyên môn, chỉ cần các chỉ số trong phiếu phân tích đất</div></div>")
    for col, (n, t, d) in zip(st.columns(3, gap="large"), [("01", "Nhập chỉ số", "Điền N (đạm), P (lân), K (kali), pH và khí hậu nơi trồng. Chưa có số liệu khí hậu vẫn dùng được chế độ chỉ đất."),
                                                         ("02", "Nhận gợi ý", "Random Forest gồm 100 cây quyết định bỏ phiếu, trả về các loại cây phù hợp nhất kèm tỉ lệ tin cậy."),
                                                         ("03", "Xem cách trồng", "Mỗi cây có cẩm nang riêng: mùa vụ, bón phân, tưới, sâu bệnh, thu hoạch và lời khuyên cải tạo đất của bạn.")]):
        col.markdown(f"<div class='step'><div class='n'>{n}</div><h4>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)
    H("<div class='center'><div class='eyebrow'>Thư viện cây trồng</div><div class='sec'>Những cây được quan tâm nhiều</div>"
      "<div class='sub'>Bấm vào từng cây để xem cẩm nang trồng chi tiết</div></div>")
    for col, k in zip(st.columns(4), ["rice", "coffee", "banana", "mango"]):
        with col:
            crop_card(k, "home")
    H("<div class='band'><div class='eyebrow' style='text-align:center'>Vì sao đáng tin</div>"
      f"<div class='sec'>Kết quả kiểm tra trên {len(D['te'])} mẫu chưa từng thấy</div>"
      "<div class='sub'>Mô hình được cài đặt từ công thức và đánh giá trên tập test tách riêng</div><div class='grid'>"
      + "".join(f"<div class='k'><b>{v}</b><span>{t}</span></div>" for v, t in
                [(pc(EF["acc"]), "đúng ngay gợi ý số 1"), (pc(ES["acc"]), "nếu chỉ dùng chỉ số đất"), (pc(EF["macro"]), "Macro F1"), ("100", "cây quyết định bỏ phiếu")])
      + "</div></div>")
    H("<div class='sec'>Bạn có thể thắc mắc</div><div class='sub'></div>")
    faq([("Tôi cần những chỉ số nào để dùng hệ thống?", "N, P, K, pH của đất (có trên phiếu phân tích đất) và nhiệt độ, độ ẩm, lượng mưa trung bình nơi trồng. Chỉ có chỉ số đất vẫn dùng được nhưng kết quả kém chính xác hơn."),
         ("Vì sao hệ thống đưa nhiều gợi ý chứ không chỉ một?", "Nhiều cây có điều kiện sống gần nhau. Việc xem top-3 giúp bạn có thêm lựa chọn khi cây số 1 chưa chắc chắn."),
         ("Kết quả có thay thế được tư vấn của kỹ sư nông nghiệp không?", "Không. Đây là công cụ tham khảo dựa trên dữ liệu; hãy kết hợp với kinh nghiệm địa phương và tư vấn chuyên môn.")])
    H(f"<div class='cta' style=\"{bg('coffee' if 'coffee' in IMGS else HERO, 1400, 400)}\"><div class='ov'></div><div class='tx'>"
      "<h2>Sẵn sàng chọn cây cho mảnh đất của bạn?</h2><p>Chỉ mất đến một phút.</p></div></div>")
    st.write("")
    st.button("Thử ngay", type="primary", on_click=go, args=("Gợi ý cây trồng",), key="cta")

# ======================================================================
# GỢI Ý CÂY TRỒNG
# ======================================================================
elif page == "Gợi ý cây trồng":
    saved = st.session_state.get("saved_in", {})
    for f in FEATS:
        lo, hi = D["ax"][f]
        default = float(round(D["med"][f], 1))
        initial = min(max(float(saved.get(f, default)), lo), hi)
        st.session_state.setdefault("in_" + f, initial)
        st.session_state.setdefault("slider_" + f, initial)

    st.markdown("## Gợi ý cây theo đất và khí hậu")
    st.caption("Kết quả cập nhật mỗi khi bạn thay đổi giá trị. Mỗi chỉ số có thể kéo thanh trượt hoặc nhập trực tiếp.")

    def sync_from_slider(f):
        st.session_state["in_" + f] = st.session_state["slider_" + f]

    def sync_from_number(f):
        st.session_state["slider_" + f] = st.session_state["in_" + f]

    def quick():
        c = st.session_state.quick
        if c in NAME2KEY:
            for f in FEATS:
                lo, hi = D["ax"][f]
                value = float(round(D["rng"][NAME2KEY[c]][f][1], 1))
                value = min(max(value, lo), hi)
                st.session_state["in_" + f] = value
                st.session_state["slider_" + f] = value

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown("#### Nhập chỉ số đất và khí hậu")
        mode = st.radio("Bạn có những số liệu nào?", ["Đất + khí hậu (chính xác nhất)", "Chỉ đất"], horizontal=True)
        st.selectbox("Điền nhanh bằng đất điển hình của một loại cây (tùy chọn)", ["— Tự nhập —"] + sorted(NAME2KEY), key="quick", on_change=quick)
        feats = FEATS if mode.startswith("Đất +") else SOIL
        vals = {}

        for f in feats:
            lo, hi = D["ax"][f]
            stp = LIMIT[f][2]
            with st.container(border=True):
                st.markdown(f"**{LBL[f]}**")
                st.caption(f"Giới hạn mô hình đã học: {vn(lo)} đến {vn(hi)}")
                slider_col, number_col = st.columns([1.35, 1])
                with slider_col:
                    st.slider(
                        "Kéo để chọn giá trị",
                        min_value=float(lo),
                        max_value=float(hi),
                        step=float(stp),
                        key="slider_" + f,
                        on_change=sync_from_slider,
                        args=(f,),
                    )
                with number_col:
                    st.number_input(
                        "Hoặc nhập trực tiếp",
                        min_value=float(lo),
                        max_value=float(hi),
                        step=float(stp),
                        key="in_" + f,
                        format="%.1f",
                        on_change=sync_from_number,
                        args=(f,),
                    )
            vals[f] = st.session_state["in_" + f]

        st.session_state.used_vals = vals
        st.session_state.saved_in = {f: st.session_state["in_" + f] for f in FEATS if "in_" + f in st.session_state}
    model = D["full"] if feats == FEATS else D["soil"]
    pr = model.predict_proba(pd.DataFrame([[vals[f] for f in feats]], columns=feats))[0]
    order = np.argsort(-pr)
    top = [(model.classes_[i], float(pr[i])) for i in order]
    k1, p1 = top[0]
    with right:
        lvl = ("hi", "Độ tin cậy cao") if p1 >= 0.7 else ("mid", "Độ tin cậy trung bình") if p1 >= 0.4 else ("lo", "Độ tin cậy thấp")
        H(f'<div class="rescard" style="{bg(k1, 900, 580)}"><div class="ov"></div><span class="pill {lvl[0]}">{pc(p1, 0)} · {lvl[1]}</span>'
          f"<div class='tx'><div class='rk'>GỢI Ý SỐ 1</div><h2>{vi(k1)}</h2><div class='la'>{CROPS[k1][1]}</div></div></div>")
        st.button(f"Xem cách trồng {vi(k1).lower()}", type="primary", on_click=go, args=("Cách trồng", k1), key="r1")
        c2_, c3_ = st.columns(2)
        for col, (k, p), rk in zip((c2_, c3_), top[1:3], (2, 3)):
            with col:
                H(f'<div class="pcard sm" style="{bg(k, 480, 300)}"><div class="ov"></div><span class="pp">{pc(p, 0)}</span>'
                  f"<div class='tx'><div class='la'>HẠNG {rk}</div><div class='nm'>{vi(k)}</div></div></div>")
                st.button("Xem cách trồng", on_click=go, args=("Cách trồng", k), key=f"r{rk}")
        for f in feats:
            lo, hi = D["ax"][f]
            if not lo <= vals[f] <= hi:
                H(f"<div class='warn'>{SHORT[f]} = {vn(vals[f])} nằm NGOÀI khoảng dữ liệu huấn luyện [{lo:g}, {hi:g}], kết quả có thể không đáng tin.</div>")
        if p1 < 0.5:
            H("<div class='warn'>Độ tin cậy thấp: nhiều loại cây có điều kiện tương tự. Hãy xem cả top-3.</div>")
        if feats == SOIL:
            H("<div class='warn'>Chỉ dùng chỉ số đất nên độ chính xác thấp hơn; thêm nhiệt độ, độ ẩm, lượng mưa để dự đoán chính xác hơn.</div>")
    st.write("")
    a, b = st.columns([1, 1.4], gap="large")
    with a:
        st.markdown("##### Xác suất của 5 cây đứng đầu")
        H("".join(f'<div class="bar"><span class="th" style="{bg(k, 60, 60)}"></span><span class="l">{vi(k)}</span><span class="t"><div class="f" style="width:{p * 100:.1f}%"></div></span><span class="v">{pc(p, 0)}</span></div>' for k, p in top[:5]))
    with b:
        st.markdown(f"##### Số liệu của bạn so với điều kiện điển hình của {vi(k1).lower()}")
        compare(k1, vals)

# ======================================================================
# THƯ VIỆN CÂY TRỒNG
# ======================================================================
elif page == "Thư viện cây trồng":
    st.markdown("## Thư viện cây trồng")
    st.caption(f"{len(CROPS)} loại cây trong hệ thống. Bấm vào từng cây để xem cẩm nang trồng chi tiết.")
    f1_, f2_ = st.columns([1, 3])
    q = f1_.text_input("Tìm theo tên", placeholder="Ví dụ: chuối, banana...")
    g = f2_.radio("Nhóm cây", ["Tất cả"] + GROUPS, horizontal=True)
    keys = [k for k, v in CROPS.items() if (g == "Tất cả" or v[2] == g) and (not q or q.lower() in (v[0] + v[1] + k).lower())]
    keys.sort(key=lambda k: vi(k))
    st.write(f"Tìm thấy {len(keys)} cây.")
    for i in range(0, len(keys), 4):
        for col, k in zip(st.columns(4), keys[i:i + 4]):
            with col:
                crop_card(k, "lib")
    miss = [f"{k} ({vi(k)})" for k in CROPS if k not in IMGS]
    if miss:
        st.warning("Chưa tìm thấy ảnh cho: " + ", ".join(miss) + ". Hãy đặt tên file ảnh theo tên tiếng Anh (vd: rice.jpg, mango.jpg) hoặc tên tiếng Việt không dấu (vd: lua.jpg).")

# ======================================================================
# CÁCH TRỒNG
# ======================================================================
elif page == "Cách trồng":
    st.session_state.setdefault("crop", st.session_state.last_crop)
    k = st.selectbox("Chọn loại cây", list(CROPS), key="crop", format_func=lambda x: vi(x))
    st.session_state.last_crop = k
    n, la, g, days, note = CROPS[k]
    R = D["rng"][k]
    H(f"<div class='banner' style=\"{bg(k, 1400, 560)}\"><div class='ov'></div><div class='tx'><div class='eyebrow' style='color:#F0D9A6'>{g}</div>"
      f"<h1>{n}</h1><div class='la'>{la}</div><div><span class='chip'>{days}</span><span class='chip'>{R['temperature'][0]:.0f}–{R['temperature'][2]:.0f} °C</span></div><p>{note}</p></div></div>")
    i1, i2, i3 = st.columns(3)
    i1.markdown(f"<div class='info'><b>MÙA VỤ (THAM KHẢO)</b>{GROUP_SEASON[g]}</div>", unsafe_allow_html=True)
    i2.markdown(f"<div class='info'><b>NƯỚC</b>Lượng mưa điển hình {R['rainfall'][0]:.0f}–{R['rainfall'][2]:.0f} mm; độ ẩm không khí {R['humidity'][0]:.0f}–{R['humidity'][2]:.0f}%.</div>", unsafe_allow_html=True)
    i3.markdown(f"<div class='info'><b>ĐẤT NÊN</b>pH {vn(R['pH'][0])}–{vn(R['pH'][2])}; N {R['N'][0]:.0f}–{R['N'][2]:.0f}, P {R['P'][0]:.0f}–{R['P'][2]:.0f}, K {R['K'][0]:.0f}–{R['K'][2]:.0f}.</div>", unsafe_allow_html=True)
    st.markdown("### Điều kiện đất và khí hậu phù hợp (theo dữ liệu)")
    st.caption("Khoảng điển hình = 10%–90% mẫu của cây trong dữ liệu. Dữ liệu không nêu đơn vị đo của N, P, K.")
    compare(k, st.session_state.get("used_vals"))
    st.markdown("### Quy trình trồng từ A đến Z")
    for i, (t, d) in enumerate(STEPS_PERENNIAL if k in PERENNIAL else STEPS_ANNUAL, 1):
        H(f"<div class='sstep'><span class='n'>{i}</span><div><b>{t}</b><br>{d}</div></div>")
    st.caption("Quy trình mang tính tổng quát, không thay thế hướng dẫn kỹ thuật của địa phương.")

# ======================================================================
# VỀ HỆ THỐNG
# ======================================================================
else:
    st.markdown("## Về hệ thống")
    st.caption("Dành cho người muốn hiểu hệ thống hoạt động và đáng tin đến đâu.")
    H("<div class='center' style='margin-top:1rem'><div class='eyebrow'>Cách hoạt động</div><div class='sec'>Random Forest: 100 cây quyết định cùng bỏ phiếu</div><div class='sub'></div></div>")
    for col, (n, t, d) in zip(st.columns(4, gap="large"), [("01", "Nhập", "Người dùng nhập N, P, K, pH của đất và nhiệt độ, độ ẩm, lượng mưa nơi trồng."),
                                                         ("02", "100 cây cùng xét", "Mỗi cây học trên một mẫu bootstrap khác nhau, mỗi lần tách chỉ xét ngẫu nhiên 3 chỉ số."),
                                                         ("03", "Bỏ phiếu mềm", f"Mỗi cây cho xác suất của {len(CROPS)} loại cây; rừng lấy trung bình các xác suất."),
                                                         ("04", "Gợi ý", "Chọn các cây có xác suất cao nhất, kèm cảnh báo nếu dữ liệu nhập nằm ngoài vùng đã học.")]):
        col.markdown(f"<div class='step'><div class='n'>{n}</div><h4>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)
    H("<div class='center'><div class='eyebrow'>Độ chính xác</div>"
      f"<div class='sec'>Kết quả trên {len(D['te'])} mẫu chưa dùng khi huấn luyện</div><div class='sub'></div></div>")
    for col, (v, t) in zip(st.columns(5), [(pc(EF["acc"]), "đúng ngay gợi ý số 1"), (pc(EF["top3"]), "đúng trong 3 gợi ý đầu"), (pc(EF["macro"]), "Macro F1"),
                                           (pc(D["full"].oob_score_), "OOB accuracy"), (pc(ES["acc"]), "nếu chỉ dùng chỉ số đất")]):
        col.markdown(f"<div class='kpi2'><b>{v}</b><span>{t}</span></div>", unsafe_allow_html=True)
    st.caption(f"MyRandomForest tự cài đặt (thuat_toan.py): 100 cây, tiêu chí gini, độ sâu không giới hạn, bỏ phiếu mềm, huấn luyện trên {len(D['tr'])} mẫu × 7 chỉ số. Tập test tách riêng, không dùng để chọn tham số.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Chỉ số nào quan trọng nhất?")
        imp = sorted(zip(FEATS, D["full"].feature_importances_), key=lambda t: -t[1])
        H("".join(f"<div class='bar'><span class='l' style='width:90px'>{SHORT[f]}</span><span class='t'><div class='f' style='width:{v / imp[0][1] * 100:.0f}%'></div></span><span class='v'>{pc(v, 0)}</span></div>" for f, v in imp))
        st.caption("Tính theo độ giảm tạp chất (Gini) trung bình trên các cây.")
    with b:
        st.markdown("##### Thêm khí hậu giúp được bao nhiêu?")
        st.dataframe(pd.DataFrame({"Chỉ số dùng": ["Chỉ đất (N, P, K, pH)", "Đất + khí hậu (7 chỉ số)"],
                                   "Đúng gợi ý số 1": [pc(ES["acc"]), pc(EF["acc"])], "Đúng trong top-3": [pc(ES["top3"]), pc(EF["top3"])]}), hide_index=True)
        st.caption("Chỉ dùng đất, nhiều cây có đất giống nhau nên khó phân biệt; thêm nhiệt độ, độ ẩm, lượng mưa giúp tách chúng.")
    st.markdown("##### F1 theo từng loại cây (cây nào dễ nhầm?)")
    st.bar_chart(pd.Series({vi(k): v for k, v in EF["f1"].items()}).sort_values(), horizontal=True)
    H("<div class='center'><div class='eyebrow'>Minh bạch</div><div class='sec'>Hạn chế cần biết</div><div class='sub'></div></div>")
    st.markdown("- Chỉ được huấn luyện trên **%d loại cây** có trong dữ liệu.\n"
                "- Dùng riêng chỉ số đất thì độ chính xác chỉ khoảng %s; thêm nhiệt độ, độ ẩm, lượng mưa thì lên %s.\n"
                "- Dữ liệu gần giống dữ liệu mẫu công khai, mang tính học thuật; ngoài vùng đã học kết quả có thể sai (hệ thống sẽ cảnh báo).\n"
                "- Đơn vị đo của N, P, K trong dữ liệu không được nêu rõ, hãy dùng cùng thang đo với phiếu phân tích của bạn.\n"
                "- Nội dung cẩm nang là kiến thức nông học phổ biến, mang tính tham khảo, không thay thế tư vấn của kỹ sư nông nghiệp." % (len(CROPS), pc(ES["acc"], 0), pc(EF["acc"], 0)))
    st.markdown("### Câu hỏi thường gặp")
    faq([("Vì sao chọn Random Forest?", "Nhiều cây quyết định bỏ phiếu cho kết quả ổn định hơn một cây đơn lẻ, xử lý tốt dữ liệu số và cho được tỉ lệ tin cậy."),
         ("Hệ thống có học thêm từ dữ liệu tôi nhập không?", "Không. Mô hình được huấn luyện sẵn từ train.csv; dữ liệu bạn nhập chỉ dùng để dự đoán."),
         ("Vì sao một số cây khác nhau có kết quả giống nhau?", "Các loại đậu có điều kiện đất và khí hậu rất giống nhau nên dễ bị nhầm, xem biểu đồ F1 ở trên."),
         ("Tôi nhập giá trị ngoài khoảng thì sao?", "Hệ thống vẫn dự đoán nhưng hiện cảnh báo vì mô hình chưa từng gặp giá trị như vậy.")])
