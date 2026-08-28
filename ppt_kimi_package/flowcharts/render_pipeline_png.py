from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
from pathlib import Path

W, H = 1920, 1080
OUT = Path(__file__).with_name("agent_geoseg_pipeline.png")

def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size, index=0)
        except Exception:
            continue
    return ImageFont.load_default()

def lerp(a, b, t):
    return int(a + (b - a) * t)

def gradient_bg():
    img = Image.new("RGB", (W, H))
    pix = img.load()
    c1 = (6, 17, 31)
    c2 = (16, 24, 39)
    for y in range(H):
        for x in range(W):
            t = (x / W * 0.55 + y / H * 0.45)
            pix[x, y] = tuple(lerp(c1[i], c2[i], t) for i in range(3))
    return img.convert("RGBA")

def rounded_rect_with_shadow(base, xy, radius=28, fill=(18,36,64,245), outline=(31,59,90,255)):
    x0, y0, x1, y1 = xy
    shadow = Image.new("RGBA", base.size, (0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x0, y0+16, x1, y1+16), radius=radius, fill=(0,0,0,95))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    base.alpha_composite(shadow)
    d = ImageDraw.Draw(base)
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)

def arrow(d, start, end, color=(0,212,255,235), width=4):
    d.line([start, end], fill=color, width=width)
    ang = math.atan2(end[1]-start[1], end[0]-start[0])
    size = 18
    p1 = end
    p2 = (end[0]-size*math.cos(ang-math.pi/6), end[1]-size*math.sin(ang-math.pi/6))
    p3 = (end[0]-size*math.cos(ang+math.pi/6), end[1]-size*math.sin(ang+math.pi/6))
    d.polygon([p1,p2,p3], fill=color)

img = gradient_bg()
d = ImageDraw.Draw(img)

# Decorative glows
glow = Image.new("RGBA", img.size, (0,0,0,0))
gd = ImageDraw.Draw(glow)
gd.ellipse((1380,-130,1900,390), fill=(0,212,255,34))
gd.ellipse((-90,620,530,1240), fill=(124,58,237,45))
img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10)))

title_f = font(54, True)
subtitle_f = font(24)
step_title_f = font(30, True)
step_text_f = font(20)
num_f = font(22, True)
small_f = font(18)
tag_f = font(18, True)

d.text((110, 72), "Agent + Image Recognition Pipeline", fill=(248,250,252), font=title_f)
d.text((112, 128), "从文献图像 / 反演结果图中自动识别地质分区与界面，并生成 SEM 正演建模输入", fill=(167,199,217), font=subtitle_f)

steps = [
    (110, 240, "1", "输入资料", ["PDF / 文献图片", "概念模型 / 反演图"]),
    (505, 240, "2", "Agent 理解", ["判断图像类型", "规划处理步骤"]),
    (900, 240, "3", "目标检测", ["Panel 裁剪", "Colorbar / 坐标轴"]),
    (1295, 240, "4", "图像预处理", ["去文字 / 去噪", "颜色校正"]),
    (1295, 590, "5", "分区识别", ["颜色聚类 / SLIC", "区域融合 / 边界引导"]),
    (900, 590, "6", "质量审查", ["Agent 视觉评估", "失败重试 / 人工反馈"]),
    (505, 590, "7", "界面提取", ["Label map → 边界", "多边形 / 分区模型"]),
    (110, 590, "8", "SEM 输入", ["速度 / 材料分区", "SPECFEM 正演准备"]),
]

for x, y, n, title, lines in steps:
    rounded_rect_with_shadow(img, (x, y, x+250, y+190))
    d = ImageDraw.Draw(img)
    d.ellipse((x+19, y+19, x+57, y+57), fill=(0,212,255))
    d.text((x+31, y+26), n, fill=(8,17,31), font=num_f)
    d.text((x+30, y+76), title, fill=(248,250,252), font=step_title_f)
    d.text((x+30, y+116), lines[0], fill=(183,198,214), font=step_text_f)
    d.text((x+30, y+146), lines[1], fill=(183,198,214), font=step_text_f)

# Arrows
arrow(d, (380,335), (485,335))
arrow(d, (775,335), (880,335))
arrow(d, (1170,335), (1275,335))
arrow(d, (1420,450), (1420,565))
arrow(d, (1275,685), (1170,685))
arrow(d, (880,685), (775,685))
arrow(d, (485,685), (380,685))

# Human-in-loop pill
d.rounded_rectangle((650,860,1270,946), radius=43, fill=(0,212,255,36), outline=(0,212,255,180), width=2)
d.text((700, 883), "Human-in-the-loop：用户只在关键节点确认、修正或筛选代表性结果", fill=(159,180,200), font=small_f)
d.text((700, 914), "Agent 将自然语言反馈转化为重分割、参数调整和质量审查动作", fill=(159,180,200), font=small_f)

# Tags
tags = [("Agent", 1335, 875, 130, (0,212,255)), ("Computer Vision", 1483, 875, 170, (255,176,0)), ("SEM", 1671, 875, 128, (167,139,250))]
for text, x, y, w, color in tags:
    d.rounded_rectangle((x,y,x+w,y+34), radius=17, fill=color)
    d.text((x+22, y+7), text, fill=(6,17,31), font=tag_f)

img.save(OUT)
print(OUT)
