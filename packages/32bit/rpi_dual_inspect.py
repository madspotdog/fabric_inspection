# -*- coding: utf-8 -*-
"""32-bit Raspberry Pi dual-model inspector; single frame by default."""
from __future__ import annotations
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np

FONT_CANDIDATES = ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                   "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                   "/usr/share/opentype/noto/NotoSansCJK-Regular.ttc")

COTTON_DEFAULT = ("Tier0_Low", "Tier1_Mid", "Tier2_High")
FIBER_DEFAULT = ("丝", "亚麻", "其他", "棉", "涤纶", "羊毛")

def cton(x):
    return {"Tier0_Low":"低棉", "Tier1_Mid":"中棉", "Tier2_High":"高棉",
            "低棉":"低棉", "中棉":"中棉", "高棉":"高棉"}.get(x, "其他")

def fton(x):
    return {"棉":"纯棉", "羊毛":"纯羊毛", "涤纶":"纯涤纶", "纯棉":"纯棉",
            "纯羊毛":"纯羊毛", "纯涤纶":"纯涤纶", "丝":"其他", "亚麻":"其他",
            "其他":"其他"}.get(x, "其他")

def labels(param, default):
    try:
        return tuple(json.loads(Path(param).with_name("labels.json").read_text(encoding="utf-8")))
    except Exception:
        return tuple(default)

def crop(image, size, roi=660):
    import cv2
    h, w = image.shape[:2]; side = min(h, w, roi)
    y, x = (h-side)//2, (w-side)//2
    return cv2.resize(image[y:y+side, x:x+side], (size, size), interpolation=cv2.INTER_AREA)

def probs(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if np.all(v >= -1e-5) and abs(float(v.sum())-1) < 1e-3:
        return v / max(float(v.sum()), 1e-12)
    v -= v.max(); p = np.exp(v); return p / max(float(p.sum()), 1e-12)

class Net:
    def __init__(self, param, names, size, threads, imagenet, fp16):
        import ncnn
        self.ncnn, self.names, self.size, self.imagenet = ncnn, names, size, imagenet
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_packed = bool(fp16)
        self.net.opt.use_fp16_storage = bool(fp16)
        self.net.opt.use_packing_layout = True
        if self.net.load_param(str(param)) != 0 or self.net.load_model(str(Path(param).with_suffix('.bin'))) != 0:
            raise RuntimeError("模型加载失败: " + str(param))
    def predict(self, image):
        rgb = crop(image, self.size)[:, :, ::-1].copy()
        mat = self.ncnn.Mat.from_pixels(rgb.tobytes(), self.ncnn.Mat.PixelType.PIXEL_RGB, self.size, self.size)
        if self.imagenet:
            mat.substract_mean_normalize([.485*255,.456*255,.406*255], [1/(.229*255),1/(.224*255),1/(.225*255)])
        else:
            mat.substract_mean_normalize([0,0,0], [1/255,1/255,1/255])
        ex = self.net.create_extractor(); ex.input("in0", mat)
        code, out = ex.extract("out0")
        if code != 0: raise RuntimeError("推理失败")
        p = probs(out); i = int(p.argmax()); return self.names[i], float(p[i]), p

def inspect(cotton, fiber, image, args):
    import cv2
    t = time.perf_counter(); ft_raw, fc, fp = fiber.predict(image)
    ft = fton(ft_raw)
    cotton_index = next((i for i, name in enumerate(fiber.names) if name == "棉"), None)
    cotton_gate_prob = float(fp[cotton_index]) if cotton_index is not None else 0.0
    cotton_entered = cotton_gate_prob >= args.cotton_gate
    ct_raw, cc, cp = "未运行", 0.0, np.zeros(3, dtype=np.float32)
    ct = "不适用"
    cotton_margin = 0.0
    if cotton_entered:
        ct_raw, cc, cp = cotton.predict(image)
        ct = cton(ct_raw)
        order = np.sort(cp)
        cotton_margin = float(order[-1] - order[-2])
        if cc < args.cotton_threshold or cotton_margin < args.cotton_margin: ct = "待定"
    if fc < args.fiber_threshold: ft = "其他"
    passed = cotton_entered and ((ct == "低棉" and ft == "其他") or (ct == "高棉" and ft == "纯棉"))
    roi = crop(image, 660, 660)
    focus = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if focus < args.focus_threshold: passed = False
    return {"cotton_tier":ct, "cotton_raw":ct_raw, "cotton_conf":round(cc,5),
            "pure_fiber":ft, "pure_fiber_raw":ft_raw, "pure_fiber_conf":round(fc,5),
            "cross_check":"通过" if passed else "复核", "elapsed_ms":round((time.perf_counter()-t)*1000,1),
            "focus":round(focus,1), "cotton_margin":round(cotton_margin,5),
            "cotton_gate_prob":round(cotton_gate_prob,5), "cotton_entered":cotton_entered,
            "cotton_probs":[round(float(x),5) for x in cp], "fiber_probs":[round(float(x),5) for x in fp]}

def font(size):
    from PIL import ImageFont
    for name in FONT_CANDIDATES:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()

def panel(image, result):
    import cv2
    from PIL import Image, ImageDraw
    h,w=image.shape[:2]; pw=min(500,max(400,w//3)); x0=w-pw
    cv2.rectangle(image,(x0,0),(w,h),(12,18,26),-1)
    pil=Image.fromarray(image[:,:,::-1]); d=ImageDraw.Draw(pil); x=x0+24
    d.text((x,18),'面料智能检测',font=font(30),fill='white'); d.line((x,68,w-24,68),fill=(105,125,145),width=2)
    y=100
    for label,value,conf in [('棉含量档位',result['cotton_tier'],result['cotton_conf']),('纯面料判断',result['pure_fiber'],result['pure_fiber_conf'])]:
        d.text((x,y),label,font=font(21),fill=(220,232,242)); d.text((x,y+30),value,font=font(36),fill='white'); d.text((x,y+78),f'置信度 {conf:.1%}',font=font(18),fill=(225,235,242)); y+=130
    d.text((x,y-12),f"棉门控 P(棉) {result['cotton_gate_prob']:.1%}",font=font(18),fill=(255,225,90))
    d.text((x,y+14),'已进入棉含量识别' if result['cotton_entered'] else '未进入棉含量识别',font=font(18),fill=(225,235,242)); y+=48
    color=(28,145,75) if result['cross_check']=='通过' else (190,105,20)
    d.text((x,y),'最终判定',font=font(21),fill='white'); d.rounded_rectangle((x,y+32,w-24,y+106),radius=10,fill=color,outline='white',width=2)
    d.text((x+18,y+42),result['cross_check'],font=font(44),fill='white')
    d.text((x,y+124),f"推理时间  {result['elapsed_ms']:.0f} ms",font=font(26),fill=(255,225,90))
    focus_color=(80,220,125) if result['focus']>=75 else (255,175,55)
    d.text((x,y+164),f"清晰度  {result['focus']:.0f}",font=font(21),fill=focus_color)
    d.text((x,h-40),'SPACE 识别    Q / ESC 退出',font=font(18),fill=(205,220,232))
    return np.asarray(pil)[:,:,::-1].copy()

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent; m = here / "models"
    ap.add_argument("--cotton-param", default=str(m/"cotton_tier/model.ncnn.param")); ap.add_argument("--fiber-param", default=str(m/"pure_fiber/model.ncnn.param"))
    ap.add_argument("--image"); ap.add_argument("--dir"); ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720); ap.add_argument("--mjpg", action="store_true")
    ap.add_argument("--cotton-threshold", type=float, default=.55); ap.add_argument("--cotton-margin", type=float, default=.10)
    ap.add_argument("--cotton-gate", type=float, default=.20)
    ap.add_argument("--fiber-threshold", type=float, default=.50); ap.add_argument("--focus-threshold", type=float, default=75.0)
    ap.add_argument("--no-fp16", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); fp16 = not a.no_fp16
    cotton = Net(a.cotton_param, labels(a.cotton_param,COTTON_DEFAULT),384,a.threads,True,fp16)
    fiber = Net(a.fiber_param, labels(a.fiber_param,FIBER_DEFAULT),288,a.threads,False,fp16)
    if a.check:
        print(inspect(cotton,fiber,np.zeros((720,1280,3),np.uint8),a)); return
    import cv2
    if a.image or a.dir:
        files=[a.image] if a.image else sorted(sum([glob.glob(os.path.join(a.dir,e)) for e in ('*.jpg','*.jpeg','*.png','*.bmp')],[]))
        for p in files:
            im=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR)
            print(os.path.basename(p), '->', inspect(cotton,fiber,im,a) if im is not None else '读取失败')
        return
    cap=cv2.VideoCapture(a.camera if a.camera is not None else 0)
    if not cap.isOpened(): raise SystemExit("打不开摄像头")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height)
    if a.mjpg: cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    print("空格识别，q/ESC退出；固定中心660 ROI，默认单帧，不做表决")
    result=None
    while True:
        ok,frame=cap.read()
        if not ok: break
        view=frame.copy(); cv2.putText(view,'USB microscope',(10,30),0,.8,(0,255,0),2)
        h,w=frame.shape[:2]; side=min(h,w,660); x1=(w-side)//2; y1=(h-side)//2
        cv2.rectangle(view,(x1,y1),(x1+side,y1+side),(0,255,0),2)
        if result: view=panel(view,result)
        cv2.imshow('dual fabric inspect 32bit',view); key=cv2.waitKey(1)&255
        if key in (ord('q'),27): break
        if key==ord(' '):
            result=inspect(cotton,fiber,frame,a); print(result)
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__': main()
