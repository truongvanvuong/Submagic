#!/usr/bin/env python3
"""So sanh file SRT app trich xuat voi file SRT chuan (dap an).

Cach dung:
    python eval_srt.py dap_an.srt app_ra.srt
    python eval_srt.py dap_an.srt app_ra.srt --tol 0.2 --csv bao_cao.csv

Chi dung thu vien chuan cua Python, khong can cai them.
Do duoc: ti le bat duoc sub (recall), ti le sub thua (precision),
ti le loi ky tu (CER), do lech thoi gian dau/cuoi, va xu huong lech
(co dinh hay tang dan) de chan doan loi thoi gian.
"""
import argparse
import csv
import re
import statistics
import sys
import unicodedata
from difflib import SequenceMatcher

TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")
    entries = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        for i, ln in enumerate(lines):
            m = TIME_RE.search(ln)
            if m:
                g = m.groups()
                start = _to_sec(*g[0:4])
                end = _to_sec(*g[4:8])
                text = " ".join(x.strip() for x in lines[i + 1:])
                entries.append({"start": start, "end": end, "text": text})
                break
    entries.sort(key=lambda e: e["start"])
    return entries


# Fallback Traditional to Simplified Chinese character mapping dictionary
T2S_FALLBACK_MAP = {
    "誰": "谁", "敗": "败", "別": "别", "俠": "侠", "沒": "没", "無": "无",
    "說": "说", "麼": "么", "聽": "听", "認": "认", "點": "点", "處": "处",
    "這": "这", "個": "个", "們": "们", "頭": "头", "興": "兴", "東": "东",
    "來": "来", "帶": "带", "後": "后", "為": "为", "時": "时", "會": "会",
    "過": "过", "發": "发", "學": "学", "樣": "样", "對": "对", "開": "开",
    "現": "现", "國": "国", "動": "动", "見": "见", "親": "亲", "經": "经",
    "進": "进", "長": "长", "問": "问", "間": "间", "題": "题", "電": "电",
    "車": "车", "話": "话", "機": "机", "風": "风", "飛": "飞", "寫": "写",
    "體": "体", "聲": "声", "書": "书", "門": "门", "傳": "传", "師": "师",
    "業": "业", "務": "务", "華": "华", "萬": "万", "應": "应", "總": "总",
    "義": "义", "關": "关", "與": "与", "條": "条", "內": "内", "連": "连",
    "據": "据", "兩": "两", "幾": "几", "號": "号", "剛": "刚", "創": "创",
    "劇": "剧", "辦": "办", "協": "协", "參": "参", "雙": "双", "變": "变",
    "員": "员", "區": "区", "醫": "医", "賣": "卖", "單": "单", "嚴": "严",
    "團": "团", "圖": "图", "園": "园", "圍": "围", "場": "场", "報": "报",
    "壓": "压", "壞": "坏", "壯": "壮", "壽": "寿", "備": "备", "複": "复",
    "婦": "妇", "媽": "妈", "實": "实", "審": "审", "導": "导", "尋": "寻",
    "專": "专", "將": "将", "屬": "属", "屢": "屡", "歲": "岁", "島": "岛",
    "嶺": "岭", "嶽": "岳", "幫": "帮", "廣": "广", "庫": "库", "店": "店",
    "廢": "废", "強": "强", "歸": "归", "當": "当", "錄": "录", "徹": "彻",
    "徵": "征", "憶": "忆", "懷": "怀", "態": "态", "憐": "怜", "戀": "恋",
    "恆": "恒", "惡": "恶", "惱": "恼", "懸": "悬", "戰": "战", "戶": "户",
    "戲": "戏", "擴": "扩", "掃": "扫", "揚": "扬", "護": "护", "擔": "担",
    "擇": "择", "擊": "击", "擋": "挡", "擬": "拟", "攬": "揽", "數": "数",
    "斷": "断", "舊": "旧", "晉": "晋", "晝": "昼", "顯": "显", "曉": "晓",
    "暫": "暂", "權": "权", "極": "极", "標": "标", "檢": "检", "樓": "楼",
    "樂": "乐", "樹": "树", "橋": "桥", "檔": "档", "歷": "历", "殘": "残",
    "殺": "杀", "殼": "壳", "氣": "气", "漢": "汉", "湯": "汤", "決": "决",
    "況": "况", "沖": "冲", "溝": "沟", "治": "治", "泉": "泉", "淚": "泪",
    "準": "准", "滅": "灭", "滿": "满", "滯": "滞", "潔": "洁", "濟": "济",
    "濤": "涛", "濃": "浓", "燈": "灯", "營": "营", "燒": "烧", "熱": "热",
    "愛": "爱", "獨": "独", "獲": "获", "環": "环", "產": "产", "畫": "画",
    "療": "疗", "盜": "盗", "盡": "尽", "盤": "盘", "盧": "卢", "眾": "众",
    "確": "确", "碼": "码", "礎": "础", "禮": "礼", "祥": "祥", "禍": "祸",
    "福": "福", "離": "离", "種": "种", "積": "积", "稱": "称", "穩": "稳",
    "穫": "获", "窩": "窝", "窮": "穷", "窺": "窥", "競": "竞", "簡": "简",
    "算": "算", "管": "管", "篇": "篇", "籤": "签", "類": "类", "紀": "纪",
    "約": "约", "紅": "红", "納": "纳", "紙": "纸", "級": "级", "紛": "纷",
    "素": "素", "索": "索", "紫": "紫", "細": "细", "終": "终", "組": "组",
    "結": "结", "給": "给", "絡": "络", "統": "统", "絲": "丝", "綜": "综",
    "綠": "绿", "網": "网", "緊": "紧", "線": "线", "編": "编", "練": "练",
    "緣": "缘", "縛": "缚", "縣": "县", "績": "绩", "繁": "繁", "纖": "纤",
    "繼": "继", "續": "续", "缺": "缺", "罰": "罚", "罷": "罢", "羅": "罗",
    "習": "习", "翻": "翻", "翼": "翼", "聖": "圣", "聞": "闻", "聯": "联",
    "聰": "聪", "職": "职", "肅": "肃", "膽": "胆", "腦": "脑", "腳": "脚",
    "脫": "脱", "臉": "脸", "臺": "台", "舉": "举", "艦": "舰", "藝": "艺",
    "節": "节", "藍": "蓝", "薦": "荐", "藥": "药", "蘇": "苏", "蘋": "苹",
    "虛": "虚", "蟲": "虫", "蟻": "蚁", "蠻": "蛮", "術": "术", "衛": "卫",
    "衝": "冲", "衰": "衰", "補": "补", "裝": "装", "裡": "里", "製": "制",
    "訊": "讯", "記": "记", "訪": "访", "設": "设", "許": "许", "訴": "诉",
    "診": "诊", "詞": "词", "評": "评", "識": "识", "試": "试", "詩": "诗",
    "該": "该", "詳": "详", "語": "语", "誕": "诞", "誘": "诱", "誠": "诚",
    "誤": "误", "調": "调", "論": "论", "請": "请", "諸": "诸", "諾": "诺",
    "讀": "读", "讓": "让", "讚": "赞", "貴": "贵", "費": "费", "賀": "贺",
    "賈": "贾", "資": "资", "賓": "宾", "賜": "赐", "賞": "赏", "賢": "贤",
    "質": "质", "購": "购", "賽": "赛", "贊": "赞", "贈": "赠", "贏": "赢",
    "趨": "趋", "踴": "踊", "軍": "军", "軒": "轩", "軟": "软", "較": "较",
    "輔": "辅", "輕": "轻", "載": "载", "輛": "辆", "輩": "辈", "輪": "轮",
    "輯": "辑", "輸": "输", "轉": "转", "轟": "轰", "辭": "辞", "辯": "辩",
    "農": "农", "迎": "迎", "返": "返", "週": "周", "運": "运", "達": "达",
    "遷": "迁", "選": "选", "還": "还", "邊": "边", "邏": "逻", "鄉": "乡",
    "釋": "释", "針": "针", "釣": "钓", "鈴": "铃", "銀": "银", "銷": "销",
    "鋒": "锋", "銳": "锐", "錯": "错", "錢": "钱", "錦": "锦", "錫": "锡",
    "鍊": "链", "鍵": "键", "鎖": "锁", "鐘": "钟", "鐵": "铁", "鑑": "鉴",
    "閃": "闪", "閏": "闰", "閑": "闲", "閔": "闵", "閣": "阁", "閱": "阅",
    "闊": "阔", "隊": "队", "陽": "阳", "階": "阶", "際": "际", "陸": "陆",
    "陳": "陈", "陰": "阴", "險": "险", "隨": "随", "隱": "隐", "隻": "只",
    "難": "难", "雲": "云", "靈": "灵", "靜": "静", "靴": "靴", "韓": "韩",
    "韻": "韵", "響": "响", "頁": "页", "頂": "顶", "項": "项", "順": "顺",
    "須": "须", "預": "预", "頑": "顽", "頓": "顿", "頗": "颇", "領": "领",
    "頡": "颉", "頻": "频", "額": "额", "願": "愿", "顧": "顾", "飯": "饭",
    "飲": "饮", "飾": "饰", "養": "养", "館": "馆", "馬": "马", "馳": "驰",
    "駁": "驳", "駐": "驻", "駕": "驾", "驗": "验", "駿": "骏", "騎": "骑",
    "騙": "骗", "髮": "发", "鬧": "闹", "鬱": "郁", "鬼": "鬼", "魚": "鱼",
    "鮮": "鲜", "鳥": "鸟", "鳴": "鸣", "鴨": "鸭", "鴻": "鸿", "鵝": "鹅",
    "鶴": "鹤", "鷹": "鹰", "麥": "麦", "黃": "黄", "黨": "党", "齊": "齐",
    "齒": "齿", "齡": "龄", "龍": "龙", "龜": "龟",
}

_opencc_t2s = None


def convert_t2s(text: str) -> str:
    """Chuyen doi chu Han Phon the sang Gian the bang OpenCC hoac tu dien fallback."""
    global _opencc_t2s
    if _opencc_t2s is None:
        try:
            import opencc
            _opencc_t2s = opencc.OpenCC("t2s")
        except Exception:
            _opencc_t2s = False

    if _opencc_t2s is not False and _opencc_t2s is not None:
        try:
            return _opencc_t2s.convert(text)
        except Exception:
            pass

    return "".join(T2S_FALLBACK_MAP.get(ch, ch) for ch in text)


def strip_vietnamese_diacritics(text: str) -> str:
    """Loai bo dau thanh va cac ky tu co dau trong tieng Viet de tinh CER khong dau."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", text)


def normalize(text: str, fold_script: str = "none", strip_diacritics: bool = False) -> str:
    if fold_script == "t2s":
        text = convert_t2s(text)
    if strip_diacritics:
        text = strip_vietnamese_diacritics(text)
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        ch for ch in text
        if not ch.isspace() and unicodedata.category(ch)[0] not in ("P", "S")
    )


def edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match(gt, pred, window, min_sim, fold_script: str = "none"):
    """Ghep 1-1 dong dap an voi dong app: gan thoi gian, giong noi dung nhat."""
    cands = []
    for gi, g in enumerate(gt):
        gn = normalize(g["text"], fold_script=fold_script)
        for pi, p in enumerate(pred):
            near = abs(p["start"] - g["start"]) <= window or (
                p["start"] < g["end"] and p["end"] > g["start"]
            )
            if not near:
                continue
            pn = normalize(p["text"], fold_script=fold_script)
            sim = SequenceMatcher(None, gn, pn).ratio() if (gn or pn) else 0.0
            if sim >= min_sim:
                cands.append((sim, gi, pi))
    cands.sort(reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for sim, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, sim))
    pairs.sort()
    return pairs, used_g, used_p


def linear_fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def fmt(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ground_truth", help="SRT chuan (dap an)")
    ap.add_argument("predicted", help="SRT do app trich xuat")
    ap.add_argument("--tol", type=float, default=0.2, help="nguong lech thoi gian chap nhan, giay (mac dinh 0.2)")
    ap.add_argument("--window", type=float, default=3.0, help="cua so tim dong tuong ung, giay (mac dinh 3.0)")
    ap.add_argument("--min-sim", type=float, default=0.5, help="do giong noi dung toi thieu de coi la khop (mac dinh 0.5)")
    ap.add_argument(
        "--fold-script",
        choices=["none", "t2s"],
        default="none",
        help="chuan hoa he chu: none (mac dinh) hoac t2s (Phon the sang Gian the)",
    )
    ap.add_argument(
        "--vietnamese-diacritics",
        action="store_true",
        help="danh gia ca CER tieu chuan va CER bo qua dau tieng Viet",
    )
    ap.add_argument("--csv", help="ghi bao cao tung dong ra file CSV")
    args = ap.parse_args()

    gt = parse_srt(args.ground_truth)
    pred = parse_srt(args.predicted)
    if not gt:
        sys.exit("Khong doc duoc dong nao trong file dap an.")

    pairs, used_g, used_p = match(gt, pred, args.window, args.min_sim, fold_script=args.fold_script)

    matched = len(pairs)
    recall = matched / len(gt)
    precision = matched / len(pred) if pred else 0.0

    fold_mode = args.fold_script
    err = sum(edit_distance(normalize(gt[g]["text"], fold_script=fold_mode), normalize(pred[p]["text"], fold_script=fold_mode)) for g, p, _ in pairs)
    total_chars = sum(len(normalize(gt[g]["text"], fold_script=fold_mode)) for g, _, _ in pairs)
    cer = err / total_chars if total_chars else 0.0
    exact = sum(1 for g, p, _ in pairs if normalize(gt[g]["text"], fold_script=fold_mode) == normalize(pred[p]["text"], fold_script=fold_mode))

    s_off = [pred[p]["start"] - gt[g]["start"] for g, p, _ in pairs]
    e_off = [pred[p]["end"] - gt[g]["end"] for g, p, _ in pairs]

    print("=" * 60)
    if args.fold_script != "none":
        print(f"Chuan hoa chu (fold-script) : {args.fold_script}")
    print(f"Dong dap an: {len(gt)}   Dong app tra ve: {len(pred)}   Khop: {matched}")
    print(f"Recall (bat duoc sub)    : {recall:6.1%}   ({len(gt) - matched} dong bi sot)")
    print(f"Precision (khong thua)   : {precision:6.1%}   ({len(pred) - matched} dong thua/sai)")
    print(f"Doc dung hoan toan       : {exact}/{matched} dong khop")
    print(f"Ti le loi ky tu (CER)    : {cer:6.1%}   (chi tinh tren cac dong khop)")

    if args.vietnamese_diacritics and matched:
        err_vn = sum(edit_distance(normalize(gt[g]["text"], fold_script=fold_mode, strip_diacritics=True), normalize(pred[p]["text"], fold_script=fold_mode, strip_diacritics=True)) for g, p, _ in pairs)
        total_chars_vn = sum(len(normalize(gt[g]["text"], fold_script=fold_mode, strip_diacritics=True)) for g, _, _ in pairs)
        cer_vn = err_vn / total_chars_vn if total_chars_vn else 0.0
        exact_vn = sum(1 for g, p, _ in pairs if normalize(gt[g]["text"], fold_script=fold_mode, strip_diacritics=True) == normalize(pred[p]["text"], fold_script=fold_mode, strip_diacritics=True))
        print(f"CER (bo qua dau tieng Viet): {cer_vn:6.1%}")
        print(f"Doc dung (bo qua dau)    : {exact_vn}/{matched} dong khop")

    if matched:
        def stats(name, offs):
            abs_off = [abs(x) for x in offs]
            within = sum(1 for x in abs_off if x <= args.tol) / len(abs_off)
            print(
                f"{name}: trung vi {statistics.median(offs) * 1000:+7.0f} ms | "
                f"lech tuyet doi TB {statistics.mean(abs_off) * 1000:6.0f} ms | "
                f"trong +-{args.tol * 1000:.0f} ms: {within:5.1%}"
            )
        stats("Thoi gian BAT DAU ", s_off)
        stats("Thoi gian KET THUC", e_off)

        fit = linear_fit([gt[g]["start"] for g, _, _ in pairs], s_off)
        if fit:
            slope, intercept = fit
            per_min = slope * 60 * 1000
            print(f"Xu huong lech bat dau    : {intercept * 1000:+.0f} ms luc dau, thay doi {per_min:+.0f} ms moi phut")
            if abs(per_min) > 100:
                print("  -> Lech TANG DAN theo thoi gian: nghi do tinh thoi gian bang so khung chia fps (VFR).")
            elif abs(statistics.median(s_off)) > 0.15:
                print("  -> Lech CO DINH: nghi do tua keyframe (-ss) hoac start_time cua video khac 0.")
            else:
                print("  -> Khong thay lech co he thong; sai so con lai chu yeu do lay mau thua.")

    missed = [i for i in range(len(gt)) if i not in used_g]
    if missed:
        print("-" * 60)
        print("CAC DONG BI SOT:")
        for i in missed:
            print(f"  [{fmt(gt[i]['start'])}] {gt[i]['text']}")

    extra = [i for i in range(len(pred)) if i not in used_p]
    if extra:
        print("-" * 60)
        print("CAC DONG THUA/SAI (app co nhung dap an khong co):")
        for i in extra[:30]:
            print(f"  [{fmt(pred[i]['start'])}] {pred[i]['text']}")
        if len(extra) > 30:
            print(f"  ... va {len(extra) - 30} dong nua")
    print("=" * 60)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["trang_thai", "gt_start", "gt_end", "gt_text", "pred_start", "pred_end", "pred_text", "do_giong", "lech_start_ms", "lech_end_ms"])
            pair_by_g = {g: (p, s) for g, p, s in pairs}
            for gi, g in enumerate(gt):
                if gi in pair_by_g:
                    p, s = pair_by_g[gi]
                    q = pred[p]
                    w.writerow(["khop", g["start"], g["end"], g["text"], q["start"], q["end"], q["text"], f"{s:.2f}",
                                round((q["start"] - g["start"]) * 1000), round((q["end"] - g["end"]) * 1000)])
                else:
                    w.writerow(["sot", g["start"], g["end"], g["text"], "", "", "", "", "", ""])
            for pi in extra:
                q = pred[pi]
                w.writerow(["thua", "", "", "", q["start"], q["end"], q["text"], "", "", ""])
        print(f"Da ghi bao cao chi tiet: {args.csv}")


if __name__ == "__main__":
    main()
