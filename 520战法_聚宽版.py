# -*- coding: utf-8 -*-
"""
520 战法 · 聚宽(JoinQuant)回测版
==================================
对应文档：520战法_规则说明书_v22.md
运行方式：聚宽「策略研究 → 回测」，周期选 **分钟**，股票池不必选（脚本自己筛）。

一、结构
    initialize          参数、费率、四个定时任务
    prepare_daily       09:35  重建股票池 + 用昨日及以前日线算形态快照
    check_s4            every_bar 盘中 S4：价格回落到前高 H 才卖 1/3
    decide_at_1455      14:55  先卖出(S1>S2>S3 / 突破仓) 再买入
    mark_s4_expired     14:57  跳空在前高上方、全天不回落 → S4 名额作废

二、规则 → 代码 索引
    R0 金叉G早于回踩段          _scan_structure() 里 G 的搜索
    R1 MA5>MA20                decide_at_1455() 的 ma5_t > ma20_t
    R2 存在回踩段 + 不破20线    _scan_structure() 的 S 与两条作废判断
    R3 温和带量                _buy_scan() 的 vol >= TOL*VMA5 且 >= TOL*VMA10
    R4 收盘>MA5                _buy_scan() 的 price > ma5_t
    R5 非巨量                  _buy_scan() 的 vol <= 2*VMA10
    N1 末段涨幅+踩20线          _scan_structure() 的 last5_gain / touched_ma20
    N2 距平均成本5%~20%         _lookup_vwap() 的 120 日 VWAP 替代
    S1 破20线清仓              decide_at_1455() 第一段
    S2 破5线减1/3              decide_at_1455() 第二段
    S3 巨量减1/3               decide_at_1455() 第三段
    S4 摸前高减1/3             check_s4()
    S5 减满3份即清仓           _cut() 里的 cuts
    摊薄成本                    _cut() 里重算
    突破仓                      _buy_scan() 记类型 + decide 里的分支

三、与文档的差异（详见《520战法_聚宽版_说明.md》）
    1. N2 的平均成本：聚宽没有筹码分布，用 120 日 VWAP 代替
    2. 次新：文档只说"剔除次新"，这里按上市满 3 年
    3. 只做主板（沪 600/601/603/605，深 000/001）
    4. 温和带量的 VMA5/VMA10 含当日累计量（同花顺口径），可关
    5. S4 用每分钟 last_price 采样 high/low，不用逐分钟 get_price
"""

import numpy as np
import pandas as pd
from jqdata import *


# ==================== 参数区（改这里就够） ====================
CFG = {
    # --- 仓位 ---
    'MAX_HOLD': 5,              # 同时最多持有 5 只
    'MAX_WEIGHT': 0.25,         # 单只上限 25%

    # --- 股票池 ---
    'MIN_LIST_DAYS': 365 * 3,   # 次新：上市不满 3 年剔除
    'BOARDS': ('600', '601', '603', '605', '000', '001'),   # 只做主板

    # --- 指标窗口 ---
    'LOOKBACK': 60,             # 形态回溯天数（要够找金叉 G）
    'VWAP_DAYS': 120,           # N2 用的 VWAP 窗口

    # --- 阈值 ---
    'TOL_VOL': 0.9,             # 温和带量容差
    'BIG_VOL': 2.0,            # 巨量 = BIG_VOL × VMA10
    'BUF': 1.01,                # 跌破缓冲：收盘 × 1.01 < 均线
    'N1_GAIN': 0.15,            # 末段 5 根涨幅否决线
    'N2_LO': 0.05,              # 距 VWAP 下界
    'N2_HI': 0.20,              # 距 VWAP 上界
    'S3_GAP': 1.05,             # 两次 S3 之间的最小涨幅

    # --- 费用（聚宽 OrderCost）---
    'COMMISSION': 0.00025,      # 佣金 万 2.5（双向）
    'TRANSFER': 0.00001,        # 过户费 十万分之一（折进佣金）
    'STAMP': 0.001,             # 印花税 千分之一（仅卖出）
    'MIN_COMMISSION': 5,         # 单笔最低 5 元
    'SLIPPAGE': 0.002,          # 滑点 0.2%

    # --- 其他 ---
    'VMA_INCLUDE_TODAY': True,  # 量能均线是否含当日累计量（同花顺口径）
}


def initialize(context):
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_benchmark('000300.XSHG')
    log.set_level('order', 'error')

    # 交易成本：佣金+过户费折一起；印花税只在卖出收
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=CFG['STAMP'],
        open_commission=CFG['COMMISSION'] + CFG['TRANSFER'],
        close_commission=CFG['COMMISSION'] + CFG['TRANSFER'],
        close_today_commission=0,
        min_commission=CFG['MIN_COMMISSION'],
    ), type='stock')
    set_slippage(PriceRelatedSlippage(CFG['SLIPPAGE']))

    # 运行期状态
    g.hist = {}     # code -> dict(close, vol, ma5, ma20, last_date)
    g.snap = {}     # code -> 形态快照（只存"昨天仍在回踩段里"的票）
    g.pos = {}      # code -> 仓位账本
    g.pool = []

    run_daily(prepare_daily, '09:35')
    run_daily(check_s4, 'every_bar')
    run_daily(decide_at_1455, '14:55')
    run_daily(mark_s4_expired, '14:57')


# ==================== 工具 ====================

def _prev_trade_day(context):
    days = get_trade_days(end_date=context.current_dt.date(), count=2)
    return days[-2]


def _is_main_board(code):
    return code[:3] in CFG['BOARDS']


def _today_vol_upto(context, code):
    """当日 09:30 至当前分钟的成交量合计（14:55 时即当日累计量）。"""
    start = context.current_dt.replace(hour=9, minute=30, second=0)
    df = get_price(code, start_date=start, end_date=context.current_dt,
                   frequency='1m', fields=['volume'], skip_paused=True, fq=None)
    if df is None or len(df) == 0:
        return 0.0
    return float(np.nansum(df['volume'].values))


# ==================== 09:35 准备 ====================

def prepare_daily(context):
    today = context.current_dt.date()
    prev = _prev_trade_day(context)

    g.pool = _build_pool(context, today)
    g.snap = {}
    if not g.pool:
        return

    df = get_price(g.pool, end_date=prev, count=CFG['LOOKBACK'],
                   fields=['open', 'close', 'high', 'low', 'volume'],
                   fq='pre', frequency='daily', panel=False)
    if df is None or len(df) == 0:
        return

    piv = {f: df.pivot(index='time', columns='code', values=f) for f in
           ('open', 'close', 'high', 'low', 'volume')}
    close_df, vol_df = piv['close'], piv['volume']
    ma5_df = close_df.rolling(5).mean()
    ma20_df = close_df.rolling(20).mean()

    g.hist = {}
    for code in close_df.columns:
        c = close_df[code].values.astype(float)
        v = vol_df[code].values.astype(float)
        m5 = ma5_df[code].values.astype(float)
        m20 = ma20_df[code].values.astype(float)
        if len(c) < 25 or np.isnan(c[-1]) or np.isnan(m20[-1]) or np.isnan(v[-1]):
            continue
        if not np.isfinite(c[-20:]).all():
            continue
        g.hist[code] = dict(close=c, vol=v, ma5=m5, ma20=m20)

        st = _scan_structure(c, v, m5, m20)
        if st is not None:
            g.snap[code] = st

    log.info('[%s] 池 %d，回踩候选 %d' % (today, len(g.hist), len(g.snap)))


def _build_pool(context, today):
    secs = get_all_securities(types=['stock'], date=today)
    codes = []
    for code, row in secs.iterrows():
        if not _is_main_board(code):
            continue
        if (pd.Timestamp(today) - pd.Timestamp(row['start_date'])).days < CFG['MIN_LIST_DAYS']:
            continue
        codes.append(code)
    if not codes:
        return []

    # 剔除 ST
    try:
        st = get_extras('is_st', codes, end_date=today, count=1, df=True)
        if st is not None and len(st) > 0:
            st_row = st.iloc[-1]
            codes = [c for c in codes if not bool(st_row.get(c, False))]
    except Exception as e:
        log.warn('is_st 取不到，跳过 ST 过滤：%s' % e)
    return codes


def _scan_structure(c, v, m5, m20):
    """形态引擎。c/v/m5/m20 都是截止到"昨天"的数组。返回快照 dict 或 None。

    只挑"昨天仍然收在 MA5 下方"的票——今天收盘站回 MA5 就是买点候选。
    """
    n = len(c)
    if not (c[-1] < m5[-1]):
        return None

    # --- R2：回踩段起点 S（往前扫，直到上一根还在 MA5 上方）---
    i = n - 1
    while i - 1 >= 0 and c[i - 1] < m5[i - 1]:
        i -= 1
    S = i
    if S <= 0:
        return None

    # --- R0：G = 最近一次 MA5 上穿 MA20，必须早于 S ---
    G = -1
    for j in range(S - 1, 0, -1):
        if m5[j] > m20[j] and m5[j - 1] <= m20[j - 1]:
            G = j
            break
    if G < 0:
        return None

    seg = range(S, n)
    # --- R2'：回踩途中任一根有效跌破 20 日线 → 作废 ---
    for j in seg:
        if c[j] * CFG['BUF'] < m20[j]:
            return None
    # --- 作废：回踩途中 MA5 下穿 MA20 ---
    for j in seg:
        if m5[j] < m20[j]:
            return None

    # --- 前高 H：从 S-1 往左，第一个局部高点（收盘） ---
    H = None
    for j in range(S - 1, G, -1):
        if c[j] >= c[j - 1] and c[j] >= c[j + 1]:
            H = float(c[j])
            break
    if H is None:
        H = float(np.max(c[G:S]))

    # --- N1 用：上涨段最后 5 根累计涨幅 ---
    last5 = None
    if S - 5 >= G:
        base = c[S - 5]
        if base > 0:
            last5 = float(c[S - 1] / base - 1.0)

    # --- N1 用：回踩段内是否出现过 收盘 <= MA20 ---
    touched_ma20 = any(c[j] <= m20[j] for j in seg)

    return dict(S=S, G=G, H=H, last5=last5, touched_ma20=touched_ma20)


def _today_ma(hist, price, dt_vol):
    """把 14:55 的当时价 / 当日累计量并进序列，算含当日的均线。"""
    c, v = hist['close'], hist['vol']
    ma5 = (np.nansum(c[-4:]) + price) / 5.0
    ma20 = (np.nansum(c[-19:]) + price) / 20.0
    if CFG['VMA_INCLUDE_TODAY']:
        vma5 = (np.nansum(v[-4:]) + dt_vol) / 5.0
        vma10 = (np.nansum(v[-9:]) + dt_vol) / 10.0
    else:
        vma5 = np.nanmean(v[-5:])
        vma10 = np.nanmean(v[-10:])
    return ma5, ma20, vma5, vma10


# ==================== 盘中：S4 ====================

def check_s4(context):
    now = context.current_dt
    today = now.date()
    if now.hour < 9 or (now.hour == 9 and now.minute < 31):
        return
    if now.hour >= 15:
        return

    data = get_current_data()
    for code in list(context.portfolio.positions.keys()):
        p = g.pos.get(code)
        if p is None or p['type'] != '回踩' or p['touched']:
            continue
        if p['buy_date'] == today:          # T+1
            continue

        px = data[code].last_price
        if px is None or not np.isfinite(px) or px <= 0:
            continue

        # 当日最高/最低采样（每天重置）
        if p.get('hi_date') != today:
            p['hi_date'] = today
            p['day_hi'] = px
            p['day_lo'] = px
        p['day_hi'] = max(p['day_hi'], px)
        p['day_lo'] = min(p['day_lo'], px)

        H = p['H']
        # 价格到过 H 之上（或正好到 H），随后价格回到 H 及以下 → 挂在 H 的卖单成交
        if p['day_hi'] >= H and px <= H:
            if _cut(context, code, p, px, 'S4摸前高') > 0:
                p['touched'] = True
                p['sold_today'] = today


def mark_s4_expired(context):
    """跳空开在前高上方、全天没回落 → S4 名额作废（不卖、不再挂单）。"""
    today = context.current_dt.date()
    for code in list(context.portfolio.positions.keys()):
        p = g.pos.get(code)
        if p is None or p['type'] != '回踩' or p['touched']:
            continue
        if p.get('hi_date') != today:
            continue
        if p['day_lo'] > p['H']:
            p['touched'] = True
            log.info('[%s] %s 全天在前高上方，S4 名额作废' % (today, code))


# ==================== 14:55 决策 ====================

def decide_at_1455(context):
    today = context.current_dt.date()
    data = get_current_data()

    # ---------- 第一步：卖出（S1 > S2 > S3 / 突破仓） ----------
    for code in list(context.portfolio.positions.keys()):
        p = g.pos.get(code)
        hist = g.hist.get(code)
        pos = context.portfolio.positions[code] if code in context.portfolio.positions else None
        if p is None or hist is None or pos is None or pos.total_amount <= 0:
            _forget(code)
            continue

        px = data[code].last_price
        if px is None or not np.isfinite(px) or px <= 0:
            continue

        dt_vol = _today_vol_upto(context, code)
        ma5_t, ma20_t, vma5_t, vma10_t = _today_ma(hist, px, dt_vol)

        # --- S1：收盘有效跌破 20 日线 → 清仓（不看盈亏）---
        if px * CFG['BUF'] < ma20_t:
            _cut(context, code, p, px, 'S1破20线', all_out=True)
            continue

        # --- 突破仓：从 T+1 起，第一个"收盘 > 摊薄成本"的日子卖 1/3（替代 S4）---
        if p['type'] == '突破' and not p['touched'] and px > p['cost']:
            _cut(context, code, p, px, '突破仓止盈')
            p['touched'] = True
            p['sold_today'] = today
            continue

        # --- 浮盈前置：收盘 <= 摊薄成本 → S2/S3 不动 ---
        if px <= p['cost']:
            continue

        # --- S2：跌破 5 日线（穿越事件，1% 缓冲）---
        prev_close = hist['close'][-1]
        prev_ma5 = hist['ma5'][-1]
        if px * CFG['BUF'] < ma5_t and not (prev_close * CFG['BUF'] < prev_ma5):
            _cut(context, code, p, px, 'S2破5线')
            continue

        # --- S3：巨量日（最低优先级）---
        if dt_vol > CFG['BIG_VOL'] * vma10_t and p.get('sold_today') != today:
            last_s3 = p.get('s3_last_close')
            if last_s3 is None or px > last_s3 * CFG['S3_GAP']:
                _cut(context, code, p, px, 'S3巨量')
                p['s3_last_close'] = px
                p['sold_today'] = today

    # ---------- 第二步：买入 ----------
    if len(context.portfolio.positions) >= CFG['MAX_HOLD']:
        return
    if context.portfolio.available_cash <= 0:
        return

    cands = _buy_scan(context, data, today)
    for item in cands:
        if len(context.portfolio.positions) >= CFG['MAX_HOLD']:
            break
        if context.portfolio.available_cash <= 0:
            break
        _buy(context, item)


def _buy_scan(context, data, today):
    """14:55 逐个候选复核 R1/R3/R4/R5 + N1 + N2，返回按优先度排序的买入清单。"""
    raw = []
    for code, st in g.snap.items():
        if code in context.portfolio.positions:
            continue
        hist = g.hist.get(code)
        if hist is None:
            continue
        d = data[code]
        if d.paused or d.is_st:
            continue
        px = d.last_price
        if px is None or not np.isfinite(px) or px <= 0:
            continue
        # 涨停不追（买不到）
        if d.high_limit and px >= d.high_limit:
            continue

        ma5_t, ma20_t, vma5_t, vma10_t = _today_ma(hist, px, 0.0)
        # --- R1 ---
        if ma5_t <= ma20_t:
            continue
        # --- R4：收盘（14:55 价）站回 5 日线 ---
        if px <= ma5_t:
            continue
        raw.append((code, st, hist, px, ma5_t, ma20_t, vma5_t, vma10_t))

    if not raw:
        return []

    # 只对价格已过关的票取分钟量（省时间）
    out = []
    for code, st, hist, px, ma5_t, ma20_t, vma5_t, vma10_t in raw:
        dt_vol = _today_vol_upto(context, code)
        if dt_vol <= 0:
            continue
        _, _, vma5_t, vma10_t = _today_ma(hist, px, dt_vol)

        # --- R3：温和带量（涨停日本来就不买，故豁免在此不会用到）---
        if dt_vol < CFG['TOL_VOL'] * vma5_t or dt_vol < CFG['TOL_VOL'] * vma10_t:
            continue
        # --- R5：当日不是巨量日 ---
        if dt_vol > CFG['BIG_VOL'] * vma10_t:
            continue
        # --- N1：末段 5 根涨幅 > 15% 且回踩踩到 20 日线 → 冲高出货，不做 ---
        if (st['last5'] is not None and st['last5'] > CFG['N1_GAIN']
                and st['touched_ma20']):
            continue

        out.append(dict(code=code, price=px, H=st['H'],
                        type='突破' if px >= st['H'] else '回踩'))

    if not out:
        return []

    # --- N2：距 120 日 VWAP 5% ~ 20% ---
    vwap = _lookup_vwap(context, [o['code'] for o in out])
    out = [o for o in out if _n2_pass(vwap.get(o['code']), o['price'])]
    if not out:
        return []

    # 优先度：回踩仓优先；同类型里"离前高越远空间越大"优先
    out.sort(key=lambda o: (0 if o['type'] == '回踩' else 1,
                            -(o['H'] / o['price'] - 1.0)))
    return out


def _lookup_vwap(context, codes):
    """N2 的替代：用 120 日成交额加权均价 Σ(收盘×量)/Σ量 代替同花顺"平均成本"。"""
    prev = _prev_trade_day(context)
    df = get_price(codes, end_date=prev, count=CFG['VWAP_DAYS'],
                   fields=['close', 'volume'], fq='pre', frequency='daily',
                   panel=False)
    res = {}
    if df is None or len(df) == 0:
        return res
    for code, sub in df.groupby('code'):
        pv = float(np.nansum(sub['close'].values * sub['volume'].values))
        vv = float(np.nansum(sub['volume'].values))
        if vv > 0:
            res[code] = pv / vv
    return res


def _n2_pass(vwap, px):
    if vwap is None or not np.isfinite(vwap) or vwap <= 0:
        return False           # 拿不到 → 跳过该票
    r = px / vwap - 1.0
    return CFG['N2_LO'] <= r <= CFG['N2_HI']


# ==================== 交易与账本 ====================

def _cut_shares(p):
    """每次减原始量的 1/3；第 3 次把余数一次清掉。"""
    left = p['left']
    if left <= 0:
        return 0
    if p['cuts'] >= 2:
        return left
    q = int(p['orig'] / 3.0 / 100.0) * 100
    return max(0, min(q, left))


def _cut(context, code, p, px, reason, all_out=False):
    shares = p['left'] if all_out else _cut_shares(p)
    if shares <= 0:
        _forget_if_empty(code, p)
        return 0

    rec = order(code, -shares)
    filled = int(rec.filled) if rec is not None and rec.filled else 0
    if filled <= 0:
        return 0
    deal = float(rec.price) if rec is not None and rec.price else px

    p['proceeds'] += deal * filled
    p['left'] -= filled
    p['cuts'] += 1
    if p['left'] > 0:
        p['cost'] = (p['amount'] - p['proceeds']) / p['left']
    else:
        log.info('[%s] %s 清仓（%s），共 %d 刀' % (
            context.current_dt.date(), code, reason, p['cuts']))
        _forget(code)
    return filled


def _buy(context, item):
    code, px = item['code'], item['price']
    slots = CFG['MAX_HOLD'] - len(context.portfolio.positions)
    if slots <= 0:
        return
    cash = context.portfolio.available_cash
    tv = context.portfolio.total_value
    budget = min(CFG['MAX_WEIGHT'] * tv, cash / slots)
    shares = int(budget / px / 100.0) * 100
    if shares < 100:
        return

    rec = order(code, shares)
    filled = int(rec.filled) if rec is not None and rec.filled else 0
    if filled <= 0:
        return
    deal = float(rec.price) if rec is not None and rec.price else px

    g.pos[code] = dict(
        orig=filled, left=filled, buys=filled,
        amount=deal * filled, proceeds=0.0, cost=deal,
        type=item['type'], H=item['H'],
        touched=False, cuts=0,
        buy_date=context.current_dt.date(),
        sold_today=None, s3_last_close=None,
        hi_date=None, day_hi=None, day_lo=None,
    )
    log.info('[%s] 买入 %s %d 股 @ %.2f（%s，前高 %.2f）' % (
        context.current_dt.date(), code, filled, deal, item['type'], item['H']))


def _forget(code):
    if code in g.pos:
        del g.pos[code]


def _forget_if_empty(code, p):
    if p['left'] <= 0:
        _forget(code)
