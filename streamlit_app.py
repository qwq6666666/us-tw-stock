"""美股 × 台股整合分析：網頁介面（Streamlit）。資料來自 data/ 資料夾，由每日排程更新。

版面：總覽（儀表板）／個股（選一檔，走勢、體質、機率、新聞一次看）／市場（績效、連動、ADR）／新聞／模型（成績單與驗證）。
顏色採台股慣例：紅漲、綠跌。
"""
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
import fundamentals
import model as probmodel
import outlook
import scorecard
import relstrength
import trend
import universe
import volmodel

st.set_page_config(page_title='美股 × 台股 整合分析', page_icon='📈', layout='wide', initial_sidebar_state='collapsed')

UP, DOWN, FLAT = '#e5484d', '#2f9e6b', '#8a8f98'      # 台股慣例：紅漲、綠跌

st.markdown('''
<style>
.block-container {max-width: 1200px; padding-top: 2.2rem;}
h1 {font-size: 1.9rem !important; margin-bottom: 0 !important;}
.kgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 8px;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 12px 16px; background: rgba(128,128,128,.07);}
.kpi .t {font-size: .82rem; opacity: .7;}
.kpi .v {font-size: 1.65rem; font-weight: 700; line-height: 1.35;}
.kpi .c {font-size: .95rem; font-weight: 600;}
.kpi .s {font-size: .78rem; opacity: .65;}
.pill {display: inline-block; padding: 2px 11px; border-radius: 999px; font-size: .82rem; margin: 2px 6px 2px 0;
       border: 1px solid rgba(128,128,128,.3); background: rgba(128,128,128,.10);}
.hero {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 16px 20px; background: rgba(128,128,128,.06);}
.hero .n {font-size: 1.25rem; font-weight: 700;}
.hero .p {font-size: 2rem; font-weight: 800; line-height: 1.3;}
.note {font-size: .85rem; opacity: .75;}
button[data-baseweb="tab"] {font-size: 1rem;}
</style>
''', unsafe_allow_html=True)


def _mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


# 以檔案修改時間當快取鍵：每日排程存入新資料後，網頁會自動讀到最新內容
@st.cache_resource(show_spinner=False)
def get_market(mtime):
    return core.load_market()


@st.cache_data(show_spinner=False)
def get_news(mtime):
    return core.load_news()


@st.cache_data(ttl=1800, show_spinner='向 Yahoo Finance 查詢中…')
def fetch_live(ticker):
    df = core.download_close([ticker])
    return df[ticker].dropna() if ticker in df.columns else None


def show(fig):
    st.plotly_chart(fig, width='stretch')


def color_of(x):
    return UP if x > 0 else DOWN if x < 0 else FLAT


def chg_html(x, fmt='{:+.2%}'):
    arrow = '▲' if x > 0 else '▼' if x < 0 else '■'
    return f'<span style="color:{color_of(x)}">{arrow} {fmt.format(x)}</span>'


def kpi(title, value, change_html='', sub=''):
    return (f'<div class="kpi"><div class="t">{title}</div><div class="v">{value}</div>'
            f'<div class="c">{change_html}</div><div class="s">{sub}</div></div>')


def price_chart(s, title, days=None):
    if days:
        s_full, s = s, s.tail(days)
    else:
        s_full = s
    ma = {n: s_full.rolling(n).mean().reindex(s.index) for n in (20, 60, 240)}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=s.index, y=s, name='收盤價', line=dict(width=2, color='#3b6fe0')), row=1, col=1)
    for (n, nm), c in zip(((20, '月線'), (60, '季線'), (240, '年線')), ('#f5a623', '#9b59d0', '#7f8c8d')):
        fig.add_trace(go.Scatter(x=s.index, y=ma[n], name=nm, line=dict(width=1.1, color=c)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s.index, y=core.rsi(s_full).reindex(s.index), name='RSI', showlegend=False,
                             line=dict(color='#3b6fe0', width=1.2)), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=UP, opacity=0.08, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=DOWN, opacity=0.08, line_width=0, row=2, col=1)
    fig.update_layout(title=title, height=500, hovermode='x unified', margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation='h', y=1.1))
    fig.update_yaxes(range=[0, 100], title_text='RSI', row=2, col=1)
    return fig


def lookup(code, M):
    """回傳（名稱, 價格序列, 錯誤訊息）。清單內的用已存資料，其餘現場查詢。"""
    code = code.strip().upper()
    candidates = [code + '.TW', code + '.TWO'] if re.fullmatch(r'\d{4,6}[A-Z]?', code) else [code]
    for c in candidates:
        if c in M.close_all.columns:
            return core.label(c), M.close_all[c].dropna(), None
    err = ''
    for c in candidates:
        try:
            s = fetch_live(c)
            if s is not None and len(s) > 5:
                return c, s, None
        except Exception as e:
            err = f'（{type(e).__name__}）'
    return None, None, (f'查不到「{code}」的資料{err}。請確認代號：台股輸入數字即可，美股輸入英文代號。'
                        '若代號正確，可能是資料來源暫時限流，請稍後再試。')


# ───────────── 載入資料 ─────────────
META = json.load(open(core.META_JSON, encoding='utf-8')) if os.path.exists(core.META_JSON) else {}

st.title('📈 美股 × 台股 整合分析')
stale = ''
try:
    age = (datetime.now() - datetime.strptime(META['updated_at'], '%Y-%m-%d %H:%M')).days
    stale = f'　⚠️ 已 {age} 天沒更新' if age >= 4 else ''
except Exception:
    pass
st.caption(f"資料更新：{META.get('updated_at', '尚未更新')}（台北時間）{stale}　｜　僅供研究與學習，不構成投資建議　｜　顏色：紅漲、綠跌")

if not os.path.exists(core.PRICES_CSV):
    st.warning('尚無資料。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
    st.stop()

M = get_market(_mtime(core.PRICES_CSV))
NEWS = get_news(_mtime(core.NEWS_CSV))
SUMMARY = open(core.SUMMARY_MD, encoding='utf-8').read() if os.path.exists(core.SUMMARY_MD) else ''
FUND = fundamentals.load()
PROB = probmodel.load()
VOL = volmodel.load()
TR = trend.load()
REL = relstrength.load()


@st.cache_data(show_spinner=False)
def get_uni(mtime):
    return universe.load()


UNI = get_uni(_mtime(universe.PATHS['volume']))
# 標的清單：台股個股 → 美股個股 → 指數
ORDER = ([t for t in M.tw_cols if not t.startswith('^')] + [t for t in M.us_cols if not t.startswith('^')]
         + [t for t in M.tw_cols + M.us_cols if t.startswith('^')])

TREND_SHORT = {'長期與中期趨勢都向上': '多頭', '長期趨勢向上，但近期轉弱': '高檔轉弱',
               '長期趨勢偏弱，近期有反彈': '弱勢反彈', '長期與中期趨勢都偏弱': '空頭'}


def series(t):
    return M.close_all[t].dropna()


def fund_of(t):
    if not FUND:
        return None
    return FUND.get('tw', {}).get(t[:-3]) if t.endswith('.TW') else FUND.get('us', {}).get(t)


def fund_total(t):
    d = fund_of(t)
    return None if d is None else d['total']


def trend_up(t):
    """趨勢多空：收盤價在季線之上，且季線比 20 個交易日前高才算「多」。"""
    s = series(t)
    ma = s.rolling(60).mean()
    if len(s) < 90:
        return None
    return bool(s.iloc[-1] > ma.iloc[-1] and ma.iloc[-1] > ma.iloc[-21])


def prob_of(t, h='5'):
    return ((PROB or {}).get('predictions', {}).get(t, {}) or {}).get(h)


def fmt_metric(key, v):
    if v is None:
        return '—'
    return f'{v:.1%}' if fundamentals.METRICS[key][1] == 'pct' else f'{v:.2f} 倍'


tab_home, tab_stock, tab_market, tab_news, tab_model = st.tabs(['🏠 總覽', '🔎 個股', '🌐 市場', '📰 新聞', '🧪 模型'])

# ═════════════════════════ 總覽 ═════════════════════════
with tab_home:
    cards = []
    for t, nm in (('^TWII', '加權指數'), ('^SOX', '費城半導體（美股）'), ('^GSPC', 'S&P 500')):
        if t in M.close_all.columns:
            s_ = series(t)
            cards.append(kpi(nm, f'{s_.iloc[-1]:,.0f}', chg_html(s_.iloc[-1] / s_.iloc[-2] - 1), f'資料至 {s_.index[-1]:%m/%d}'))
    fx = M.usdtwd
    cards.append(kpi('美元兌新台幣', f'{fx.iloc[-1]:.2f}', chg_html(fx.iloc[-1] / fx.iloc[-2] - 1), '數字變大＝台幣貶值'))
    prem = M.adr_premium()
    if '台積電' in prem.columns:
        p = prem['台積電'].dropna()
        cards.append(kpi('台積電 ADR 溢價', f'{p.iloc[-1]:+.1%}',
                         f'<span style="color:{FLAT}">近一年平均 {p.tail(core.TRADING_DAYS).mean():+.1%}</span>', 'ADR 比台股貴多少'))
    st.markdown('<div class="kgrid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)

    st.markdown('#### 關注清單')
    rows = []
    for t in ORDER:
        s = series(t)
        if len(s) < 3:
            continue
        rows.append({
            '標的': core.label(t), '市場': '美股' if M.is_us(t) else '台股', '收盤': float(s.iloc[-1]),
            '日漲跌': float(s.iloc[-1] / s.iloc[-2] - 1) * 100, '近 60 日': s.tail(60).tolist(),
            '趨勢': TREND_SHORT.get(core.trend_word(s), '—'), '趨勢強度': ((TR or {}).get('current', {}).get(t) or {}).get('z60'),
            'RSI': float(core.rsi(s).iloc[-1]),
            '體質分': fund_total(t)})
    tbl = pd.DataFrame(rows)

    def _color(v):
        return f'color: {color_of(v)}; font-weight: 600' if pd.notna(v) else ''

    sty = tbl.style.map(_color, subset=['日漲跌'])
    st.dataframe(sty, hide_index=True, width='stretch', height=min(40 + 36 * len(tbl), 520), column_config={
        '收盤': st.column_config.NumberColumn(format='%,.2f'),
        '日漲跌': st.column_config.NumberColumn(format='%+.2f%%'),
        '近 60 日': st.column_config.LineChartColumn(width='small'),
        '趨勢強度': st.column_config.NumberColumn(format='%+.1f', help='近 60 日漲跌相當於平常波動的幾倍：±0.5 以內算盤整，超過 ±1.5 算強勢。描述現況，不是預測'),
        'RSI': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%d', help='高於 70 偏熱、低於 30 偏冷'),
        '體質分': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%d', help='公司體質綜合分數（個股分頁有細項）'),
    })
    st.caption('漲跌機率模型在歷史驗證中沒有勝過基準，所以不放在總覽；想看請到「🔎 個股 → 漲跌機率」，並先讀「🧪 模型」的說明。')

    left, right = st.columns([3, 2])
    with left:
        st.markdown('#### 體質好壞 × 趨勢多空')
        st.caption('體質：公司體質綜合分 55 分以上算好。趨勢：收盤價在季線之上、且季線比 20 個交易日前高算「多」。描述現況，不是買賣訊號。')
        cells = {(True, True): [], (True, False): [], (False, True): [], (False, False): []}
        nodata = []
        for tt in ORDER:
            if tt.startswith('^'):
                continue
            ft, tr = fund_total(tt), trend_up(tt)
            if ft is None or tr is None:
                nodata.append(core.label(tt))
            else:
                cells[(ft >= 55, tr)].append(f'{core.NAMES.get(tt, tt)} {ft:.0f}')
        desc = {(True, True): ('體質好 ＋ 趨勢多', '基本面與走勢都偏正面'),
                (True, False): ('體質好 ＋ 趨勢空', '體質不差但股價走弱，值得查原因'),
                (False, True): ('體質普通 ＋ 趨勢多', '股價強但體質普通，波動風險較高'),
                (False, False): ('體質普通 ＋ 趨勢空', '基本面與走勢都偏弱')}
        for row in ((True, True), (True, False)), ((False, True), (False, False)):
            for col, key in zip(st.columns(2), row):
                with col.container(border=True):
                    st.markdown(f'**{desc[key][0]}**')
                    st.caption(desc[key][1])
                    st.markdown('、'.join(cells[key]) if cells[key] else '（目前沒有）')
        if nodata:
            st.caption('無體質或趨勢資料（ETF、資料不足）：' + '、'.join(nodata))
    with right:
        st.markdown('#### 今日摘要')
        with st.container(border=True, height=560):
            st.markdown(SUMMARY, unsafe_allow_html=True)

# ═════════════════════════ 個股 ═════════════════════════
with tab_stock:
    c1, c2 = st.columns([2, 1])
    labels = {core.label(t): t for t in ORDER}
    pick = c1.selectbox('選擇標的', list(labels), key='stock_pick')
    other = c2.text_input('或輸入其他代號', placeholder='台股 2603、美股 TSLA', key='stock_other')
    live = None
    if other.strip():
        name, s_live, err = lookup(other, M)
        if err:
            st.error(err)
        else:
            live = (name, s_live)
    if live:
        t, name, s = None, live[0], live[1]
    else:
        t = labels[pick]
        name, s = core.label(t), series(t)

    last, ch = s.iloc[-1], s.iloc[-1] / s.iloc[-2] - 1
    pills = [f"<span class='pill'>{TREND_SHORT.get(core.trend_word(s), '趨勢：資料不足')}</span>",
             f"<span class='pill'>RSI {core.rsi(s).iloc[-1]:.0f}</span>"]
    if t:
        fd, pr = fund_of(t), prob_of(t, '5')
        if fd:
            pills.append(f"<span class='pill'>體質 {fd['total']:.0f} 分（{fd['grade']}）</span>")
        if pr:
            pills.append(f"<span class='pill'>1 週上漲機率 {pr['prob']:.0%}（參考）</span>")
    cur_tr = ((TR or {}).get('current', {}).get(t) or {}) if t else {}
    if cur_tr.get('z60') is not None:
        pills.append(f"<span class='pill'>趨勢強度 {cur_tr['z60']:+.1f}（{cur_tr['word']}）</span>")
    if t and VOL and VOL.get('predictions', {}).get(t, {}).get('5'):
        pills.append(f"<span class='pill'>波動{volmodel.level_word(VOL['predictions'][t]['5']['pct'])}</span>")
    st.markdown(f"<div class='hero'><div class='n'>{name}</div><div class='p'>{last:,.2f} {chg_html(ch)}</div>"
                f"<div>{''.join(pills)}</div><div class='note'>資料至 {s.index[-1]:%Y-%m-%d}</div></div>", unsafe_allow_html=True)

    t_chart, t_vol, t_fund, t_prob, t_news = st.tabs(['走勢', '波動預測', '公司體質', '漲跌機率', '相關新聞'])

    with t_chart:
        rng = st.segmented_control('期間', ['3 個月', '6 個月', '1 年', '全部'], default='1 年', key='rng')
        days = {'3 個月': 63, '6 個月': 126, '1 年': 252, '全部': None}[rng or '1 年']
        st.markdown(chr(10).join(core.describe_stock(s, name).split(chr(10))[2:]))      # 標題與價格已在上方卡片
        if cur_tr.get('z60') is not None and cur_tr.get('ret60') is not None:
            st.caption(f"趨勢強度：近 60 個交易日漲跌 {cur_tr['ret60']:+.1%}，相當於這檔平常波動的 {cur_tr['z60']:+.1f} 倍 → {cur_tr['word']}。"
                       '這是把漲跌幅除以波動後的尺度，方便不同股票互相比較；它描述現況，實測顯示對之後的漲跌沒有可靠的預示力（見「🌐 市場 → 趨勢規則實測」）。')
        show(price_chart(s, name, days))
        vd = UNI.get('volume') if UNI else None
        if t and vd is not None and t in vd.columns and not t.startswith('^'):
            vv = vd[t].dropna().tail(120)
            if len(vv) > 60:
                vr = vd[t].dropna().tail(20).mean() / vd[t].dropna().tail(60).mean()
                fv = go.Figure(go.Bar(x=vv.index, y=vv, marker_color='#8a8f98'))
                fv.add_trace(go.Scatter(x=vv.index, y=vv.rolling(20).mean(), name='20 日均量', line=dict(color='#f5a623', width=1.6)))
                fv.update_layout(title='成交量（近 120 個交易日）', height=240, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                show(fv)
                st.caption(f"量能比（近 20 日均量 ÷ 近 60 日均量）：**{vr:.2f}**。大於 1 代表最近交易比較熱絡，小於 1 代表比較冷清；它描述關注度，不預測漲跌。")
        with st.expander('名詞說明'):
            st.markdown(core.GLOSSARY, unsafe_allow_html=True)

    with t_vol:
        vp = (VOL or {}).get('predictions', {}).get(t) if t else None
        if t is None:
            st.info('其他代號只提供走勢；波動預測僅限關注清單內的標的。')
        elif not vp:
            st.info('尚無這檔的波動預測。')
        else:
            mk = next(iter(vp.values()))['market']
            st.caption('這裡預測的是「接下來漲跌的幅度大概多大」，**不是**會漲還是會跌。用途是估風險：波動越大，同樣金額可能的賺賠都越大。')
            cols = st.columns(len(vp))
            for col, (h, p) in zip(cols, sorted(vp.items(), key=lambda kv: int(kv[0]))):
                v_ = ((VOL.get('validation') or {}).get(mk) or {}).get(h) or {}
                with col.container(border=True):
                    st.markdown(f"**未來 {VOL['horizons'][h]}**（{h} 個交易日）的漲跌，大多落在")
                    st.markdown(f"<div style='font-size:1.8rem;font-weight:800;line-height:1.3'>±{p['range']:.1%}</div>"
                                f"<div class='note'>約 {p['close'] * (1 - p['range']):,.2f} ～ {p['close'] * (1 + p['range']):,.2f}</div>"
                                f"<span class='pill'>波動{volmodel.level_word(p['pct'])}</span>", unsafe_allow_html=True)
                    if v_:
                        st.caption(f"歷史驗證中，實際漲跌有 {v_['cover1']:.0%} 落在這個範圍內、{v_['cover2']:.0%} 落在 2 倍範圍內。")
                        if v_.get('beats_baseline'):
                            st.caption(f"✅ 通過嚴格驗證：比簡單公式準 {v_['improve']:.0%}")
                        elif v_.get('has_skill'):
                            st.caption(f"◯ 有預測力（比只猜長期平均準 {v_['skill_vs_mean']:.0%}），但未證明優於簡單公式")
                        else:
                            st.caption('△ 預測力未獲證明，僅供參考')
            p5 = vp.get('5') or next(iter(vp.values()))
            lvl = '' if p5['pct'] is None else f"預測的波動高於這檔近三年 {p5['pct']:.0%} 的時候（{volmodel.level_word(p5['pct'])}）。"
            st.markdown(f"資料截至 **{p5['as_of']}** 收盤。近 5 日每天平均震 **{p5['recent5']:.1%}**、近一個月 **{p5['recent22']:.1%}**；"
                        f"模型預估未來 1 週每天約 **{p5['daily']:.1%}**。{lvl}")
            lr_ = np.log(s).diff()
            hv = (np.sqrt((lr_ ** 2).rolling(22).mean()) * np.sqrt(core.TRADING_DAYS)).dropna().tail(500)
            fig = go.Figure(go.Scatter(x=hv.index, y=hv, name='過去 22 日實際波動（年化）', line=dict(color='#3b6fe0', width=1.6)))
            p20 = vp.get('20') or p5
            fig.add_hline(y=p20['annual'], line_dash='dash', line_color='#f5a623',
                          annotation_text=f"模型預估未來 4 週 {p20['annual']:.0%}", annotation_position='bottom right')
            fig.update_layout(title='波動的變化（年化，數字越大代表漲跌越劇烈）', height=340, yaxis_tickformat='.0%',
                              margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
            show(fig)
            with st.expander('怎麼讀？限制是什麼？'):
                st.markdown(chr(10).join([
                    '- 「±3.9%」的意思是：依模型估計，這段期間結束時的漲跌幅，大多數情況會落在正負這個數字以內。實際落在範圍內的比例寫在每張卡片上（不是保證）。',
                    '- 股市偶爾會出現遠超過範圍的大漲大跌（例如重大消息），模型抓不到這種突發事件。',
                    '- 波動有延續性：最近震得大，接下來通常也震得大，這是模型主要的依據。它完全不判斷方向。',
                    '- 只用每日收盤價，沒有盤中高低點，所以對當天盤中的劇烈震盪不敏感。',
                    '- 模型好壞的驗證與上線後的真實成績，請看「🧪 模型」。']))

    with t_fund:
        d = fund_of(t) if t else None
        if t is None:
            st.info('其他代號只提供走勢；公司體質僅限關注清單內的個股。')
        elif d is None:
            st.info('這檔是指數或 ETF，沒有公司財報，因此沒有體質評分。')
        else:
            st.caption(f"更新：{FUND['updated_at']}　｜　台股：{FUND.get('tw_period', '—')}；美股：yfinance 最新一季　｜　"
                       '評分用固定門檻換算（0 最差、100 最好），僅供研究。')
            a, b = st.columns([1, 2])
            a.metric('體質綜合分數', f"{d['total']:.0f} / 100", d['grade'], delta_color='off')
            b.markdown(d['verdict'])
            cats = [k for k in fundamentals.DIMENSIONS if d['scores'].get(k) is not None]
            vals = [d['scores'][k] for k in cats]
            left, right = st.columns(2)
            if cats:
                fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill='toself', name=name,
                                                line=dict(color='#3b6fe0')))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=360, showlegend=False,
                                  margin=dict(l=30, r=30, t=30, b=30))
                left.plotly_chart(fig, width='stretch')
            with right:
                for dim in fundamentals.DIMENSIONS:
                    sc = d['scores'].get(dim)
                    st.markdown(f'**{dim}**　{"資料不足" if sc is None else f"{sc:.0f} 分"}')
                    st.progress(0 if sc is None else int(sc))
                    mrows = [{'指標': fundamentals.METRICS[k][0], '數值': fmt_metric(k, d['metrics'].get(k)),
                              '意思': fundamentals.METRICS[k][2]}
                             for k, _, _ in fundamentals.SCORING[dim] if k in d['metrics']]
                    if mrows:
                        with st.expander('看計算依據'):
                            st.dataframe(pd.DataFrame(mrows), hide_index=True, width='stretch')
            st.markdown('##### 同業比較')
            st.caption(f"比較對象：{d['peer_label']}。「同業排名」100 = 同業最好、0 = 最差（負債比、本益比等越低越好的指標已反向計算）。")
            prow = [{'指標': fundamentals.METRICS[r['key']][0], '本檔': fmt_metric(r['key'], r['mine']),
                     '同業中位數': fmt_metric(r['key'], r['median']),
                     '同業排名': '樣本太少' if r['rank'] is None else f"{r['rank']}（共 {r['n']} 家）"} for r in d['peers']]
            st.dataframe(pd.DataFrame(prow), hide_index=True, width='stretch')
            with st.expander('評分怎麼算？限制是什麼？'):
                st.markdown(chr(10).join([
                    '- 每個指標依固定門檻換算成 0～100 分（例如營業利益率 0% 得 0 分、25% 以上得 100 分），面向分數是該面向指標的平均，綜合分數是五個面向的平均。',
                    '- 門檻不分產業：代工、零售本來毛利低，所以請搭配同業比較看。',
                    '- 「股東回報」只看現金殖利率，成長股常把錢拿去再投資或買回庫藏股，分數偏低不代表不好。',
                    '- 估值越便宜分數越高，但高成長公司本益比本來就高；「成長強、估值貴」要留意預期落空的風險。',
                    '- 台股損益是年初累計數，ROE 以累計淨利年化估算；獲利年增率台股暫缺（證交所沒有去年同期）。',
                    '- 美股 ADR 若財報幣別與股價不同，不計股價淨值比。美股同業只有關注清單內的幾檔。']))

    with t_prob:
        preds = (PROB or {}).get('predictions', {}).get(t) if t else None
        if t is None:
            st.info('其他代號只提供走勢；漲跌機率僅限關注清單內的標的。')
        elif not preds:
            st.info('尚無這檔的模型結果。')
        else:
            st.warning('模型在按時間切分的歷史驗證中**沒有勝過**「永遠猜漲」基準，下面的機率只是貼近歷史上漲比例的參考，不要當進出依據。')
            cols = st.columns(len(preds))
            for col, (h, p) in zip(cols, sorted(preds.items(), key=lambda kv: int(kv[0]))):
                col.metric(f"未來 {PROB['horizons'][h]}（{h} 個交易日）上漲機率", f"{p['prob']:.0%}",
                           f"歷史上平均 {p['base']:.0%} 會漲", delta_color='off')
            any_p = next(iter(preds.values()))
            st.caption(f"資料截至 {any_p['as_of']} 收盤；台股預測只用嚴格早於該日的美股收盤價。")
            st.markdown('**這次機率主要受哪些因素影響**（以 1 週預測為例）')
            for f in (preds.get('5') or any_p)['factors']:
                st.markdown(f"- {f['name']}：{f['value']} → **{f['effect']}** 上漲機率")

    with t_news:
        code = t.split('.')[0] if t else re.sub(r'\.(TW|TWO)$', '', (live[0] if live else '')).upper()
        if not code:
            st.info('沒有相關新聞。')
        else:
            hit = NEWS[NEWS['ticker'].str.split(',').map(lambda cs: code in cs)] if len(NEWS) else NEWS
            if hit.empty:
                st.info(f'新聞庫裡目前沒有 {code} 的相關新聞（累積中）。')
            else:
                for r in hit.head(15).itertuples():
                    when = r.time.strftime('%m-%d %H:%M') if pd.notna(r.time) else ''
                    st.markdown(f"- [{r.title}]({r.url})　<span class='note'>{r.source}・{when}</span>", unsafe_allow_html=True)

# ═════════════════════════ 市場 ═════════════════════════
with tab_market:
    t_perf, t_link, t_adr, t_rule, t_rel = st.tabs(['績效比較', '美股 → 台股連動', 'ADR 溢價', '趨勢規則實測', '相對強弱'])

    with t_perf:
        options = M.sel_us + M.sel_tw
        c1, c2 = st.columns([3, 1])
        picks = c1.multiselect('標的', options, default=options, format_func=core.label)
        currency = c2.radio('計價幣別', ['新台幣', '美元', '各自原幣'])
        if picks:
            px_df = M.normalized(picks, currency)
            nm = px_df / px_df.iloc[0] * 100
            fig = go.Figure()
            for tt in nm.columns:
                fig.add_trace(go.Scatter(x=nm.index, y=nm[tt], name=core.label(tt),
                                         line=dict(dash='solid' if M.is_us(tt) else 'dot')))
            fig.update_layout(title=f'累積績效（起點 = 100，{currency}；實線美股、虛線台股）', yaxis_type='log',
                              hovermode='x unified', height=480, margin=dict(l=10, r=10))
            show(fig)
            stats = core.perf_stats(px_df).sort_values('夏普值', ascending=False)
            for col in ['年化報酬', '年化波動', '最大回撤', '期間總報酬']:
                stats[col] = stats[col].map('{:.1%}'.format)
            stats['夏普值'] = stats['夏普值'].map('{:.2f}'.format)
            st.dataframe(stats, width='stretch')
            if len(picks) >= 2:
                corr = px_df.resample('W-FRI').last().pct_change().dropna(how='all').corr()
                corr.index = corr.columns = [core.NAMES.get(tt, tt) for tt in corr.columns]
                fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                                aspect='auto', title='週報酬相關係數（越接近 1，越常一起漲跌）', height=480)
                fig.update_layout(margin=dict(l=10, r=10))
                show(fig)

    with t_link:
        c1, c2 = st.columns(2)
        us = c1.selectbox('美股指標', M.sel_us, index=M.sel_us.index(core.US_LEAD) if core.US_LEAD in M.sel_us else 0,
                          format_func=core.label)
        tw = c2.selectbox('台股標的', M.sel_tw, index=M.sel_tw.index(core.TW_TARGET) if core.TW_TARGET in M.sel_tw else 0,
                          format_func=core.label)
        pair, bucket = M.leadlag(us, tw)
        k, b = np.polyfit(pair['us'], pair['tw'], 1)
        st.markdown(f'樣本 {len(pair)} 天，相關係數 **{pair.us.corr(pair.tw):.2f}**；美股每漲 1%，台股隔日平均變動 '
                    f'**{k:.2f}%**；方向一致比例 **{(np.sign(pair.us) == np.sign(pair.tw)).mean():.0%}**')
        fig = px.scatter(pair, x='us', y='tw', opacity=0.4, height=440, labels={'us': '美股隔夜報酬', 'tw': '台股隔日報酬'})
        xs = np.array([pair.us.min(), pair.us.max()])
        fig.add_trace(go.Scatter(x=xs, y=b + k * xs, mode='lines', name='迴歸線'))
        fig.update_layout(xaxis_tickformat='.0%', yaxis_tickformat='.0%', margin=dict(l=10, r=10))
        show(fig)
        st.dataframe(pd.DataFrame({'美股隔夜': bucket.index.astype(str), '天數': bucket['天數'].values,
                                   '台股平均報酬': bucket['台股平均報酬'].map('{:+.2%}'.format).values,
                                   '台股上漲機率': bucket['台股上漲機率'].map('{:.0%}'.format).values}),
                     hide_index=True, width='stretch')
        st.caption('台股的隔日報酬是收盤對收盤，其中大部分通常在開盤跳空時就已反映。')

    with t_adr:
        fig = px.line(prem, height=460, title='ADR 相對台股的溢價（正）／折價（負）')
        fig.add_hline(y=0, line_dash='dash', line_color='gray')
        fig.update_layout(yaxis_tickformat='.0%', hovermode='x unified', legend_title='', margin=dict(l=10, r=10))
        show(fig)
        st.dataframe(pd.DataFrame({'最新': prem.ffill().iloc[-1], '近一年平均': prem.tail(core.TRADING_DAYS).mean(),
                                   '期間最高': prem.max(), '期間最低': prem.min()}).map('{:+.1%}'.format), width='stretch')

    with t_rule:
        if not TR or not TR.get('backtest'):
            st.info('尚無趨勢實測結果。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
        else:
            st.caption('這裡不預測漲跌，而是檢驗「跟著趨勢走」的常見規則，過去實際表現如何。規則和評判標準都在看結果前固定，沒有調參數。')
            mk = 'tw' if st.radio('市場', ['台股', '美股'], horizontal=True, key='rule_mk') == '台股' else 'us'
            bt = TR['backtest'].get(mk)
            if not bt:
                st.info('這個市場的資料不足。')
            else:
                P = TR['primary']
                tp, pf = bt['tests'][P], bt['portfolio']
                st.markdown(f"##### 主要規則：{TR['rules'][P]}（{bt['from']} ～ {bt['to']}，{bt['n_tickers']} 檔等權重）")
                with st.container(border=True):
                    st.markdown(('✅' if tp['q1_less_drawdown'] else '❌') + f" **大跌時少賠？** 組合最大回撤 {pf['bh']['mdd']:.0%} → {pf[P]['mdd']:.0%}，"
                                f"{tp['mdd_better_count']}/{bt['n_tickers']} 檔標的回撤較小。")
                    if tp['q3_better_return']:
                        q3 = f"✅ **賺得比較多？** 每年多約 {tp['ret_diff']:.1%}。"
                    elif tp['ret_ci'][1] is not None and tp['ret_ci'][1] < 0:
                        q3 = (f"❌ **賺得比較多？** 沒有，而且明顯比較少：年化報酬 {pf['bh']['cagr']:.1%} → {pf[P]['cagr']:.1%}，"
                              f"每年少約 {-tp['ret_diff']:.1%}（95% 信賴區間 {-tp['ret_ci'][1]:.1%}～{-tp['ret_ci'][0]:.1%}）。")
                    else:
                        q3 = (f"◯ **賺得比較多？** 沒有證據：年化報酬 {pf['bh']['cagr']:.1%} → {pf[P]['cagr']:.1%}，差距的信賴區間含 0。")
                    st.markdown(q3)
                    st.markdown(('✅' if tp['q2_better_sharpe'] else '◯') + f" **承擔每一分風險換到的報酬比較高？**（夏普值）{pf['bh']['sharpe']:.2f} → {pf[P]['sharpe']:.2f}，"
                                + ('明顯較高。' if tp['q2_better_sharpe'] else '沒有證據顯示較好（差距的信賴區間含 0）。'))
                    st.caption(f"這個規則平均只有 {tp['in_market']:.0%} 的時間持有、每年進出約 {tp['switches_per_year']:.0f} 次。"
                               '白話結論：這段期間照趨勢規則做，少賠一些，但錯過的漲幅更多。')

                st.dataframe(pd.DataFrame(
                    [{'做法': '一直持有', '年化報酬': f"{pf['bh']['cagr']:.1%}", '最大回撤': f"{pf['bh']['mdd']:.1%}", '夏普值': f"{pf['bh']['sharpe']:.2f}",
                      '年化波動': f"{pf['bh']['vol']:.1%}", '持有時間': '100%', '每年進出': '0', '回撤較小的標的': '—'}] +
                    [{'做法': TR['rules'][k] + ('（主要）' if k == P else ''), '年化報酬': f"{pf[k]['cagr']:.1%}", '最大回撤': f"{pf[k]['mdd']:.1%}",
                      '夏普值': f"{pf[k]['sharpe']:.2f}", '年化波動': f"{pf[k]['vol']:.1%}", '持有時間': f"{bt['tests'][k]['in_market']:.0%}",
                      '每年進出': f"{bt['tests'][k]['switches_per_year']:.0f}", '回撤較小的標的': f"{bt['tests'][k]['mdd_better_count']}/{bt['n_tickers']}"}
                     for k in TR['rules']]), hide_index=True, width='stretch')
                st.caption('其餘三個規則只當參考：同時看好幾個規則，容易碰巧有一個看起來不錯。')

                eq = bt['equity']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=eq['dates'], y=eq['bh'], name='一直持有', line=dict(width=2.4, color='#3b6fe0')))
                for k, col in zip(TR['rules'], ('#e5484d', '#f5a623', '#9b59d0', '#2f9e6b')):
                    fig.add_trace(go.Scatter(x=eq['dates'], y=eq[k], name=TR['rules'][k], line=dict(width=1.3, color=col)))
                fig.update_layout(title='等權重組合的累積淨值（起點 = 1，已扣交易成本）', yaxis_type='log', height=420, hovermode='x unified',
                                  margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation='h', y=-0.15))
                show(fig)

                with st.expander('各標的明細（主要規則）'):
                    st.dataframe(pd.DataFrame([{
                        '標的': core.label(t), '起算日': v['from'], '一直持有 年化': f"{v['bh']['cagr']:.1%}", '規則 年化': f"{v['rules'][P]['cagr']:.1%}",
                        '一直持有 最大回撤': f"{v['bh']['mdd']:.1%}", '規則 最大回撤': f"{v['rules'][P]['mdd']:.1%}",
                        '持有時間': f"{v['rules'][P]['in_market']:.0%}", '進出次數': v['rules'][P]['switches']} for t, v in bt['per_ticker'].items()]),
                        hide_index=True, width='stretch')

                stt = (TR.get('states') or {}).get(mk)
                if stt:
                    st.markdown(f"##### 「趨勢多」和「趨勢空」之後 {TR['fwd']} 個交易日，實際發生什麼")
                    rci, vci = stt['ret_diff_ci'], stt['vol_diff_ci']
                    sig_r = rci[0] is not None and (rci[0] > 0 or rci[1] < 0)
                    st.caption('趨勢多 = 收盤價在季線之上且季線上揚（與總覽的定義相同）。'
                               + ('兩種狀態之後的報酬有明顯差別。' if sig_r else '兩種狀態之後的平均報酬**沒有明顯差別**（差距的信賴區間含 0）：趨勢標籤描述的是現況，對之後漲跌沒有可靠的預示力。'))
                    st.dataframe(pd.DataFrame([{'當時狀態': nm, '樣本數': g['n'], f"之後 {TR['fwd']} 日平均報酬": f"{g['ret']:+.1%}",
                                                '之後上漲比例': f"{g['up_rate']:.0%}", '之後的波動（年化）': f"{g['vol']:.0%}"}
                                               for nm, g in (('趨勢多', stt['up']), ('趨勢空', stt['down']))]), hide_index=True, width='stretch')
                    if stt.get('buckets'):
                        st.markdown('依「趨勢強度」分組')
                        st.dataframe(pd.DataFrame([{'趨勢強度': b['name'], '樣本數': b['n'], f"之後 {TR['fwd']} 日平均報酬": f"{b['ret']:+.1%}",
                                                    '之後上漲比例': f"{b['up_rate']:.0%}", '之後的波動（年化）': f"{b['vol']:.0%}"} for b in stt['buckets']]),
                                     hide_index=True, width='stretch')
                        st.caption('如果趨勢強度能預示之後的漲跌，報酬應該由上到下有規律地變化；沒有規律就代表它只適合描述現況。')

                with st.expander('怎麼做的？限制是什麼？'):
                    st.markdown(chr(10).join([
                        '- 第 t 日收盤後依規則決定，第 t+1 日才生效，不會用到當天還不知道的資訊。不持有時當作現金、報酬為 0（沒算利息）。',
                        f"- 交易成本：每次進或出扣 {bt['cost']:.2%}（台股含手續費與證交稅的概估；美股以低成本估計）。",
                        '- 每檔標的都從年線可計算的那天起算；指數不能直接買賣，這裡當作追蹤它的 ETF 來看。',
                        '- 信賴區間用區塊拔靴法（考慮相鄰日期的相關性）。最大回撤只有一條歷史路徑，所以只能描述、無法檢定。',
                        '- **最大的限制是期間**：2021 年底到現在，雖然包含 2022 年的下跌，但整體是大多頭。趨勢規則的價值主要出現在長時間的空頭（例如 2000、2008 年那種），這段資料檢驗不到。',
                        '- 所以這個結果的正確讀法是：「在這五年，照趨勢規則做並沒有比較好」，而不是「趨勢規則永遠沒用」。']))

    with t_rel:
        if not REL or not REL.get('markets'):
            st.info('尚無相對強弱結果。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
        else:
            st.caption('這裡不問「會不會漲」，而是問「在一群股票裡，過去相對強的，之後一個月是不是也相對強？」。規則與判準在看結果前固定，沒有調參數。')
            mkr = 'tw' if st.radio('市場', ['台股', '美股'], horizontal=True, key='rel_mk') == '台股' else 'us'
            R_ = REL['markets'].get(mkr)
            if not R_:
                st.info('這個市場的資料不足。')
            else:
                P_ = R_['results'].get(REL['primary'])
                bias = R_.get('bias')
                if P_ and P_['has_evidence']:
                    st.success(f"**有證據：** 過去 12 個月（跳過最近 1 個月）表現較強的股票，之後一個月傾向也較強（IC {P_['ic_mean']:+.3f}，95% 區間不含 0）。但名單有存活者偏誤，請打折看。")
                elif P_:
                    st.warning(f"**沒有足夠證據：** 動能排名與之後一個月報酬的相關（IC）平均 {P_['ic_mean']:+.3f}，方向是正的，"
                               f"{P_['ic_pos_rate']:.0%} 的月份為正，但 95% 信賴區間 {P_['ic_ci'][0]:+.3f}～{P_['ic_ci'][1]:+.3f} 含 0，不能排除是運氣。")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('股票數', f"{R_['n_stocks']} 檔")
                if P_:
                    c2.metric('驗證月份', f"{P_['months']} 個月", f"{P_['from']} ～ {P_['to']}", delta_color='off')
                    c3.metric('前 20% 每月平均', f"{P_['top_mean']:+.2%}", f"後 20% {P_['bottom_mean']:+.2%}", delta_color='off')
                    c4.metric('平均每月換手', f"{P_['turnover']:.0%}", '前 20% 名單的更換比例', delta_color='off')
                if bias:
                    with st.container(border=True):
                        st.markdown('**⚠️ 存活者偏誤：先看這個**')
                        st.markdown(f"名單是「現在的」大型股。同一段期間，名單等權重年化報酬 **{bias['universe']:.1%}**，大盤指數 **{bias['index']:.1%}**，"
                                    f"名單多賺 **{bias['gap']:.1%}**。這個差距的一部分來自「現在是大型股的公司，過去多半表現好」，所以「強者恆強」會被高估，下面的數字都要往保守的方向修正。")
                if P_:
                    nm_ = REL['signals']
                    st.markdown('##### 各訊號的結果')
                    st.caption('IC = 每月「分數高低」與「之後報酬高低」的排名相關，0 = 無關，越高越好。除主要訊號外，其餘只當參考（同時看多個，容易碰巧有一個好看）。')
                    st.dataframe(pd.DataFrame([{
                        '訊號': nm_[k] + ('（判定用）' if k == REL['primary'] else ''), '月數': r['months'], '平均 IC': f"{r['ic_mean']:+.3f}",
                        'IC 的 95% 區間': f"{r['ic_ci'][0]:+.3f} ～ {r['ic_ci'][1]:+.3f}", 'IC 為正的月份': f"{r['ic_pos_rate']:.0%}",
                        '前 20% − 後 20%（每月）': f"{r['spread_mean']:+.2%}", '前 20% − 全體（每月）': f"{r['excess_mean']:+.2%}",
                        '換手': f"{r['turnover']:.0%}", '判定': '有證據' if r['has_evidence'] else '沒有證據'} for k, r in R_['results'].items()]),
                        hide_index=True, width='stretch')
                cv = R_.get('curve')
                if cv:
                    fig = go.Figure()
                    for key, nm, col in (('top', '動能最強的 20%', '#e5484d'), ('all', '全體平均', '#8a8f98'), ('bottom', '動能最弱的 20%', '#2f9e6b')):
                        fig.add_trace(go.Scatter(x=cv['dates'], y=cv[key], name=nm, line=dict(color=col, width=2)))
                    fig.update_layout(title='每月依動能排名持有的累積淨值（起點 = 1，未扣交易成本）', height=380, hovermode='x unified',
                                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation='h', y=-0.15))
                    show(fig)
                    st.caption('這張圖的差距包含存活者偏誤與未扣成本；換手越高，成本影響越大。')
                cur = R_['current']
                st.markdown(f"##### 目前的排名（資料至 {cur['as_of']}）")
                st.caption('依 12-1 動能由強到弱。第 1 組 = 最強的 20%。這是描述「過去誰比較強」，不是推薦。')
                st.dataframe(pd.DataFrame([{
                    '排名': r['rank'], '標的': core.label(r['ticker']) if r['ticker'] in core.NAMES else f"{r['name']} ({r['ticker']})",
                    '組別': f"第 {r['quintile']} 組", '12-1 動能': r['score'] * 100, '近 1 月': None if r['ret1m'] is None else r['ret1m'] * 100,
                    '近 3 月': None if r['ret3m'] is None else r['ret3m'] * 100, '趨勢強度': r['z60'], '量能比': r['vol_ratio']} for r in cur['rows']]),
                    hide_index=True, width='stretch', height=520, column_config={
                        '12-1 動能': st.column_config.NumberColumn(format='%+.0f%%'), '近 1 月': st.column_config.NumberColumn(format='%+.1f%%'),
                        '近 3 月': st.column_config.NumberColumn(format='%+.1f%%'),
                        '趨勢強度': st.column_config.NumberColumn(format='%+.1f', help='近 60 日漲跌相當於平常波動的幾倍'),
                        '量能比': st.column_config.NumberColumn(format='%.2f', help='近 20 日平均成交量 ÷ 近 60 日平均成交量；大於 1 代表最近比較熱絡')})
                with st.expander('怎麼做的？限制是什麼？'):
                    st.markdown(chr(10).join([
                        '- 訊號：12-1 動能 = 前 12 個月報酬，跳過最近 1 個月（避開短期反轉），是學術文獻中最標準的定義。',
                        '- 每個月最後一個交易日排名，答案是下一個月底的報酬；只用當天已知的資料。各市場分開排名。',
                        '- 判準（事先訂好）：每月 IC 的平均大於 0，且以月為單位的區塊拔靴法 95% 區間下界大於 0。月數只有約 57 個，統計檢定力有限，沒有證據不等於沒有效果。',
                        '- 台股池不含 ETF，美股池不含 ADR（避免與台股重複）。名單固定、不是即時成分股；個別代號歷史不足會被略過。',
                        '- **存活者偏誤是最大的限制**，見上方方框；因此「沒有證據」可信，「有證據」則要打折。',
                        '- 成交量已一併存下來，目前只用在「量能比」這個描述欄位，沒有放進任何預測。']))

# ═════════════════════════ 新聞 ═════════════════════════
with tab_news:
    st.caption(f"累積 {len(NEWS)} 則　｜　" + '　｜　'.join(META.get('news_status', [])))
    c1, c2 = st.columns([1, 2])
    market = c1.radio('市場', ['全部', '台股', '美股'], horizontal=True)
    kw = c2.text_input('關鍵字或代號', placeholder='例如 台積電、NVDA')
    df = NEWS if market == '全部' else NEWS[NEWS['market'] == market]
    if kw.strip():
        k = kw.strip()
        df = df[df['title'].str.contains(k, case=False, regex=False) | df['ticker'].str.contains(k, case=False, regex=False)]
    view = pd.DataFrame({'時間': df['time'].dt.strftime('%m-%d %H:%M'), '來源': df['source'],
                         '標題': df['title'], '原文': df['url'], '相關代號': df['ticker']}).head(200)
    st.dataframe(view, hide_index=True, width='stretch', height=600,
                 column_config={'原文': st.column_config.LinkColumn(display_text='開啟', width='small'),
                                '標題': st.column_config.TextColumn(width='large')})

# ═════════════════════════ 模型 ═════════════════════════
with tab_model:
    t_score, t_valid, t_sim = st.tabs(['成績單（上線後的真實成績）', '歷史驗證（回測）', '新聞前景模擬'])

    with t_score:
        st.markdown('##### 漲跌方向成績單')
        LOGDF = scorecard.evaluate(M.close_all)
        ok, n_log, bad = scorecard.verify_log()
        ok_r, n_res, bad_r = scorecard.verify_results()
        if LOGDF.empty:
            st.info('成績單還沒有紀錄。每日更新跑過一次之後，就會開始記錄每天的判斷。')
        else:
            SC = scorecard.summarize(LOGDF)
            st.caption(f"紀錄從 {LOGDF['made_at'].min():%Y-%m-%d} 開始，之後每個交易日自動新增；不回填歷史（回填就是事後諸葛）。到期後用真實收盤價對答案。")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('累計判斷', f"{SC['total']} 筆")
            c2.metric('已到期對答案', f"{SC['resolved']} 筆")
            c3.metric('還沒到期', f"{SC['pending']} 筆")
            c4.metric('紀錄完整性', '✅ 完整' if ok and ok_r else '⚠️ 異常', f'判斷 {n_log}・結果 {n_res} 筆已驗證' if ok and ok_r else '請看下方訊息', delta_color='off')
            if not ok_r:
                st.error(f'結果紀錄的防竄改雜湊鏈在第 {bad_r} 列起不一致，到期結果可能被改動或損毀。')
            if LOGDF.attrs.get('revised'):
                st.warning(f"有 {LOGDF.attrs['revised']} 筆用目前價格重算的答案，和當時記下的不同（資料源事後修正過價格）。成績一律採用當時記下的結果。")
            if not ok:
                st.error(f'紀錄檔的防竄改雜湊鏈在第 {bad} 列起不一致，代表檔案被改動過或損毀，下列成績不可信。')
            if SC['resolved'] < 30:
                st.warning(f"目前只有 {SC['resolved']} 筆到期，樣本太少，命中率的起伏大多是運氣，請不要據此下結論。1 週期的判斷最快一週後到期。")

            st.markdown('##### 命中率：模型 vs「永遠猜漲」')
            st.caption(f'命中 = 模型說漲機率 ≥ 50% 且真的漲，或說 < 50% 且真的沒漲。「近期」只看判斷日在最近 {scorecard.RECENT_DAYS} 天內的紀錄。'
                       'Brier 是機率誤差，越低越好，要低於「基準」才算有用。')

            def fmt_stats(st_):
                if not st_['n']:
                    return ['0', '—', '—', '—', '—']
                return [str(st_['n']), f"{st_['hit']:.1%}", f"{st_['always_up']:.1%}", f"{st_['brier']:.4f}", f"{st_['brier_base']:.4f}"]

            srows = []
            for r in SC['rows']:
                for scope, key in (('累計', 'all'), ('近期', 'recent')):
                    srows.append(dict(zip(['範圍', '期間', '已到期筆數', '模型命中率', '永遠猜漲命中率', 'Brier（模型）', 'Brier（基準）'],
                                          [scope, r['label']] + fmt_stats(r[key]))))
            st.dataframe(pd.DataFrame(srows), hide_index=True, width='stretch')

            res = LOGDF[LOGDF['resolved']].sort_values('resolved_at')
            if len(res) >= 5:
                g = res.groupby('resolved_at').agg(hit=('hit', 'sum'), up=('outcome', 'sum'), n=('hit', 'count')).cumsum()
                line = pd.DataFrame({'模型累計命中率': g['hit'] / g['n'], '永遠猜漲累計命中率': g['up'] / g['n']})
                fig = px.line(line, height=340, title='累計命中率（依到期日）')
                fig.update_layout(yaxis_tickformat='.0%', hovermode='x unified', legend_title='', margin=dict(l=10, r=10))
                show(fig)

            st.markdown('##### 機率校準')
            st.caption('模型說「約 X% 會漲」的那些判斷，實際漲的比例有沒有接近 X%。兩欄越接近，機率越可信。')
            if SC['calibration']:
                st.dataframe(pd.DataFrame([{'模型給的機率區間': c['range'], '筆數': c['n'], '平均預測機率': f"{c['pred']:.0%}",
                                            '實際上漲比例': f"{c['actual']:.0%}"} for c in SC['calibration']]),
                             hide_index=True, width='stretch')
            else:
                st.caption('還沒有到期的紀錄。')

            st.markdown('##### 最近對過的答案')
            if len(res):
                recent = res.sort_values('resolved_at', ascending=False).head(40)
                st.dataframe(pd.DataFrame({
                    '判斷日': recent['made_at'].dt.strftime('%Y-%m-%d'), '標的': [core.label(tt) for tt in recent['ticker']],
                    '期間(交易日)': recent['horizon'], '當時預測上漲機率': recent['prob'].map('{:.0%}'.format),
                    '到期日': recent['resolved_at'].dt.strftime('%Y-%m-%d'), '實際報酬': recent['ret'].map('{:+.1%}'.format),
                    '結果': ['✅ 猜對' if h == 1 else '❌ 猜錯' for h in recent['hit']]}), hide_index=True, width='stretch')
            else:
                st.caption('還沒有到期的紀錄。')

            with st.expander('這份成績單怎麼做的？可信在哪、限制在哪？'):
                st.markdown(chr(10).join([
                    '- 每次每日更新，把當天對每檔標的、1／2／4 週的上漲機率寫進 `data/predictions_log.csv`。檔案只在檔尾新增，每列帶 SHA-256 雜湊鏈，改動或刪除任何一列都會被「紀錄完整性」抓到；GitHub 的 commit 歷史也能回頭查證。',
                    '- 同一個（判斷日, 標的, 期間）只記第一次，重跑不會覆蓋。',
                    '- 對答案不另外存檔：用到期日的收盤價直接算。價格是還原權息後的收盤價，除權息時歷史價格可能有極小調整。',
                    '- 模型每次更新都用「截至當天所有已知結果」重新訓練（沒有偷看未來）。',
                    '- 同一天不同標的的結果高度相關、相鄰日期也重疊，實際獨立的樣本比筆數少，結論要更保守。']))

        st.divider()
        st.markdown('##### 波動預測成績單')
        (okl, nl, bl), (okr, nr, br) = volmodel.verify_logs()
        VDF, VS = volmodel.scorecard()
        if not VS['total']:
            st.info('波動預測還沒有紀錄。每日更新跑過一次之後就會開始記錄。')
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric('累計預測', f"{VS['total']} 筆")
            c2.metric('已到期對答案', f"{VS['resolved']} 筆")
            c3.metric('紀錄完整性', '✅ 完整' if okl and okr else '⚠️ 異常', f'預測 {nl}・結果 {nr} 筆已驗證' if okl and okr else '紀錄被改動或損毀', delta_color='off')
            if not (okl and okr):
                st.error('波動預測紀錄的防竄改雜湊鏈不一致，下列成績不可信。')
            if VS['resolved'] < 30:
                st.warning(f"目前只有 {VS['resolved']} 筆到期，樣本太少，請不要據此下結論。")
            st.caption('每筆預測當下同時記下簡單公式（EWMA）的預測，到期後比誰的誤差小；並統計實際漲跌落在預測範圍內的比例。')
            st.dataframe(pd.DataFrame([{
                '期間': r['label'], '已到期筆數': r['n'],
                '誤差（模型）': f"{r['err_model']:.4f}" if r['n'] else '—', '誤差（EWMA）': f"{r['err_base']:.4f}" if r['n'] else '—',
                '落在 1 倍範圍內': f"{r['cover1']:.0%}" if r['n'] else '—', '落在 2 倍範圍內': f"{r['cover2']:.0%}" if r['n'] else '—'}
                for r in VS['rows']]), hide_index=True, width='stretch')

    with t_valid:
        st.markdown('##### 波動預測模型（預測震盪幅度，不預測方向）')
        if not VOL or not VOL.get('validation'):
            st.info('尚無波動模型結果。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
        else:
            vall = [(mk, h, r) for mk in ('tw', 'us') for h, r in sorted((VOL['validation'].get(mk) or {}).items(), key=lambda kv: int(kv[0])) if r]
            n_pass = sum(1 for _, _, r in vall if r['beats_baseline'])
            n_skill = sum(1 for _, _, r in vall if r.get('has_skill'))
            st.success(f'**波動可以預測。** {len(vall)} 組驗證中，有 {n_skill} 組明顯比「只猜長期平均」準；'
                       f'其中 {n_pass} 組進一步通過較嚴格的標準：明顯比常用的簡單公式（EWMA）更準。沒通過嚴格標準的組別，模型的誤差仍低於簡單公式，只是差距還不足以排除運氣。')
            st.caption('誤差 = 預測與實際波動（取對數）的均方誤差，越低越好。「明顯」的定義：以日期為單位做區塊拔靴法，95% 信賴區間不含 0（相鄰日期的答案高度重疊，一般的檢定會高估把握）。'
                       '排序相關：預測高的時候實際是否也高，1 為完全一致、0 為無關。')
            names_ = {'har': 'HAR 線性', 'lgb': 'LightGBM'}
            st.dataframe(pd.DataFrame([{
                '市場': '台股' if mk == 'tw' else '美股', '期間': VOL['horizons'][h], '採用模型': names_[r['model']], '驗證樣本': r['n'],
                '誤差（模型）': f"{r['mse'][r['model']]:.4f}", '誤差（EWMA）': f"{r['mse']['ewma']:.4f}", '誤差（長期平均）': f"{r['mse']['mean']:.4f}",
                '比最佳簡單公式準': f"{r['improve']:+.1%}", '嚴格標準': '✅ 通過' if r['beats_baseline'] else '未通過',
                '比長期平均準': f"{r['skill_vs_mean']:+.1%}" + ('（明顯）' if r.get('has_skill') else ''),
                '排序相關': f"{r['rank_corr']:.2f}", '範圍涵蓋率（1 倍／2 倍）': f"{r['cover1']:.0%}／{r['cover2']:.0%}"} for mk, h, r in vall]),
                hide_index=True, width='stretch')
            with st.expander('這個模型怎麼做的？規則是什麼？'):
                st.markdown(chr(10).join([
                    '- 預測目標：未來 1／2／4 週的實現波動（每日報酬平方的平均再開根號）。',
                    '- 候選模型事先固定兩個：HAR（用近 1／5／22／66 日波動的線性模型，波動預測的經典做法）與 LightGBM（另加報酬、回撤、大盤與美股波動等）。每個市場選平均誤差較低的；LightGBM 沒有比較好就用較簡單的 HAR。',
                    '- 驗證按時間切分，訓練與驗證之間隔開預測天數；台股用到的美股資料一律嚴格早於當日。',
                    '- 嚴格標準（看結果前訂好）：誤差要明顯低於 EWMA 與「延續過去同樣天數的波動」兩個簡單公式中較好的那個。',
                    '- 「比長期平均準」是看到第一次結果後才補上的描述性數字，用來回答「波動到底能不能預測」，不影響嚴格標準的判定。',
                    '- 範圍涵蓋率若低於 68%／95%，代表模型給的範圍略窄（股市大漲大跌的頻率比常態分布高），網頁上顯示的是實際涵蓋率。']))
            rab = (VOL or {}).get('range_ab')
            if rab:
                st.markdown('##### 加入「每日最高／最低價」有沒有讓波動預測更準？（A/B 驗證）')
                st.caption('特徵：Parkinson 範圍波動（用每日最高與最低價估計當天震盪幅度，比只看收盤價更完整）。規則事先訂好：同一套驗證期與參數，三個期間的平均誤差較低才採用。')
                st.dataframe(pd.DataFrame([{'市場': '台股' if mk == 'tw' else '美股', '平均誤差（只用收盤價）': f"{x['mse_without']:.4f}",
                                            '平均誤差（加高低價）': f"{x['mse_with']:.4f}", '改善': f"{1 - x['mse_with'] / x['mse_without']:+.1%}",
                                            '結果': '採用' if x['used'] else '不採用'} for mk, x in rab.items()]), hide_index=True, width='stretch')
                st.caption('改善幅度不大，但兩個市場方向一致。')
        st.divider()
        st.markdown('##### 漲跌方向模型')
        if not PROB or not PROB.get('predictions'):
            st.info('尚無模型結果。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
        else:
            val = PROB['validation']
            n_ok = sum(1 for mk in val.values() for r in mk.values() if r and r['beats_baseline'])
            n_all = sum(1 for mk in val.values() for r in mk.values() if r)
            if n_ok == 0:
                st.warning('**請先看這裡：** 這個模型在「按時間切分」的歷史驗證中，**沒有勝過**最簡單的基準（永遠猜漲／永遠猜歷史上漲比例）。'
                           '意思是：只靠價量、美股隔夜、匯率，目前還抓不到可靠的規律。個股頁的機率只是貼近歷史上漲比例的參考。')
            else:
                st.info(f'{n_all} 組驗證中有 {n_ok} 組勝過基準；沒勝過的組別，機率請特別保守看待。')
            st.caption('做法：只用某個時間點「以前」的資料訓練，預測「之後」半年，再往後滾動；訓練與驗證之間隔開預測天數，避免偷看未來。'
                       'AUC 0.5 代表跟亂猜一樣、越高越好；Brier 是機率誤差，越低越好，要低於「基準」才算有用。')
            vrows = []
            for mk, mname in (('tw', '台股'), ('us', '美股')):
                for h, r in sorted(val.get(mk, {}).items(), key=lambda kv: int(kv[0])):
                    if not r:
                        continue
                    vrows.append({'市場': mname, '期間': PROB['horizons'][h], '驗證樣本': r['n'], 'AUC': f"{r['auc']:.3f}",
                                  '模型命中率': f"{r['acc']:.1%}", '永遠猜漲命中率': f"{r['acc_always_up']:.1%}",
                                  'Brier（模型）': f"{r['brier']:.4f}", 'Brier（基準）': f"{r['brier_base']:.4f}",
                                  '結論': '勝過基準' if r['beats_baseline'] else '沒勝過基準'})
            st.dataframe(pd.DataFrame(vrows), hide_index=True, width='stretch')
            ab = PROB.get('inst_ab')
            if ab:
                st.markdown('##### 台股加入「法人買賣超」有沒有幫助？（A/B 驗證）')
                st.caption(f"用 {ab['days']} 個交易日的外資、投信與全市場三大法人買賣超。規則事先訂好：同一套驗證期與參數，三個預測期間的平均 AUC 較高才採用。"
                           + ('結果：**採用**。' if ab['used'] else '結果：**不採用**。'))
                st.dataframe(pd.DataFrame([{'期間': PROB['horizons'][h], 'AUC（不含法人）': f"{x['without']['auc']:.3f}",
                                            'AUC（含法人）': f"{x['with']['auc']:.3f}",
                                            'Brier（不含）': f"{x['without']['brier']:.4f}", 'Brier（含）': f"{x['with']['brier']:.4f}",
                                            'Brier（基準）': f"{x['with']['brier_base']:.4f}"} for h, x in ab['detail'].items()]
                                           + [{'期間': '平均', 'AUC（不含法人）': f"{ab['auc_without']:.3f}", 'AUC（含法人）': f"{ab['auc_with']:.3f}",
                                               'Brier（不含）': '', 'Brier（含）': '', 'Brier（基準）': ''}]),
                             hide_index=True, width='stretch')
                st.caption('法人特徵讓 AUC 略升，但仍低於 0.5、Brier 仍不優於基準，所以整體結論不變：模型沒有勝過基準。')
            with st.expander('怎麼讀這張表？為什麼可能沒勝過基準？'):
                st.markdown(chr(10).join([
                    '- 這幾年股市大多在漲，所以「永遠猜漲」本身就有 56%～65% 的命中率，是很難超越的基準。',
                    '- 只用價量、美股隔夜、匯率，資訊量有限；短期漲跌多半是隨機的，沒勝過基準是常見且誠實的結果。',
                    '- 樣本只有約 5 年，相鄰日期的結果高度重疊，真正獨立的樣本比表上的數字少很多。',
                    '- 還沒納入法人買賣超（證交所沒有歷史資料，需要先每天累積）與新聞情緒，是後續改進方向。',
                    '- 模型採保守設定（淺樹、強正則化），並試過三種複雜度皆未勝過基準；沒有再針對驗證期反覆調參。']))

    with t_sim:
        OUT = outlook.load()
        if not OUT or not OUT.get('summary'):
            st.info('尚無新聞前景模擬結果。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
        else:
            S = OUT['summary']
            if S['beats_baseline']:
                st.success('在事先訂好的三項判準下，新聞模型勝過基準。但請先看下方的限制說明，再決定要不要參考。')
            else:
                st.warning('**請先看這裡：** 用歷史新聞做「不偷看未來」的模擬，這個新聞模型**沒有勝過**基準。'
                           '意思是：用這種簡單的文字分類方法、這份新聞量，還抓不到新聞和之後漲跌之間可靠的規律。')
            st.caption(f"更新：{OUT['updated_at']}　｜　新聞 {OUT['n_news']:,} 則（自 {OUT['news_from']}）　｜　"
                       f"預測目標：判斷日收盤後 {OUT['horizon']} 個交易日的漲跌")
            with st.container(border=True):
                st.markdown('**模擬怎麼做的（每一條都是為了不偷看未來）**')
                st.markdown(chr(10).join([
                    '1. 每個判斷日，只用「當天收盤前已發布」的新聞（台股 13:30、美股 16:00 紐約時間；收盤後的新聞算下一個交易日）。',
                    f"2. 按月滾動：預測某個月時，只用更早、且隔開 {OUT['horizon']} 日答案期間的資料訓練；預測完才公布答案。",
                    '3. 每個月重新訓練一次，一路模擬到最近，所以下面的結果等於「如果過去每天都用這個模型，會怎樣」。',
                    '4. 判準在看結果前就訂好：AUC 大於 0.53、機率誤差（Brier）優於基準、且排列檢定 p 值小於 0.05，三項都要達到。']))

            c = st.columns(4)
            c[0].metric('模擬樣本', f"{S['n']:,} 筆", f"{S['from']} ～ {S['to']}", delta_color='off')
            c[1].metric('AUC（0.5＝亂猜）', f"{S['auc']:.3f}", f"排列檢定 p = {S['perm_p']:.2f}", delta_color='off')
            c[2].metric('命中率', f"{S['acc']:.1%}", f"永遠猜漲 {S['acc_always_up']:.1%}", delta_color='off')
            c[3].metric('Brier（越低越好）', f"{S['brier']:.4f}", f"基準 {S['brier_base']:.4f}", delta_color='off')
            st.caption('排列檢定 p 值：把答案隨機打亂 300 次，看純靠運氣能得到這麼高 AUC 的機率；越小越不像運氣，一般要小於 0.05。')

            st.markdown('##### 模型「最看好」與「最不看好」的各 20%，之後實際報酬')
            sp = pd.DataFrame({'分組': ['最看好 20%', '全部', '最不看好 20%'],
                               '之後平均報酬': [f"{S['top_ret']:+.2%}", f"{S['all_ret']:+.2%}", f"{S['bottom_ret']:+.2%}"],
                               '之後上漲比例': [f"{S['top_up']:.0%}", '—', f"{S['bottom_up']:.0%}"]})
            st.dataframe(sp, hide_index=True, width='stretch')
            st.caption('如果模型有用，「最看好」應該明顯高於「最不看好」。這裡的方向若相反或差不多，就代表沒有可用的訊號。')

            fo = pd.DataFrame([f for f in S['folds'] if f['auc'] is not None])
            if len(fo):
                fig = px.bar(fo, x='month', y='auc', height=300, title='各月 AUC（0.5 為亂猜線）', labels={'month': '月份', 'auc': 'AUC'})
                fig.add_hline(y=0.5, line_dash='dash', line_color='gray')
                fig.update_yaxes(range=[0.35, 0.65])
                fig.update_layout(margin=dict(l=10, r=10))
                show(fig)

            st.markdown('##### 最近的逐筆重播：當天看到的新聞 → 模型的判斷 → 之後公布的答案')
            rp = pd.DataFrame(OUT['replay'])
            if len(rp):
                rp['hit'] = [(p >= 0.5) == u for p, u in zip(rp['prob'], rp['up'])]
                st.dataframe(pd.DataFrame({
                    '判斷日': rp['date'], '標的': [core.label(t) for t in rp['ticker']], '當天的新聞': rp['title'],
                    '模型說上漲機率': rp['prob'].map('{:.0%}'.format), f"{OUT['horizon']} 日後實際報酬": rp['ret'].map('{:+.1%}'.format),
                    '結果': ['✅ 猜對' if h else '❌ 猜錯' for h in rp['hit']]}), hide_index=True, width='stretch', height=420)

            cur = OUT.get('current') or {}
            if cur:
                with st.expander('目前各檔的「新聞前景分」（僅供參考）'):
                    st.caption('用最近 7 天的新聞算出的平均上漲機率，與歷史上平均上漲比例比較。既然模擬沒有勝過基準，這個分數不要當進出依據。')
                    st.dataframe(pd.DataFrame([{'標的': core.label(t), '新聞前景分': f"{v['score']:.0%}", '歷史平均': f"{v['base']:.0%}",
                                                '近期新聞則數': v['n_news'], '最新一則': v['latest']} for t, v in cur.items()]),
                                 hide_index=True, width='stretch')
            with st.expander('這個模型是什麼？限制在哪？下一步呢？'):
                st.markdown(chr(10).join([
                    '- 模型：把新聞標題加摘要切成 2～3 個字的片段，用 TF-IDF（詞彙重要性權重）加邏輯迴歸，學「哪些詞出現時之後比較容易漲」。沒有預訓練語言模型，不懂語意。',
                    '- 新聞來源目前只有鉅亨網，且只收與關注清單相關的新聞；同一天同一檔標的的多則新聞會合併成一筆。',
                    '- 沒勝過基準不代表新聞完全沒有資訊，只代表這個簡單方法抓不到。下一步是用同一套模擬，在 Colab 換成 BERT 類模型或加 LoRA 微調，看是否更好。',
                    '- 不論用什麼模型，都要通過同樣的模擬與判準，才算有用。']))
