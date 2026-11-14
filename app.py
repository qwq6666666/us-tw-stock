"""美股 × 台股整合分析：網頁介面（Gradio）。資料來自 data/ 資料夾，由每日排程更新。"""
import json
import os
import re

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import core

# 第一次啟動、排程還沒跑過時，先試著現場抓一次資料
if not os.path.exists(core.PRICES_CSV):
    try:
        import update
        update.main()
    except BaseException as e:
        print('初次建立資料失敗：', e)

READY = os.path.exists(core.PRICES_CSV)
M = core.load_market() if READY else None
NEWS = core.load_news()
META = json.load(open(core.META_JSON, encoding='utf-8')) if os.path.exists(core.META_JSON) else {}
SUMMARY = open(core.SUMMARY_MD, encoding='utf-8').read() if os.path.exists(core.SUMMARY_MD) else ''

PCT_COLS = ['年化報酬', '年化波動', '最大回撤', '期間總報酬']


def price_chart(s, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=s.index, y=s, name='收盤價'), row=1, col=1)
    for n, nm in ((20, '月線'), (60, '季線'), (240, '年線')):
        fig.add_trace(go.Scatter(x=s.index, y=s.rolling(n).mean(), name=nm, line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s.index, y=core.rsi(s), name='RSI', showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line_dash='dot', line_color='gray', row=2, col=1)
    fig.add_hline(y=30, line_dash='dot', line_color='gray', row=2, col=1)
    fig.update_layout(title=title, height=520, hovermode='x unified', margin=dict(l=10, r=10),
                      legend=dict(orientation='h', y=1.08))
    fig.update_yaxes(range=[0, 100], title_text='RSI', row=2, col=1)
    return fig


def ui_lookup(code):
    """輸入任意代號：清單內的用已存資料，其餘現場向 Yahoo Finance 查詢。"""
    code = (code or '').strip().upper()
    if not code:
        return '請輸入股票代號，例如 2330、0050、NVDA、TSLA。', None
    candidates = [code + '.TW', code + '.TWO'] if re.fullmatch(r'\d{4,6}[A-Z]?', code) else [code]
    for c in candidates:
        if M is not None and c in M.close_all.columns:
            s = M.close_all[c].dropna()
            return core.describe_stock(s, core.label(c)), price_chart(s, core.label(c))
    last_err = ''
    for c in candidates:
        try:
            df = core.download_close([c])
            if c in df.columns and len(df[c].dropna()) > 5:
                s = df[c].dropna()
                return core.describe_stock(s, c), price_chart(s, c)
        except Exception as e:
            last_err = f'{type(e).__name__}'
    hint = f'（{last_err}）' if last_err else ''
    return (f'查不到「{code}」的資料{hint}。請確認代號：台股輸入數字即可，美股輸入英文代號。'
            '若代號正確，可能是資料來源暫時限流，請稍後再試。'), None


def ui_perf(tickers, currency):
    if not tickers:
        return None, None
    px_df = M.normalized(tickers, currency)
    nm = px_df / px_df.iloc[0] * 100
    fig = go.Figure()
    for t in nm.columns:
        fig.add_trace(go.Scatter(x=nm.index, y=nm[t], name=core.label(t),
                                 line=dict(dash='solid' if M.is_us(t) else 'dot')))
    fig.update_layout(title=f'累積績效（起點 = 100，{currency}；實線美股、虛線台股）', yaxis_type='log',
                      hovermode='x unified', height=480, margin=dict(l=10, r=10))
    st = core.perf_stats(px_df).sort_values('夏普值', ascending=False)
    for c in PCT_COLS:
        st[c] = st[c].map('{:.1%}'.format)
    st['夏普值'] = st['夏普值'].map('{:.2f}'.format)
    return fig, st.reset_index(names='標的')


def ui_corr(tickers, currency):
    if not tickers or len(tickers) < 2:
        return None
    weekly = M.normalized(tickers, currency).resample('W-FRI').last().pct_change().dropna(how='all')
    corr = weekly.corr()
    corr.index = corr.columns = [core.NAMES.get(t, t) for t in corr.columns]
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto',
                    title='週報酬相關係數（越接近 1，越常一起漲跌）', height=480)
    fig.update_layout(margin=dict(l=10, r=10))
    return fig


def ui_leadlag(us, tw):
    pair, bucket = M.leadlag(us, tw)
    k, b = np.polyfit(pair['us'], pair['tw'], 1)
    text = (f'**{core.label(us)} 隔夜 → {core.label(tw)} 隔日**　樣本 {len(pair)} 天，相關係數 '
            f'{pair.us.corr(pair.tw):.2f}，美股每漲 1% 台股隔日平均變動 {k:.2f}%，'
            f'方向一致比例 {(np.sign(pair.us) == np.sign(pair.tw)).mean():.0%}')
    fig = px.scatter(pair, x='us', y='tw', opacity=0.4, height=440,
                     labels={'us': '美股隔夜報酬', 'tw': '台股隔日報酬'})
    xs = np.array([pair.us.min(), pair.us.max()])
    fig.add_trace(go.Scatter(x=xs, y=b + k * xs, mode='lines', name='迴歸線'))
    fig.update_layout(xaxis_tickformat='.0%', yaxis_tickformat='.0%', margin=dict(l=10, r=10))
    bk = pd.DataFrame({'美股隔夜': bucket.index.astype(str), '天數': bucket['天數'].values,
                       '台股平均報酬': bucket['台股平均報酬'].map('{:+.2%}'.format).values,
                       '台股上漲機率': bucket['台股上漲機率'].map('{:.0%}'.format).values})
    return text, fig, bk


def ui_adr():
    fig = px.line(M.adr_premium(), height=460, title='ADR 相對台股的溢價（正）／折價（負）')
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.update_layout(yaxis_tickformat='.0%', hovermode='x unified', legend_title='', margin=dict(l=10, r=10))
    return fig


def ui_news(market, keyword):
    df = NEWS
    if market != '全部':
        df = df[df['market'] == market]
    if keyword and keyword.strip():
        k = keyword.strip()
        df = df[df['title'].str.contains(k, case=False, regex=False)
                | df['ticker'].str.contains(k, case=False, regex=False)]
    df = df.head(150)
    safe = df['title'].str.replace('[', '［', regex=False).str.replace(']', '］', regex=False)
    return pd.DataFrame({'時間': df['time'].dt.strftime('%m-%d %H:%M').fillna(''), '市場': df['market'],
                         '來源': df['source'], '代號': df['ticker'],
                         '標題（點擊開啟原文）': '[' + safe + '](' + df['url'] + ')'})


with gr.Blocks(title='美股 × 台股 整合分析') as demo:
    gr.Markdown(f"# 美股 × 台股 整合分析\n資料更新時間：{META.get('updated_at', '尚未更新')}（台北時間）")

    if not READY:
        gr.Markdown('### 尚無資料\n每日排程還沒有成功執行過。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
    else:
        with gr.Tab('今日摘要'):
            gr.Markdown(SUMMARY)

        with gr.Tab('個股查詢'):
            with gr.Row():
                in_code = gr.Textbox(label='股票代號', placeholder='台股輸入數字（2330、0050），美股輸入英文（NVDA、TSLA）', scale=4)
                btn_q = gr.Button('查詢', variant='primary', scale=1)
            out_qt, out_qf = gr.Markdown(), gr.Plot()
            btn_q.click(ui_lookup, in_code, [out_qt, out_qf])
            in_code.submit(ui_lookup, in_code, [out_qt, out_qf])

        with gr.Tab('新聞'):
            gr.Markdown(f"累積 {len(NEWS)} 則。{'　｜　'.join(META.get('news_status', []))}")
            with gr.Row():
                in_nm = gr.Radio(['全部', '台股', '美股'], value='全部', label='市場')
                in_nk = gr.Textbox(label='關鍵字或代號（輸入後按 Enter）')
            out_nt = gr.Dataframe(datatype=['str', 'str', 'str', 'str', 'markdown'], wrap=True)
            in_nm.change(ui_news, [in_nm, in_nk], out_nt)
            in_nk.submit(ui_news, [in_nm, in_nk], out_nt)
            demo.load(ui_news, [in_nm, in_nk], out_nt)

        with gr.Tab('績效比較'):
            choices = [(core.label(t), t) for t in M.sel_us + M.sel_tw]
            with gr.Row():
                in_tk = gr.Dropdown(choices, value=M.sel_us + M.sel_tw, multiselect=True, label='標的', scale=3)
                in_cur = gr.Radio(['美元', '新台幣', '各自原幣'], value='新台幣', label='計價幣別', scale=1)
            out_pf, out_ps, out_pc = gr.Plot(), gr.Dataframe(), gr.Plot()
            for comp in (in_tk, in_cur):
                comp.change(ui_perf, [in_tk, in_cur], [out_pf, out_ps])
                comp.change(ui_corr, [in_tk, in_cur], out_pc)
            demo.load(ui_perf, [in_tk, in_cur], [out_pf, out_ps])
            demo.load(ui_corr, [in_tk, in_cur], out_pc)

        with gr.Tab('美股 → 台股連動'):
            with gr.Row():
                in_us = gr.Dropdown([(core.label(t), t) for t in M.sel_us], value=core.US_LEAD, label='美股指標')
                in_tw = gr.Dropdown([(core.label(t), t) for t in M.sel_tw], value=core.TW_TARGET, label='台股標的')
            out_lt, out_lf, out_lb = gr.Markdown(), gr.Plot(), gr.Dataframe()
            for comp in (in_us, in_tw):
                comp.change(ui_leadlag, [in_us, in_tw], [out_lt, out_lf, out_lb])
            demo.load(ui_leadlag, [in_us, in_tw], [out_lt, out_lf, out_lb])

        with gr.Tab('ADR 溢價'):
            out_adr = gr.Plot()
            demo.load(ui_adr, None, out_adr)

    gr.Markdown('僅供研究與學習，不構成投資建議。')

if __name__ == '__main__':
    demo.launch()
