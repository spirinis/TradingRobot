import os
import pandas as pd
from pandas import DataFrame
from functools import lru_cache
from PyQt5.QtWidgets import QComboBox, QCheckBox, QWidget, QGridLayout
from PyQt5.QtCore import QRectF
import pyqtgraph as pg
import time
from numpy import isnan
from datetime import datetime, timedelta, timezone
from tinkoff.invest import (Client, HistoricCandle, Candle, CandleInterval, Quotation, MarketDataResponse)
from threading import Thread
from math import isnan
import math

import finplot as fplt
fplt.foreground = '#FF00FF'  # '#000'
fplt.background = '#303030'  # '#fff'
fplt.odd_plot_background = fplt.background  # '#eaeaea'
fplt.volume_bull_color = '#70d270'  # '#92d2cc'
fplt.volume_bear_color = '#f75957'  # '#f7a9a7'
fplt.cross_hair_color = '#C0C0C0'  # '#0007'
fplt.draw_line_color = '#fff'  # '#000'
fplt.draw_done_color = '#ddd6'  # '#555'

fplt.candle_bear_body_color = fplt.background   # fplt.candle_bear_color

token_read = os.getenv('TITOKEN_READ_ALL', 'Ключа нет')
global work
speed = 'real'
prev_speed_index = 1
counter_list = []
counter_dict = {}
subs_candles = 0
subs_order_book = 0
i = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


def cast_money(v: Quotation) -> float:
    return v.units + v.nano / 1e9  # nano - 9 нулей


def create_df(candles: [HistoricCandle]) -> pd.DataFrame or dict:
    from tinkoff.invest import (HistoricCandle, Candle)
    if isinstance(candles, list):
        df = DataFrame([{
            'time': c.time,
            'volume': c.volume,
            'open': cast_money(c.open),
            'close': cast_money(c.close),
            'high': cast_money(c.high),
            'low': cast_money(c.low),
        } for c in candles])
        return df
    elif isinstance(candles, Candle):
        c = candles
        df = {
            'time': c.time,
            'volume': c.volume,
            'open': cast_money(c.open),
            'close': cast_money(c.close),
            'high': cast_money(c.high),
            'low': cast_money(c.low),
        }
        return df


def time_str(to_dt_string: str) -> datetime:
    """
    %Y %m %d %H %M\n
    %Y %m %d\n
    %m %d %H %M\n
    %m %d
    """
    if to_dt_string is not None:
        if to_dt_string.count(' ') == 4:  # "%Y %m %d %H %M"
            time_format = "%Y %m %d %H %M"
        elif to_dt_string.count(' ') == 2:  # "%Y %m %d"
            time_format = "%Y %m %d"
            to_dt_string = to_dt_string
        elif to_dt_string.count(' ') == 3:   # "%m %d %H %M"
            time_format = "%Y %m %d %H %M"
            to_dt_string = str(datetime.today().year) + ' ' + to_dt_string
        else:   # "%m %d"
            time_format = "%Y %m %d"
            to_dt_string = str(datetime.today().year) + ' ' + to_dt_string
        required_time = datetime.strptime(to_dt_string, time_format) - timedelta(minutes=1)  # - timedelta(hours=3)
    else:
        required_time = datetime.now()
    return required_time


def FindShareOrTicker(ticker: str = '', figi: str = '') -> dict:
    shares = pd.read_csv('used_shares.csv')
    if ticker != '':
        find_share = shares[shares['ticker'] == ticker]
    elif figi != '':
        find_share = shares[shares['figi'] == figi]
    else:
        raise Exception('Чё?')
    if find_share.empty:
        all_shares = pd.read_csv('all_shares.csv')
        if ticker != '':
            find_share = all_shares[all_shares['ticker'] == ticker]
        elif figi != '':
            find_share = all_shares[all_shares['figi'] == figi]
        if find_share.empty:
            raise Exception('Нет такого тикера')
        else:
            os.remove('used_shares.csv')
            pd.concat([shares, find_share], ignore_index=True).to_csv('used_shares.csv', index=False)
    find_share = {'name': find_share.name.iloc[0], 'ticker': find_share.ticker.iloc[0], 'figi': find_share.figi.iloc[0],
                  'lot': find_share.lot.iloc[0], 'currency': find_share.currency.iloc[0],
                  'class_code': find_share.class_code.iloc[0]}
    return find_share


def do_load_price_history(ticker, interval, dt_string=None):
    from tinkoff.invest import RequestError
    figi = FindShareOrTicker(ticker=ticker).get('figi')  # 'BBG006L8G4H1'
    if not is_historic:
        if hist_date.weekday() == 0:
            dt_object = datetime.now() - timedelta(hours=3)
        else:
            dt_object = datetime.now() - timedelta(hours=3)  # поправка на часовой пояс
    else:
        if hist_date.weekday() == 0:
            dt_object = hist_date - timedelta(days=3)
        else:
            dt_object = hist_date
    candles = None
    while candles is None:
        try:
            with Client(token_read) as client:
                print('loading %s %s' % (ticker, interval))
                candles = client.market_data.get_candles(
                    figi=figi,
                    # from_=datetime.utcnow()-timedelta(hours = 24), to=datetime.utcnow(),
                    from_=dt_object - timedelta(hours=24),
                    to=dt_object,
                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                )
            df = create_df(candles.candles)
            df['iterator'] = range(0, len(df))
            df['g_extr_up'] = float("NaN")
            df['g_extr_dwn'] = float("NaN")
            df['g_extr_des'] = float("NaN")
            return df  # .set_index('Time')
        except RequestError as error:
            print(__file__ + ' do_load_price_history '+str(type(error)) + ' | ' + str(error))
            exception_log = open('spectator1 data/exception_log.txt', 'a+', encoding='utf-8')
            exception_log.write(__file__ + ' do_load_price_history ' + str(type(error)) + ' | ' + str(error) + '\n')
            exception_log.close()
            continue


@lru_cache(maxsize=5)
def cache_load_price_history(ticker, interval):
    """Stupid caching, but works sometimes."""
    return do_load_price_history(ticker, interval)


def load_price_history(ticker, interval):
    """Use memoized, and if too old simply load the data."""
    global df
    df = cache_load_price_history(ticker, interval)
    # check if cache's newest candle is current
    t0 = df.time.iloc[-2].timestamp()
    t1 = df.time.iloc[-1].timestamp()
    t2 = t1 + (t1 - t0)
    if time.time() >= t2:
        df = do_load_price_history(ticker, interval)
    return df


def create_ctrl_panel(win):
    panel = QWidget(win)
    panel.move(60, 0)
    win.scene().addWidget(panel)
    layout = QGridLayout(panel)

    def draw_candles_toggle(condition):
        if condition:
            plots['candlestick'].show()
        else:
            plots['candlestick'].hide()

    def max_bars_change():
        global max_bars, i
        pre_max_bars = ctrl_panel.max_bars.currentText()
        if pre_max_bars == 'all':
            max_bars = i
        else:
            max_bars = int(pre_max_bars)

    panel.ticker = QComboBox(panel)
    [panel.ticker.addItem(i) for i in 'YNDX OZON FIVE'.split()]
    panel.ticker.setCurrentIndex(1)
    layout.addWidget(panel.ticker, 0, 0)
    panel.ticker.currentTextChanged.connect(change_asset)

    #layout.setColumnMinimumWidth(1, 30)

    panel.interval = QComboBox(panel)
    [panel.interval.addItem(i) for i in '1d 1h 15m 5m 1m'.split()]
    panel.interval.setCurrentIndex(6)
    layout.addWidget(panel.interval, 0, 1)
    panel.interval.currentTextChanged.connect(change_asset)

    #layout.setColumnMinimumWidth(3, 30)

    panel.indicators = QComboBox(panel)
    [panel.indicators.addItem(i) for i in 'Clean:Few indicators:Moar indicators'.split(':')]
    panel.indicators.setCurrentIndex(1)
    layout.addWidget(panel.indicators, 0, 2)
    panel.indicators.currentTextChanged.connect(change_asset)

    #layout.setColumnMinimumWidth(5, 30)

    panel.speed = QComboBox(panel)
    [panel.speed.addItem(i) for i in 'pause real real/5 1s 0.01s 0s stop'.split()]
    panel.speed.setCurrentIndex(1)
    layout.addWidget(panel.speed, 0, 3)
    panel.speed.currentTextChanged.connect(start_pause)

    #layout.setColumnMinimumWidth(5, 30)

    panel.fps = QComboBox(panel)
    [panel.fps.addItem(i) for i in '1 10 100 0'.split()]
    panel.fps.setCurrentIndex(1)
    layout.addWidget(panel.fps, 0, 4)
    panel.fps.currentTextChanged.connect(fps_change)

    panel.max_bars = QComboBox(panel)
    [panel.max_bars.addItem(i) for i in '1500 4500 all'.split()]
    panel.max_bars.setCurrentIndex(0)
    layout.addWidget(panel.max_bars, 0, 5)
    panel.max_bars.currentTextChanged.connect(max_bars_change)

    panel.candles = QCheckBox(panel)
    panel.candles.setText('Candles')
    panel.candles.setCheckState(pg.Qt.QtCore.Qt.CheckState.Checked)
    panel.candles.toggled.connect(draw_candles_toggle)
    layout.addWidget(panel.candles, 0, 6)

    return panel


def start_pause():
    global speed, prev_speed_index, go
    if ctrl_panel.speed.currentIndex() != 0:
        prev_speed_index = ctrl_panel.speed.currentIndex()
    speed = ctrl_panel.speed.currentText()
    if ('stop' in speed) or ('pause' in speed):
        global fps
        fps = 0
        ctrl_panel.fps.setCurrentIndex(3)
        fps_change()
        if 'stop' in speed:
            go = False
    elif speed == '1s':
        fps = 1
        ctrl_panel.fps.setCurrentIndex(0)
        fps_change()
    elif (speed == 'real') or (speed == 'real/5') or (speed == '0.01s') or (speed == '0s'):
        fps = 100
        ctrl_panel.fps.setCurrentIndex(2)
        fps_change()


def fps_change():
    global fps
    fps = int(ctrl_panel.fps.currentText())
    if fps == 0:
        if timer.isActive():
            timer.stop()
    else:
        if not timer.isActive():
            timer.start()
        timer.setInterval(int(1000 / fps))


def time_jump(n_jump):
    pass
    '''
    global i, extr_df
    global g_napr, g_extr_up, g_extr_dwn, g_i_extr_up, g_i_extr_dwn
    print('AAAAAAAAAAAAAAAAAAAAAAAA')
    i -= n_jump
    globals()['candle_df_%s' % ticker] = globals()['candle_df_%s' % ticker][:-n_jump]
    extr_df = extr_df[extr_df.iterator < globals()['candle_df_%s' % ticker].iterator.iloc[-1]]
    # g_napr g_extr_up g_extr_dwn g_prev_i_extr g_prev_extr g_i_extr_up, g_i_extr_dwn
    if isnan(extr_df.g_extr_dwn.iloc[-1]):
        g_napr = -1
        g_extr_up = extr_df.g_extr_up.iloc[-1]
        g_i_extr_up = extr_df.iterator.iloc[-1]
        g_extr_dwn = extr_df.g_extr_dwn.iloc[-2]
        g_i_extr_dwn = extr_df.iterator.iloc[-2]
    else:
        g_napr = 1
        g_extr_dwn = extr_df.g_extr_dwn.iloc[-1]
        g_i_extr_dwn = extr_df.iterator.iloc[-1]
        g_extr_up = extr_df.g_extr_up.iloc[-2]
        g_i_extr_up = extr_df.iterator.iloc[-2]
    '''


def _key_pressed(vb, ev):
    global speed
    if ev.text() == 'g':  # grid
        global clamp_grid
        fplt.clamp_grid = not fplt.clamp_grid
        for win in fplt.windows:
            for ax in win.axs:
                ax.crosshair.update()
    elif ev.text() == 'i':  # invert
        for win in fplt.windows:
            for ax in win.axs:
                ax.setTransform(ax.transform().scale(1, -1).translate(0, -ax.height()))
    elif ev.text() == '\r':  # enter
        vb.set_draw_line_color(fplt.draw_done_color)
        vb.draw_line = None
    elif ev.text() == ' ':  # space
        if 'pause' in speed:
            ctrl_panel.speed.setCurrentIndex(prev_speed_index)
        else:
            ctrl_panel.speed.setCurrentIndex(0)
        start_pause()
    elif ev.text() in ('\x7f', '\b'): # del, backspace
        if not vb.remove_last_roi():
            return False
    elif ev.key() == fplt.QtCore.Qt.Key.Key_Left:
        time_jump(25)
        # vb.pan_x(percent=-15)
    elif ev.key() == fplt.QtCore.Qt.Key.Key_Right:
        vb.pan_x(percent=+15)
    elif ev.key() == fplt.QtCore.Qt.Key.Key_Home:
        vb.pan_x(steps=-1e10)
        fplt._repaint_candles()
    elif ev.key() == fplt.QtCore.Qt.Key.Key_End:
        vb.pan_x(steps=+1e10)
        fplt._repaint_candles()
    elif ev.key() == fplt.QtCore.Qt.Key.Key_Escape and fplt.key_esc_close:
        vb.win.close()
    else:
        return False
    return True


fplt._key_pressed = _key_pressed


def change_asset():
    """Resets and recalculates everything, and plots for the first time."""
    # save window zoom position before resetting
    fplt._savewindata(fplt.windows[0])

    ticker = ctrl_panel.ticker.currentText()
    interval = ctrl_panel.interval.currentText()
    df = load_price_history(ticker, interval=interval)
    globals()['candle_df_%s' % ticker] = df
    # remove any previous plots
    ax.reset()
    ax_vol.reset()

    # calculate plot data
    indicators = ctrl_panel.indicators.currentText().lower()
    data, price_data = calc_plot_data(df, indicators)

    # some space for legend
    ctrl_panel.move(100 if 'clean' in indicators else 200, 0)

    # plot data
    global plots
    plots = dict()
    plots['candlestick'] = fplt.candlestick_ochl(data['candlestick'], ax=ax)
    plots['volume'] = fplt.volume_ocv(data['volume'], ax=ax_vol)
    plots['price'] = fplt.plot(data['price'], legend='price', ax=ax, color='#fff')

    global tilt_line
    point1, point2, k, b = approximation(data['price'].iterator.to_list()[-10:], data['price'].close.to_list()[-10:])
    tilt_line = fplt.add_line(point1, point2, color='#993', ax=ax, width=2)  # , interactive=True)

    global extr_up_line
    extr_up_df = extr_df[['iterator', 'g_extr_up']].dropna()
    if extr_up_df.__len__() > 1:
        point1, point2, k, b = approximation(extr_up_df.iterator.to_list()[-2:], extr_up_df.g_extr_up.to_list()[-2:])
        extr_up_line = fplt.add_line(point1, point2, color='#f66', ax=ax, width=2)  # , interactive=True)
    global extr_dwn_line
    extr_dwn_df = extr_df[['iterator', 'g_extr_dwn']].dropna()
    if extr_dwn_df.__len__() > 1:
        point1, point2, k, b = approximation(extr_dwn_df.iterator.to_list()[-2:], extr_dwn_df.g_extr_dwn.to_list()[-2:])
        extr_dwn_line = fplt.add_line(point1, point2, color='#66f', ax=ax, width=2)  # , interactive=True)

    ax.set_visible(xaxis=True)

    # price line
    ax.price_line = pg.InfiniteLine(angle=0, movable=False, pen=fplt._makepen(fplt.candle_bull_body_color, style='.'))
    ax.price_line.setPos(price_data['last_close'])
    ax.price_line.pen.setColor(pg.mkColor(price_data['last_col']))
    ax.addItem(ax.price_line, ignoreBounds=True)

    # restores saved zoom position, if in range
    fplt.refresh()

    '''
    ax, ax_vol = fplt.create_plot('terminal', rows=2)
    fplt.candlestick_ochl(data['candlestick'], ax=ax)
    fplt.volume_ocv(candle_hist_df_OZON['iterator open close volume'.split()], ax=ax_vol)
    fplt.plot(candle_hist_df_OZON['iterator close'.split()], legend='price', ax=ax, color='#fff')
    fplt.show()
    '''


def approximation(x_values, y_values):
    n = len(x_values)
    average_x = 0
    average_y = 0
    average_xy = 0
    average_xx = 0
    for i in range(n):
        average_x += x_values[i]
        average_y += y_values[i]
        average_xy += x_values[i] * y_values[i]
        average_xx += x_values[i] ** 2
    average_x = average_x / n
    average_y = average_y / n
    average_xy = average_xy / n
    average_xx = average_xx / n
    d_x = average_xx - average_x ** 2
    a = (average_xy - average_x * average_y) / d_x
    b = average_y - a * average_x
    x = [x_values[0], x_values[-1]]
    y = [a * x[0] + b, a * x[1] + b]
    point1 = [x[0], y[0]]
    point2 = [x[1], y[1]]
    k = a
    return point1, point2, k, b


tilt_line = None
tilt_line_list = []
extr_up_line = None
extr_up_line_shadow = None
extr_dwn_line = None
extr_dwn_line_shadow = None
extr_medium_line = None
i_with_hist = None


def realtime_update_plot():
    """Called at regular intervals by a timer."""
    # if ws.df is None:
    #     return
    # df = DataFrame({
    #    'time': [],
    #    'volume': [],
    #    'open': [],
    #    'close': [],
    #    'high': [],
    #    'low': [],
    #    'last_trade_ts': [],
    # })

    global i_with_hist
    ticker = ctrl_panel.ticker.currentText()
    if globals()['candle_df_%s' % ticker].empty \
            or (i_with_hist == globals()['candle_df_%s' % ticker].iterator.iloc[-1]):
        return

    i_with_hist = globals()['candle_df_%s' % ticker].iterator.iloc[-1]
    # calculate the new plot data
    indicators = ctrl_panel.indicators.currentText().lower()
    data, price_data = calc_plot_data(globals()['candle_df_%s' % ticker], indicators)

    # first update all data, then graphics (for zoom rigidity)
    for k in data:
        if data[k] is not None:
            if k not in plots:
                if k == 'g_extr_up':
                    plots['g_extr_up'] = fplt.plot(data['g_extr_up'], legend='g_extr_up', ax=ax, color='#ff6666',
                                                   width=1, style='o')
                elif k == 'g_extr_dwn':
                    plots['g_extr_dwn'] = fplt.plot(data['g_extr_dwn'], legend='g_extr_dwn', ax=ax, color='#6666ff',
                                                    width=1, style='o')
                elif k == 'g_extr_des':
                    plots['g_extr_des'] = fplt.plot(data['g_extr_des'], legend='g_extr_des', ax=ax, color='gray',
                                                    width=1, style='o')
            plots[k].update_data(data[k], gfx=False)
    for k in data:
        if data[k] is not None:
            plots[k].update_gfx()
    # plots['price'].update_gfx()
    # plots['volume'].update_gfx()
    # place and color price line
    ax.price_line.setPos(price_data['last_close'])
    ax.price_line.pen.setColor(pg.mkColor(price_data['last_col']))

    amendment = (max_bars - 1 - i_with_hist)
    if amendment > 0:
        amendment = 0
    global tilt_line, tilt_line_list
    point1, point2, k, b = approximation(data['price'].iterator.to_list()[-10:], data['price'].close.to_list()[-10:])
    tilt_line_list.append(math.degrees(math.atan(k)))
    point1[0] += amendment
    point2[0] += amendment
    # tilt_line = fplt.add_line(point1, point2, color='#993', ax=ax)  # , interactive=True)
    # tilt_line.update(QRectF(point1[0], point1[1], point2[0], point2[1]))
    tilt_line.hide()
    tilt_line = fplt.add_line(point1, point2, color='#993', ax=ax, width=2)  # , interactive=True)

    global extr_up_line
    extr_up_df = extr_df[['iterator', 'g_extr_up']].dropna()
    if extr_up_df.__len__() > 2:
        extr_up_point1, extr_up_point2, extr_up_k, extr_up_b = approximation(extr_up_df.iterator.to_list()[-2:],
                                                                             extr_up_df.g_extr_up.to_list()[-2:])

        print(point1, point2, k)

        extr_up_point1[0] += amendment
        extr_up_point2[0] += amendment
        try:
            extr_up_line.hide()
        except AttributeError:
            pass
        extr_up_lengthening_y = (i_with_hist + amendment + 2 - extr_up_point2[0])*extr_up_k
        # extr_up_point2[0] = i_with_hist + amendment + 2
        # extr_up_point2[1] += lengthening_y
        extr_up_line = fplt.add_line(
            extr_up_point1, [i_with_hist + amendment + 2, extr_up_point2[1] + extr_up_lengthening_y],
            color='#f66', ax=ax, width=2)  # , interactive=True)

    global extr_dwn_line
    extr_dwn_df = extr_df[['iterator', 'g_extr_dwn']].dropna()
    if extr_dwn_df.__len__() > 2:
        extr_dwn_point1, extr_dwn_point2, extr_dwn_k, extr_dwn_b = approximation(extr_dwn_df.iterator.to_list()[-2:],
                                                                                 extr_dwn_df.g_extr_dwn.to_list()[-2:])
        extr_dwn_point1[0] += amendment
        extr_dwn_point2[0] += amendment
        try:
            extr_dwn_line.hide()
        except AttributeError:
            pass
        extr_dwn_lengthening_y = (i_with_hist + amendment + 2 - extr_dwn_point2[0]) * extr_dwn_k
        # extr_dwn_point2[0] = i_with_hist + amendment + 2
        # extr_dwn_point2[1] += lengthening_y
        extr_dwn_line = fplt.add_line(
            extr_dwn_point1, [i_with_hist + amendment + 2, extr_dwn_point2[1] + extr_dwn_lengthening_y],
            color='#66f', ax=ax, width=2)

    global extr_up_line_shadow
    if (extr_up_df.__len__() > 2) and (extr_dwn_df.__len__() > 2):
        try:
            extr_up_line_shadow.hide()
        except AttributeError:
            pass
        extr_up_shadow_lengthening_y = (i_with_hist + amendment + 2 - extr_up_point2[0]) * extr_up_k
        # x = [x_values[0], x_values[-1]]
        # y = [a * x[0] + b, a * x[1] + b]
        extr_up_line_shadow = fplt.add_line(
            extr_dwn_point1,
            [i_with_hist + amendment + 2, extr_up_k*(i_with_hist + amendment + 2) + extr_up_b
             - (extr_up_k * extr_dwn_point1[0] + extr_up_b - extr_dwn_point1[1])],
            color='#b33', ax=ax, style='.-', width=2)

    global extr_dwn_line_shadow
    if (extr_up_df.__len__() > 2) and (extr_dwn_df.__len__() > 2):
        try:
            extr_dwn_line_shadow.hide()
        except AttributeError:
            pass
        extr_dwn_shadow_lengthening_y = (i_with_hist + amendment + 2 - extr_dwn_point2[0]) * extr_dwn_k
        # x = [x_values[0], x_values[-1]]
        # y = [a * x[0] + b, a * x[1] + b]
        extr_dwn_line_shadow = fplt.add_line(
            extr_up_point1,
            [i_with_hist + amendment + 2, extr_dwn_k * (i_with_hist + amendment + 2) + extr_dwn_b
             - (extr_dwn_k * extr_up_point1[0] + extr_dwn_b - extr_up_point1[1])],
            color='#33b', ax=ax, style='.-', width=2)
        # print('k= ', extr_up_k, 'b= ', extr_up_b, extr_dwn_line_shadow.points)
    global extr_medium_line
    if (extr_up_df.__len__() > 2) and (extr_dwn_df.__len__() > 2):
        try:
            extr_medium_line.hide()
        except AttributeError:
            pass
        extr_dwn_shadow_lengthening_y = (i_with_hist + amendment + 2 - extr_dwn_point2[0]) * extr_dwn_k
        # x = [x_values[0], x_values[-1]]
        # y = [a * x[0] + b, a * x[1] + b]
        extr_medium_line = fplt.add_line(
            [(extr_up_point1[0] + extr_dwn_point1[0])/2, (extr_up_point1[1] + extr_dwn_point1[1])/2],
            [i_with_hist + amendment + 2,
             (extr_up_point2[1] + extr_up_lengthening_y + extr_dwn_point2[1] + extr_dwn_lengthening_y)/2],
            color='#bb0', ax=ax, style='.-', width=2)

    '''x_list = extr_df_to_list.index.to_list()
    y_list = extr_df_to_list.extr_up.to_list()
    points_count = 30
    line = approximation(x_list[-points_count:], y_list[-points_count:])
    x = [x_list[-points_count], x_list[-1]]
    y = [a * x[0] + b, a * x[1] + b]

    fplt.add_line((x[0], y[0]), (x[1], y[1]), color='#993', ax=ax)  # , interactive=True)
    extr_df_to_list = pd.concat([candle_df_OZON[['iterator', 'last_trade_ts', 'g_extr_up']].dropna(),
                                 candle_df_OZON[['iterator', 'last_trade_ts', 'g_extr_dwn']].dropna()])
    extr_df_to_list.fillna(0, inplace=True)
    extr_df_to_list.g_extr_up = extr_df_to_list.g_extr_up + extr_df_to_list.g_extr_dwn
    extr_df_to_list.drop(['g_extr_dwn'], axis=1, inplace=True)
    extr_df_to_list.sort_index(inplace=True)'''


max_bars = 1500
plot_data = 0


def calc_plot_data(df, indicators):
    """Returns data for all plots and for the price line."""
    global plot_data, max_bars
    price = df['iterator close'.split()].iloc[-max_bars:]
    volume = df['iterator open close volume'.split()].iloc[-max_bars:]
    plot_data = dict(price=price, volume=volume)
    if ctrl_panel.candles.isChecked():
        candlestick = df['iterator open close high low'.split()].iloc[-max_bars:]
        plot_data['candlestick'] = candlestick
    # TODO: решение появляется вместе с экстремумом, за 1 итерацию только 1 экстремум => обновлять только 2 графика
    if (not 'g_extr_up' in plot_data) and (not df.g_extr_up.isna().all()):
        plot_data['g_extr_up'] = df[['iterator', 'g_extr_up']].iloc[-max_bars:]
    if (not 'g_extr_dwn' in plot_data) and (not df.g_extr_dwn.isna().all()):
        plot_data['g_extr_dwn'] = df[['iterator', 'g_extr_dwn']].iloc[-max_bars:]
    if (not 'g_extr_des' in plot_data) and (not df.g_extr_des.isna().all()):
        plot_data['g_extr_des'] = df[['iterator', 'g_extr_des']].iloc[-max_bars:]
    # for price line
    last_close = price.iloc[-1].close
    last_col = fplt.candle_bull_color if last_close > price.iloc[-2].close else fplt.candle_bear_color
    price_data = dict(last_close=last_close, last_col=last_col)
    return plot_data, price_data


def request_iterator_candles(figis: list, action: str):
    from tinkoff.invest import (
        MarketDataRequest,
        SubscribeCandlesRequest,
        CandleInstrument,
        SubscriptionAction,
        SubscriptionInterval
    )
    candle_instruments = []
    global subs_candles
    if action == 'subscribe':
        action = SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE
        subs_candles += 1
    elif action == 'unsubscribe':
        action = SubscriptionAction.SUBSCRIPTION_ACTION_UNSUBSCRIBE
        subs_candles -= 1
    for figi in figis:
        candle_instruments.append(
            CandleInstrument(
                figi=figi,
                interval=SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE)
        )
    print('reqc')
    yield MarketDataRequest(
        subscribe_candles_request=SubscribeCandlesRequest(
            waiting_close=is_close,
            subscription_action=action,
            instruments=candle_instruments
        )
    )
    i = 0
    while go:
        time.sleep(1)
        i += 1
        if i >= 5:
            print('sleepC ', i)
            i -= 5


def FindTickerByFigi(figi: str) -> str:
    return df_of_shares[df_of_shares.figi == figi].ticker.iloc[0]


def append_candle_df(candle: Candle) -> DataFrame:
    global counter_list
    ticker = FindTickerByFigi(candle.figi)
    df = DataFrame([{
        'time': candle.time,
        'volume': candle.volume,
        'open': cast_money(candle.open),
        'close': cast_money(candle.close),
        'high': cast_money(candle.high),
        'low': cast_money(candle.low),
        'last_trade_ts': candle.last_trade_ts,
    }], index=[counter_dict[ticker]])
    counter_dict[ticker] += 1
    # print(globals()['candle_df_%s' % ticker].iloc[[-1]])
    if ((len(globals()['candle_df_%s' % ticker]) > 0) and is_close
            and (globals()['candle_df_%s' % ticker].time.iloc[-1] == candle.time)):
        globals()['candle_df_%s' % ticker] = globals()['candle_df_%s' % ticker][:-1]
    globals()['candle_df_%s' % ticker] = pd.concat([globals()['candle_df_%s' % ticker], df])  # .iloc[1:]
    # realtime_update_plot(globals()['candle_df_%s' % ticker]['open close high low'.split()])

    df['ticker'] = [ticker]
    return df


def unpack_MarketDataResponse(mdr: MarketDataResponse):
    global to_sleep_flag
    import tinkoff.invest.schemas
    list_response = [mdr.subscribe_candles_response,
                     mdr.subscribe_order_book_response,
                     mdr.subscribe_trades_response,
                     mdr.subscribe_info_response,
                     mdr.candle,
                     mdr.trade,
                     mdr.orderbook,
                     mdr.trading_status,
                     mdr.ping,
                     mdr.subscribe_last_price_response,
                     mdr.last_price]
    list_response_translate = []
    for response in list_response:
        if isinstance(response, tinkoff.invest.schemas.SubscribeCandlesResponse):
            df_response = DataFrame([{
                'ticker': FindTickerByFigi(candles_subscription.figi),
                'candle_sbscr_status': candles_subscription.subscription_status.name,
            } for candles_subscription in response.candles_subscriptions])
            list_response_translate.append(df_response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeOrderBookResponse):
            df_response = DataFrame([{
                'ticker': FindTickerByFigi(order_book_subscription.figi),
                'order_book_sbscr_status': order_book_subscription.subscription_status.name,
            } for order_book_subscription in response.order_book_subscriptions])
            list_response_translate.append(df_response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeTradesResponse):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeInfoResponse):
            # list_response_translate.append(response.info_subscriptions[0].subscription_status.name)
            df_response = DataFrame([{
                'ticker': FindTickerByFigi(info_subscription.figi),
                'info_sbscr_status': info_subscription.subscription_status.name,
            } for info_subscription in response.info_subscriptions])
            list_response_translate.append(df_response)
        elif isinstance(response, tinkoff.invest.schemas.Candle):
            list_response_translate.append(append_candle_df(response))
        elif isinstance(response, tinkoff.invest.schemas.Trade):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.OrderBook):
            if (len(response.bids) != 0) and (len(response.asks) != 0):
                ticker = FindTickerByFigi(response.figi)
                df_bids = DataFrame([{
                    'price': cast_money(order.price),
                    'quantity_buy': order.quantity
                } for order in response.bids])
                df_asks = DataFrame([{
                    'price': cast_money(order.price),
                    'quantity_sale': order.quantity
                } for order in response.asks])
                df_response = pd.merge(df_bids, df_asks, how='outer')
                #  df_response = pd.concat([df_bids, df_asks])
                df_response['time'] = response.time
                df_response['is_consistent'] = response.is_consistent
                """df_response = DataFrame({
                    'ticker': FindTickerByFigi(response.figi),
                    'time': response.time,
                })"""
                globals()['volume_df_%s' % ticker] = pd.concat([globals()['volume_df_%s' % ticker], df_response])
                # list_response_translate.append(df_response)
                list_response_translate.append(df_response)
            else:
                list_response_translate.append('Пустой стакан')
        elif isinstance(response, tinkoff.invest.schemas.TradingStatus):
            list_response_translate.append(response.trading_status.name)
        elif isinstance(response, tinkoff.invest.schemas.Ping):
            list_response_translate.append(response)
            if (response.time.hour == 20 and response.time.minute >= 50) or (response.time.hour >= 21):
                to_sleep_flag = True
                print('анпак прерывание')
                print(response)
                raise KeyboardInterrupt
        elif isinstance(response, tinkoff.invest.schemas.SubscribeLastPriceResponse):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.LastPrice):
            list_response_translate.append(response)
        else:
            list_response_translate.append(response)
    list_response_clear = list(filter(lambda x: x is DataFrame or x is not None, list_response_translate))
    columns = ['sbsc_c', 'sbsc_or_b', 'sbsc_trades', 'sbsc_info', 'candle', 'trade', 'orderbook', 'trading_status',
               'ping', 'subscribe_last_price_response', 'last_price']
    sery = pd.Series(list_response_translate, index=columns)
    out = {'list_response_translate': list_response_translate, 'sery': sery,
           'list_response_clear': list_response_clear, }
    return out


def FindShareByTicker(ticker: str) -> DataFrame:
    from tinkoff.invest import (InstrumentStatus, RequestError)
    from tinkoff.invest.services import (InstrumentsService, )
    try:
        with Client(token_read) as client:
            instruments: InstrumentsService = client.instruments
            # market_data: MarketDataService = client.market_data
            try:
                shares = pd.read_csv('all_shares.csv')
            except FileNotFoundError:
                shares = DataFrame(
                    instruments.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE).instruments,
                    columns=['name', 'ticker', 'figi', 'lot', 'currency', 'class_code']
                )
                shares.to_csv('all_shares.csv', index=False)
            find_share = shares[shares['ticker'] == ticker]
            # r = r[r['ticker'] == ticker]['figi'].iloc[0]
            if find_share.empty:
                raise Exception('Нет такого тикера')

                #{'name': find_share.iloc[0][0], 'ticker': find_share.iloc[0][1], 'figi': find_share.iloc[0][2],
                #          'lot': find_share.iloc[0][3], 'currency': find_share.iloc[0][4],
                #          'class_code': find_share.iloc[0][5]}
            return find_share
    except RequestError as error:
        print('ошибка find' + "\n" + str(error))


def stream_candles():
    global go
    while go:
        try:
            with Client(token_read) as client:
                for marketdata in client.market_data_stream.market_data_stream(
                        request_iterator_candles(list_of_figis, 'subscribe')
                ):
                    # fplt.timer_callback(realtime_update_plot, 0.5, single_shot=True)
                    response = unpack_MarketDataResponse(marketdata).get('list_response_clear')
                    print('===== ОТВЕТ =====\n', response[0])
                    '''if isinstance(response, tinkoff.invest.schemas.Ping):
                        if (response.time.hour == 20 and response.time.minute >= 50) or (response.time.hour >= 21):
                            break
                    if subs_candles == 0:
                        break'''
                print('outc')
        except Exception as exc:
            print('stream_candles ' + str(type(exc)) + ' | ' + str(exc))
            continue


def stream_historic_candles():
    global response, i, extr_df
    ticker = ctrl_panel.ticker.currentText()
    globals()['candle_hist_df_%s' % ticker] = pd.read_csv(hist_path % (hist_date.strftime('%#d.%#m')
                                                                       + '_' + ticker))
    globals()['candle_hist_df_%s' % ticker].time = pd.to_datetime(globals()['candle_hist_df_%s' % ticker].time)
    globals()['candle_hist_df_%s' % ticker].last_trade_ts = pd.to_datetime(
       globals()['candle_hist_df_%s' % ticker].last_trade_ts)
    hist_df_len = len(globals()['candle_hist_df_%s' % ticker])
    before_df_len = len(globals()['candle_df_%s' % ticker])
    globals()['candle_hist_df_%s' % ticker]['iterator'] = range(hist_df_len)
    while True:
    # for i in range(hist_df_len):
        if not go:
            break
        while speed == 'pause':
            time.sleep(1)
        response = globals()['candle_hist_df_%s' % ticker].iloc[[i]]
        if (i != 0) and (i < hist_df_len):
            if speed == 'real':
                sleep = (response.last_trade_ts.iloc[-1] -
                         globals()['candle_df_%s' % ticker].last_trade_ts.iloc[-1]).total_seconds()
            elif speed == 'real/5':
                sleep = (response.last_trade_ts.iloc[-1] -
                         globals()['candle_df_%s' % ticker].last_trade_ts.iloc[-1]).total_seconds() / 5
            else:
                if speed == 'stop':
                    break
                sleep = float(speed[:-1])
            if sleep < 0:
                sleep = 0
            time.sleep(sleep)
        if not globals()['candle_df_%s' % ticker].empty:
            response.at[response.index[-1], 'iterator'] = globals()['candle_df_%s' % ticker].iterator.iloc[-1] + 1
        globals()['candle_df_%s' % ticker] = pd.concat([globals()['candle_df_%s' % ticker], response], ignore_index=True)
        extremum(before_df_len, i, ticker)

        print('===== ОТВЕТ =====\n', response)
        i += 1


    print('out_hist_stream')


waves_list = []
wave_period_list = []
extr_df = DataFrame({
        'iterator': [],
        'g_extr_up': [],
        'g_extr_dwn': [],
        'g_extr_any': [],
    })
ideal_sum_percent_profit = 0
g_napr = 1
g_extr_up = 0
g_extr_dwn = 999
g_i_extr = 0
g_prev_extr = 999
# g_extr_medium = price*2
g_prev_i_extr = 0
g_prev_buy_extr = 0
g_percent_extr = 0.10 / 100 # TODO 0.10 / 100
commission = 0.10 / 100
g_i_extr_up = 0
g_i_extr_dwn = 0
g_i_time = 0
g_i_time_count = 0


def extremum(before_df_len, i, ticker):
    """====== ЭКСТРЕМУМ ЗАЗОРА ======"""
    global extr_df, g_napr, g_extr_up, g_extr_up_count, g_extr_dwn, g_i_extr, g_prev_extr, g_extr_medium, g_prev_i_extr,\
        g_prev_buy_extr, g_i_extr_up, g_i_extr_dwn, g_i_time, g_i_time_count, ideal_sum_percent_profit
    df = globals()['candle_df_%s' % ticker]
    if i == 0:
        g_extr_dwn = df.close.loc[before_df_len] * 2
        g_prev_extr = df.close.loc[before_df_len]
    i += before_df_len
    if g_napr == 1 and df.close.loc[i] * (1 + g_percent_extr) < g_extr_up:
        g_napr = -1
        df.loc[g_i_extr_up, 'g_extr_up'] = g_extr_up
        df.loc[i, 'g_extr_des'] = g_extr_up
        extr_df = pd.concat([extr_df, pd.DataFrame({'iterator': [g_i_extr_up], 'g_extr_up': [g_extr_up],
                                                    'g_extr_any': [g_extr_up]})], ignore_index=True)
        # g_i_extr_medium = round((g_prev_i_extr + g_i_extr)/2)
        # g_extr_medium = (g_prev_extr + g_extr_up)/2
        # df.loc[g_i_extr_medium, 'g_extr_medium'] = g_extr_medium
        waves_list.append(round(g_extr_up / (g_prev_extr / 100) - 100, 2))
        ideal_sum_percent_profit += waves_list[-1] - commission
        g_prev_extr = g_extr_up
        g_prev_buy_extr = g_extr_up
        g_i_time += g_i_extr_up - g_prev_i_extr
        wave_period_list.append(g_i_extr_up - g_prev_i_extr)
        g_i_time_count += 1
        g_prev_i_extr = g_i_extr_up

        extr_up = g_extr_up
        i_extr_up = g_i_extr_up

        g_extr_up = 0
    elif g_napr == -1 and df.close.loc[i] * (1 - g_percent_extr) > g_extr_dwn:
        g_napr = 1
        df.loc[g_i_extr_dwn, 'g_extr_dwn'] = g_extr_dwn
        df.loc[i, 'g_extr_des'] = g_extr_dwn
        extr_df = pd.concat([extr_df, pd.DataFrame({'iterator': [g_i_extr_dwn], 'g_extr_dwn': [g_extr_dwn],
                                                    'g_extr_any': [g_extr_dwn]})], ignore_index=True)

        # g_i_extr_medium = round((g_prev_i_extr + g_i_extr)/2)
        # g_extr_medium = (g_prev_extr + g_extr_dwn)/2
        # df.loc[g_i_extr_medium, 'g_extr_medium'] = g_extr_medium
        waves_list.append(round(g_extr_dwn / (g_prev_extr / 100) - 100, 2))
        g_prev_extr = g_extr_dwn
        g_i_time += g_i_extr_dwn - g_prev_i_extr
        wave_period_list.append(g_i_extr_dwn - g_prev_i_extr)
        g_i_time_count += 1
        g_prev_i_extr = g_i_extr_dwn

        extr_dwn = g_extr_dwn
        i_extr_dwn = g_i_extr_dwn

        g_extr_dwn = df.close.loc[i] * 2
    if g_napr == 1 and df.close.loc[i] > g_extr_up:
        g_extr_up = df.close.loc[i]
        g_i_extr_up = i
    elif g_napr == -1 and df.close.loc[i] < g_extr_dwn:
        g_extr_dwn = df.close.loc[i]
        g_i_extr_dwn = i


def graph():
    global fplt
    fplt.timer_callback(realtime_update_plot, 0.01)  # update 4 раза twice every second
    print(fplt.timers)
    fplt.show()


if __name__ == "__main__":
    is_close = False # True
    is_historic = True
    go = 1
    # None
    hist_date = datetime(year=2023, month=8, day=7)
    hist_path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/%s_not_close_candle_df.csv'
    plots = {}

    # ticker = 'YNDX'
    # path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/9.8_YNDX_not_close_candle_df.csv'
    # dfc = pd.read_csv(path)
    # dfc.time = pd.to_datetime(dfc.time)

    '''if hist_path is not None:
        date = str(dt_object.day) + '.' + str(dt_object.month) + '_'
        path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/%s_not_close_candle_df.csv' % \
               (date + ticker)
        df = pd.read_csv(path)
        df.time = pd.to_datetime(df.time)
        df['iterator'] = range(0, len(df))
        return df.set_index('iterator')  # .set_index('Time')'''
    ax, ax_vol = fplt.create_plot('terminal', rows=2)
    ctrl_panel = create_ctrl_panel(ax.vb.win)
    df_of_shares = DataFrame(columns=['name', 'ticker', 'figi', 'lot', 'currency', 'class_code'])
    ticker = ctrl_panel.ticker.currentText()
    df_share = FindShareByTicker(ticker)
    df_of_shares = pd.concat([df_of_shares, df_share])
    list_of_figis = df_of_shares.figi.to_list()
    globals()['volume_df_%s' % ticker] = DataFrame({
        'price': [],
        'quantity_buy': [],
        'quantity_sale': [],
        'time': [],
        'is_consistent': [],
    })
    globals()['candle_df_%s' % ticker] = DataFrame({
        'iterator': [],
        'time': [],
        'volume': [],
        'open': [],
        'close': [],
        'high': [],
        'low': [],
        'last_trade_ts': [],
        'g_extr_up': [],
        'g_extr_dwn': [],
        'g_extr_des': [],
    })
    change_asset()

    if is_historic:
        stream_thread = Thread(target=stream_historic_candles, daemon=True)
    else:
        stream_thread = Thread(target=stream_candles, daemon=True)

    stream_thread.start()
    # stream_thread.join()

    # t_graph = Thread(target=graph, daemon=True)
    # t_graph.start()
    timer = fplt.timer_callback(realtime_update_plot, 0.01)  # update 4 раза twice every second
    fplt.show()
    """work = 0
    prev_exc = None
    prev_exc_time = datetime.now() + timedelta(hours=1)
    while True:
        try:
            with Client(token_read) as client:
                for marketdata in client.market_data_stream.market_data_stream(
                        request_iterator_candles(list_of_figis, 'subscribe'),
                        # request_iterator_order_book(list_of_figis, 'subscribe'),
                ):
                    response = unpack_MarketDataResponse(marketdata).get('list_response_clear')
                    if len(response) == 1:
                        print('===== ОТВЕТ =====\n', response[0])
                    else:
                        print('===== ОТВЕТ =====\n', response)
                    if work == 0:
                        break
        except Exception as exc:
            if (exc != prev_exc) \
                    or (prev_exc_time + timedelta(minutes=1) < datetime.now()):
                print(__file__ + ' main ' + str(type(exc)) + ' | ' + str(exc))
                exception_log = open('spectator1 data/exception_log.txt', 'a+', encoding='utf-8')
                exception_log.write(str(datetime.now()) + ' ' + str(type(exc)) + ' | ' + str(exc) + '\n')
                exception_log.close()
                prev_exc_time = datetime.now()
            continue

    fplt.timer_callback(realtime_update_plot, 0.25)  # update 4 раза twice every second
    fplt.show()"""
