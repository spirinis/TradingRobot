import os
import time

import pandas as pd
import numpy as np
import pyqtgraph as pg
from datetime import datetime, timedelta, timezone
from tinkoff.invest import (Client, GetOperationsByCursorRequest, OperationItem, Quotation, schemas, CandleInterval)
import finplot as fplt
# from tinkoff.invest.services import (OperationsService, )

token_read = os.getenv('TITOKEN_READ_ALL', 'Ключа нет')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

#  with Client(token_read) as client:
#     a = client.users.get_accounts()


def get_account():
    from tinkoff.invest.services import (UsersService)
    with Client(token_read) as client:
        accounts = client.users.get_accounts()
    for a in accounts:
        print(a)


def cast_money(v: Quotation) -> float:
    return v.units + v.nano / 1e9  # nano - 9 нулей


def create_df(candles: list) -> pd.DataFrame:
    df = pd.DataFrame([{
        'time': c.time,
        'volume': c.volume,
        'open': cast_money(c.open),
        'close': cast_money(c.close),
        'high': cast_money(c.high),
        'low': cast_money(c.low),
    } for c in candles])
    return df


def FindShareByTicker(tiker: str) -> pd.DataFrame:
    from tinkoff.invest import (InstrumentStatus, RequestError)
    from tinkoff.invest.services import (InstrumentsService, )
    try:
        with Client(token_read) as client:
            instruments: InstrumentsService = client.instruments
            # market_data: MarketDataService = client.market_data
            try:
                shares = pd.read_csv('all_shares.csv')
            except FileNotFoundError:
                shares = pd.DataFrame(
                    instruments.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE).instruments,
                    columns=['name', 'ticker', 'figi', 'lot', 'currency', 'class_code']
                )
                shares.to_csv('all_shares.csv', index=False)
            find_share = shares[shares['ticker'] == tiker]
            # r = r[r['ticker'] == tiker]['figi'].iloc[0]
            if find_share.empty:
                raise Exception('Нет такого тикера')

            return find_share
    except RequestError as error:
        print('ошибка find' + "\n" + str(error))


def turnover_summ():  # оборот суммарный
    turnover = 0
    dt_object = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    frm = dt_object - timedelta(days=2)
    to = datetime.utcnow()
    with Client(token_read) as client:
        operations = client.operations.get_operations_by_cursor(GetOperationsByCursorRequest(
            account_id='2184550636',
            from_=frm,
            to=to,
            state=schemas.OperationState.OPERATION_STATE_EXECUTED,
            without_trades=False
        ))
    operations_df = pd.DataFrame([{
        'name': operation.name,
        'time': operation.date,
        'type': operation.type.name,
        'description': operation.description,
        'state': operation.state.name,
        'figi': operation.figi,
        'payment': cast_money(operation.payment),
        'price': cast_money(operation.price),
        'commission': cast_money(operation.commission),
        'yield_': cast_money(operation.yield_),
        'yield_relative': cast_money(operation.yield_relative),
        'accrued_int': operation.accrued_int,
        'quantity': operation.quantity,
        'quantity_rest': operation.quantity_rest,
        'quantity_done': operation.quantity_done,
        'cancel_date_time': operation.cancel_date_time,
        'cancel_reason': operation.cancel_reason
    } for operation in operations.items])
    for i in range(len(operations_df)):
        if operations_df.type[i] == schemas.OperationType.OPERATION_TYPE_BUY.name\
                or operations_df.type[i] == schemas.OperationType.OPERATION_TYPE_SELL.name:
            turnover += abs(operations_df.payment[i])
    print(turnover)


def download_candles(ticker, dt_string='', days_before=3):  # 829 свечей
    from tinkoff.invest import (RequestError, )
    share = FindShareByTicker(ticker)
    figi = share.figi.iloc[0]  # 'BBG006L8G4H1'
    if dt_string != '':
        time_format = "%Y %m %d %H %M"
        time_utc_now = datetime.strptime(dt_string, time_format) - timedelta(hours=3)
    else:
        time_utc_now = datetime.utcnow()
    dt_object = time_utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
    """if time_utc_now.hour < 21:
        dt_object = time_utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        dt_object = (time_utc_now+timedelta(hours=3)).replace(hour=0, minute=0, second=0, microsecond=0)"""
    print(dt_object)
    print(time_utc_now)
    weekday_now = dt_object.weekday()
    all_candles = []
    j = 0
    one = 0
    one_time = 0
    two = 0
    three = 0
    if weekday_now == 0:  # ПН
        two = 1
        three = 1
        one_time = time_utc_now - timedelta(hours=-3)
    elif weekday_now == 1:  # ВТ
        two = -1
        three = 1
        one_time = time_utc_now - timedelta(hours=-3)
    elif weekday_now == 2:  # СР
        two = -1
        three = -1
        one_time = time_utc_now - timedelta(hours=-3)
    elif weekday_now == 3:  # ЧТ
        two = -1
        three = -1
        one_time = time_utc_now - timedelta(hours=-3)
    elif weekday_now == 4:  # ПТ
        two = -1
        three = -1
        one_time = time_utc_now - timedelta(hours=-3)
    elif weekday_now == 5:  # СБ
        two = 0
        three = 0
        one_time = dt_object
        one = 1
    elif weekday_now == 6:  # ВС
        two = 1
        three = 1
        one_time = dt_object - timedelta(days=1)
        one = 2
    with Client(token_read) as client:
        for i in range(days_before):
            # TODO if datetime.now().date() != time_utc_now.date():
            if (time_utc_now - timedelta(days=i + j)).weekday() > 4:
                j += 1

            if i == 0:
                frm = dt_object - timedelta(days=one)
                t = one_time
            elif i == 1:
                frm = dt_object - timedelta(days=i + two + 1)
                t = dt_object - timedelta(days=i + two)
            elif i == 2:
                frm = dt_object - timedelta(days=i + three + 1)
                t = dt_object - timedelta(days=i + three)
            print(i, ':= ', frm, ' -> ', t)
            candles = client.market_data.get_candles(
                                    figi=figi,
                                    from_=frm,
                                    to=t,
                                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                                )
            print(candles.candles[0].time, ' -> ', candles.candles[-1].time)
            all_candles = candles.candles + all_candles
    globals()['candle_df_%s' % ticker] = create_df(all_candles)
    ax = fplt.create_plot(ticker, rows=1)
    fplt.candlestick_ochl(globals()['candle_df_%s' % ticker][['time', 'open', 'close', 'high', 'low']], ax=ax)
    fplt.volume_ocv(globals()['candle_df_%s' % ticker][['time', 'open', 'close', 'volume']], ax=ax.overlay())
    fplt.show()
    date_ticker_str = str((time_utc_now - timedelta(days=days_before-1)).day) + '-' + str(time_utc_now.day) + '.' +\
        str(time_utc_now.month) + '.' + str(time_utc_now.year) + '_' + ticker
    if input('Сохранить массив? ') == '':
        path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/%s_candle_df.csv' % date_ticker_str
        globals()['candle_df_%s' % ticker].to_csv(path, index=False)


def download_dofiga_candles(ticker, to_dt_string='', days_before=3, except_day=''):  # 829 свечей # +2 дня?
    from tinkoff.invest import (RequestError, )
    share = FindShareByTicker(ticker)
    figi = share.figi.iloc[0]  # 'BBG006L8G4H1'
    if to_dt_string != '':
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
        time_utc_now = datetime.strptime(to_dt_string, time_format) - timedelta(minutes=1)  # - timedelta(hours=3)
    else:
        time_utc_now = datetime.utcnow()
    dt_object = time_utc_now
    if except_day != '':
        if except_day.count(' ') == 4:  # "%Y %m %d %H %M"
            time_format = "%Y %m %d %H %M"
        elif except_day.count(' ') == 2:  # "%Y %m %d"
            time_format = "%Y %m %d"
            except_day = except_day
        elif except_day.count(' ') == 3:   # "%m %d %H %M"
            time_format = "%Y %m %d %H %M"
            except_day = str(datetime.today().year) + ' ' + except_day
        else:   # "%m %d"
            time_format = "%Y %m %d"
            except_day = str(datetime.today().year) + ' ' + except_day
        except_day = datetime.strptime(except_day, time_format)  # - timedelta(hours=3)
    # dt_object = time_utc_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(minutes=1)
    """if time_utc_now.hour < 21:
        dt_object = time_utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        dt_object = (time_utc_now+timedelta(hours=3)).replace(hour=0, minute=0, second=0, microsecond=0)"""
    all_candles = []
    print(dt_object)
    print(time_utc_now)
    for i in range(days_before):
        """weekday_now = (dt_object - timedelta(hours=12)).weekday()
        if dt_object.date() == except_day.date():
            dt_object = dt_object - timedelta(days=1)
            weekday_now = (dt_object - timedelta(hours=12)).weekday()
        if weekday_now > 4:
            dt_object = dt_object - timedelta(days=1)
            # continue"""
        while (except_day != '' and dt_object.date() == except_day.date()) or ((dt_object - timedelta(hours=12)).weekday() > 4):
            dt_object = dt_object - timedelta(days=1)
        with Client(token_read) as client:
            frm = dt_object - timedelta(days=1)
            t = dt_object
            print(i, ':= ', frm, ' -> ', t)
            candles = client.market_data.get_candles(
                                    figi=figi,
                                    from_=frm,
                                    to=t,
                                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                                )
        print(candles.candles[0].time, ' -> ', candles.candles[-1].time)
        all_candles = candles.candles + all_candles
        dt_object = dt_object - timedelta(days=1)
    globals()['candle_df_%s' % ticker] = create_df(all_candles)
    dt_object = dt_object + timedelta(days=1)
    date_ticker_str = \
        str(dt_object.day) + '.'\
        + str(dt_object.month) + '-'\
        + str(time_utc_now.day) + '.'\
        + str(time_utc_now.month) + '.'\
        + str(time_utc_now.year) + '_' + ticker
    ax, ax2 = fplt.create_plot(date_ticker_str, rows=2)
    fplt.candlestick_ochl(globals()['candle_df_%s' % ticker][['time', 'open', 'close', 'high', 'low']], ax=ax)
    fplt.volume_ocv(globals()['candle_df_%s' % ticker][['time', 'open', 'close', 'volume']], ax=ax2)
    fplt.show()
    if input('enter если охранить массив? ') == '':
        path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/%s_candle_df.csv' % date_ticker_str
        print(path)
        globals()['candle_df_%s' % ticker].to_csv(path, index=False)


def open_not_close_candle_df(path=None):
    # open_not_close_candle_df('C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/9.8_YNDX_not_close_candle_df.csv')
    """две линии
    ax.price_line = pg.InfiniteLine(angle=90, movable=False, pen=fplt._makepen('#303030', style='.'))
    ax.price_line.setPos(1000)
    ax.price_line.pen.setColor(pg.mkColor('#30f030'))
    ax.addItem(ax.price_line, ignoreBounds=True)
    ax.price_line = pg.InfiniteLine(angle=90, movable=False, pen=fplt._makepen('#303030', style='.'))
    ax.price_line.setPos(2000)
    ax.price_line.pen.setColor(pg.mkColor('#30f030'))
    ax.addItem(ax.price_line, ignoreBounds=True)
    в жопу пошел. одна
    ax.time_line = pg.InfiniteLine(angle=90, movable=False, pen=fplt._makepen('#303030', style='.'))
    ax.time_line.setPos(1000)
    ax.time_line.pen.setColor(pg.mkColor('#30f030'))
    ax.addItem(ax.time_line, ignoreBounds=True)
    ax.time_line = pg.InfiniteLine(angle=90, movable=False, pen=fplt._makepen('#fff', style='.'))
    ax.time_line.setPos(2000)
    ax.time_line.pen.setColor(pg.mkColor('#303030'))
    ax.addItem(ax.time_line, ignoreBounds=True)"""
    if path is None:
        return 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/%s_not_close_candle_df.csv'
    global df
    df = pd.read_csv(path)
    df.time = pd.to_datetime(df.time)

    ax, ax2 = fplt.create_plot(rows=2, title=str.split(path, '/')[-1])

    if 'last_trade_ts' in df.columns:
        df.last_trade_ts = pd.to_datetime(df.last_trade_ts)
        if 'iterator' not in df.columns:
            df['iterator'] = range(0, len(df))
        fplt.candlestick_ochl(df[['iterator', 'open', 'close', 'high', 'low']], ax=ax)
        fplt.plot(df['iterator'], df['close'], legend='close', ax=ax, color='white', width=2)
        fplt.volume_ocv(df[['iterator', 'open', 'close', 'volume']], ax=ax2)  # .overlay())
        if 'buy' in df.columns:
            fplt.plot(df['iterator'], df['buy'], legend='buy', ax=ax, color='blue', width=2, style='^')
        if 'sale' in df.columns:
            fplt.plot(df['iterator'], df['sale'], legend='sale', ax=ax, color='red', width=2, style='v')
        if 'bad_sale' in df.columns:
            fplt.plot(df['iterator'], df['bad_sale'], legend='bad_sale', ax=ax, color='#FF00FF', width=2, style='v')
        if 'g_extr_up' in df.columns:
            fplt.plot(df['iterator'], df['g_extr_up'], legend='g_extr_up', ax=ax, color='#ff6666', width=1, style='o')
        if 'g_extr_dwn' in df.columns:
            fplt.plot(df['iterator'], df['g_extr_dwn'], legend='g_extr_dwn', ax=ax, color='#6666ff', width=1, style='o')
        if 'g_extr_des' in df.columns:
            fplt.plot(df['iterator'], df['g_extr_des'], legend='g_extr_des', ax=ax, color='gray', width=1, style='o')
    else:
        fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
        # fplt.plot(df['time'], df['close'], legend='close', ax=ax, color='white', width=2)
        fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax2)  # ax.overlay())
    fplt.show()


def draw_df(df: pd.DataFrame):
    ax, ax2 = fplt.create_plot(rows=2)
    fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
    fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax2)
    fplt.show()


def cut_df(dt_string: str, df: pd.DataFrame, ) -> pd.DataFrame:
    '''
    нужно несколько итераций, пока ничего не выведет
    вывод:
    2432
2433
2434
2435
2436
2437
2438
2439

2432
2433
2434
2435

2432
2433

2432

2432
    '''
    path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/%s_not_close_candle_df.csv'
    time_format = "%m %d"
    # time_utc_now = datetime.strptime(to_dt_string, time_format) - timedelta(hours=3)
    find = False
    j = 0
    time_from = datetime.strptime(dt_string + '.2023', '%d.%m.%Y').replace(tzinfo=timezone.utc)
    time_to = time_from + timedelta(days=1)
    for i in range(len(df)):
        if (df.time.iloc[i] > time_from) and (df.time.iloc[i] < time_to):
            df = df.drop(i)
            j += 1
            find = True
            print(i)
        if i == len(df) - j:
            break
    # df.to_csv(path % (frm + '-' + t + '_' + ticker), index=False)
    return df


def average_day_volume(rdf: pd.DataFrame = '', future=False):  # 829 свечей в день
    if type(rdf) != pd.DataFrame:
        global df
        df = pd.read_csv(
            'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/7.6-7.7.2023_YNDX_candle_df2.csv')
        df.time = pd.to_datetime(df.time)
        rdf = df
    global sdf
    sdf = pd.DataFrame(columns=['time', 'volume'])
    '''for i in range(830):  # 829 минутных свечей в дне
        if (i < 520):
            sdf.loc[i, 'time'] = datetime.strptime('7 0', '%H %M').replace(tzinfo=timezone.utc) + timedelta(
                minutes=i)
    for i in range(830):  # 829 минутных свечей в дне
        if (i > 524) and (i < 530):
            sdf.loc[i, 'time'] = datetime.strptime('7 0', '%H %M').replace(tzinfo=timezone.utc) + timedelta(
                minutes=i)
    for i in range(830):  # 829 минутных свечей в дне
        if (i > 543):
            sdf.loc[i, 'time'] = datetime.strptime('7 0', '%H %M').replace(tzinfo=timezone.utc) + timedelta(
                minutes=i)
    sdf.reset_index(drop=True, inplace=True)'''
    for i in range(len(rdf)):
        if not sdf.isin([rdf.time[i].time()]).time.any():
            sdf = pd.concat([sdf, pd.Series(
                data={'time': rdf.time[i].time(), 'volume': rdf.volume[i], 'cnt': 1}).to_frame().T],
                            ignore_index=True)
        else:
            j = sdf.index[sdf['time'] == rdf.time[i].time()].tolist()[0]
            sdf.loc[j, 'volume'] += rdf.volume[i]
            sdf.loc[j, 'cnt'] += 1
    sdf['tt'] = 1
    sdf['ma10'] = np.nan
    sdf['ma50'] = np.nan
    l10: int = len(sdf) - 10
    l25: int = len(sdf) - 25
    for i in range(len(sdf)):
        sdf.loc[i, 'tt'] = pd.to_datetime((datetime.fromisoformat('2000-01-01') + timedelta(
            hours=sdf.time[i].hour, minutes=sdf.time[i].minute)), utc=True)
        sdf.loc[i, 'volume'] /= sdf.cnt[i]
    for i in range(len(sdf)):
        if future:
            if i < 10:
                sdf.loc[i, 'ma10'] = sum(sdf.volume[0:i + 1]) / (i + 1)
            elif i > l10:
                sdf.loc[i, 'ma10'] = sum(sdf.volume[i:len(sdf) + 1]) / (len(sdf) - i)
            else:
                sdf.loc[i, 'ma10'] = sum(sdf.volume[(i - 5):(i + 4)]) / 10
            if i < 25:
                sdf.loc[i, 'ma50'] = sum(sdf.volume[0:i + 1]) / (i + 1)
            elif i > l25:
                sdf.loc[i, 'ma50'] = sum(sdf.volume[i:len(sdf) + 1]) / (len(sdf) - i)
            else:
                sdf.loc[i, 'ma50'] = sum(sdf.volume[(i - 25):(i + 25)]) / 50
        else:
            if i >= 10:
                sdf.loc[i, 'ma10'] = sum(sdf.volume[i - 10:i]) / 10
            else:
                sdf.loc[i, 'ma10'] = sum(sdf.volume[0:i + 1]) / (i + 1)
            if i >= 50:
                sdf.loc[i, 'ma50'] = sum(sdf.volume[i - 50:i]) / 50
            else:
                sdf.loc[i, 'ma50'] = sum(sdf.volume[0:i+1]) / (i+1)

        '''if (i >= 25) and (i <= l):
            sdf.loc[i, 'ma50'] = sum(sdf.volume[i - 25:i+25]) / 25
        else:
            sdf.loc[i, 'ma50'] = sum(sdf.volume[0:i+1]) / (i+1)
        else:
            sdf.loc[i, 'ma50'] = sum(sdf.volume[l:i+1]) / (len(sdf) - i)'''
    '''for i in range(len(df)):
        if df.time.iloc[i] > prev_time:
            volume_df.append(df.volume.iloc[i])
            j += 1
        if (df.time.iloc[i] > time_from) and (df.time.iloc[i] < time_to):
            df = df.drop(i)
            j += 1
            find = True
            print(i)
        prev_time = df.time.iloc[i]'''
    '''for i in range(len(df3)):
    if df.time[i].time() == df3.time[i].time():
        df3.loc[i, 'volume'] += df.v[i]
    if df2.time[i].time() == df3.time[i].time():
        df3.loc[i, 'volume'] += df2.v[i]
for i in range(len(sdf)):
    if rdf.time[i].time() == sdf.time[i].time():
        sdf.loc[i, 'volume'] += rdf.volume[i]
    else:
        print(i)
        '''
    print(sdf[20:25])
    sdf.volume = sdf.volume.astype(float)
    sdf.tt = sdf.tt.astype(np.datetime64)
    print(sdf.dtypes)
    print(sdf[20:25])
    ax = fplt.create_plot('Day_volume', rows=1)
    fplt.plot(sdf['tt'], sdf['volume'], ax=ax)
    fplt.plot(sdf['tt'], sdf['ma10'], legend='ma10', ax=ax, color='red', width=2)
    fplt.plot(sdf['tt'], sdf['ma50'], legend='ma50', ax=ax, color='yellow', width=2)
    fplt.show()
    ticker = 'YNDX'
    if input('enter если охранить массив? ') == '':
        if future:
            path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/volume/%s_average_day_volume_future_df.csv'
        else:
            path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/volume/%s_average_day_volume_df.csv'
        sdf.to_csv(path % (ticker), index=False)
    return sdf


def open_average_day_volume(rdf: pd.DataFrame = '', ticker='YNDX'):  # 829 свечей в день
    # average_day_volume_df.loc[:, 'cma50'] = sdf.ma50
    if type(rdf) != pd.DataFrame:
        global average_day_volume_df
        path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/volume/%s_average_day_volume_df.csv'
        average_day_volume_df = pd.read_csv(path % ticker)
        average_day_volume_df.tt = pd.to_datetime(average_day_volume_df.tt)
        rdf = average_day_volume_df
    ax = fplt.create_plot('%s_average_day_volume' % ticker, rows=1)
    fplt.plot(rdf['tt'], rdf['volume'], ax=ax)
    fplt.plot(rdf['tt'], rdf['ma10'], legend='ma10', ax=ax, color='red', width=2)
    fplt.plot(rdf['tt'], rdf['ma50'], legend='ma50', ax=ax, color='yellow', width=2)
    if 'cma10' in rdf.columns:
        fplt.plot(rdf['tt'], rdf['cma10'], legend='cma10', ax=ax, color='blue', width=2)
        fplt.plot(rdf['tt'], rdf['cma50'], legend='cma50', ax=ax, color='green', width=2)
    fplt.show()


def buyout_time(dt_string, volume, path=''):
    """
    :param dt_string: '%H %M'
    :param volume: в лотах
    :param path:
    """
    import math
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    global indf
    global df
    '''    for i in range(0, len(df) + 1, 40):  # five 1703 - 1722.5 yndx 2474 - 2483.8 ozon 2045 - 2067
        if df.time[i]
            df = volume_df_OZON[i:i + 40]
            t = t - timedelta(seconds=t.second, microseconds=t.microsecond)'''
    is_big = 3000000
    i_list = []
    if path == '':
        path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/13.07_YNDX_volume_df.csv'
        # 7:34-9:17
    indf = pd.read_csv(path)
    indf.time = pd.to_datetime(indf.time)
    time = datetime.strptime(indf.time[0].date().strftime('%Y %m %d') + ' ' + dt_string,
                             '%Y %m %d %H %M').replace(tzinfo=timezone.utc)
    for i in range(40, len(indf)+1, 40):
        if time <= indf.time[i]:
            while (indf.time[i] >= time - timedelta(minutes=1)) and (i >= 40):
                i_list.append(i)
                i -= 40
            i = i_list[0]
            break
    i_list.reverse()
    print(len(i_list), 'стаканов')
    for j in i_list[::math.ceil(len(i_list)/20)]:  # TODO: 20?
        df = indf.iloc[j:j + 40]
        df.reset_index(drop=True, inplace=True)
        start_price = df.price[20]
        out_str = ''
        buy = 0
        sale = 0
        buy_flag = False
        sale_flag = False
        for i in range(int(len(df) / 2)):
            buy += df.quantity_buy[i]
            sale += df.quantity_sale[i + 20]
            #  если первый, то в эту сторону сложнее двигаться, нужно больше объёмов
            #  большие объёмы увеличивают конкуренцию, эта сторона поддаётся
            if (buy >= volume) and not buy_flag:  # покупатели ставят цену выше, выпродать глубоко сложнее
                current_price = df.price[i]
                if sale_flag:
                    out_str = 'проще выпродать в ' + str(round((current_price-start_price)/start_price*100, 2)) + '% '\
                              + str(current_price) + ' ' + out_str
                else:
                    out_str += ', сложнее выпродать в ' + str(round((current_price - start_price)/start_price * 100, 2))\
                               + '% ' + str(current_price) + ' '
                buy_flag = True
            if (sale >= volume) and not sale_flag:  # продавцы ставят цену ниже, выкупить глубоко сложнее
                current_price = df.price[i + 20]
                if buy_flag:
                    out_str = 'проще выкупить в +' + str(round((current_price-start_price)/start_price*100, 2)) + '% '\
                               + str(current_price) + ' ' + out_str
                else:
                    out_str += ', сложнее выкупить в +' + str(round((current_price-start_price)/start_price*100, 2))\
                               + '% ' + str(current_price) + ' '
                sale_flag = True
            # покупателей больше. Будут выставлять цену всё выше
            '''cost_buy = df.quantity_buy[i] * df.price[i]
            cost_sale = df.quantity_sale[i + 20] * df.price[i + 20]
            if cost_buy >= is_big:
                print('плита на покупку', int(df.quantity_buy[i]),  'по', df.price[i],
                      '=', round(cost_buy/10**6, 2), 'M')
            if cost_sale >= is_big:
                print('плита на продажу', int(df.quantity_sale[i + 20]),  'по', df.price[i + 20],
                      '=', round(cost_sale/10**6, 2), 'M')'''

        if (not buy_flag) and (not sale_flag):
            out_str = 'Объём великоват, cтакана 20 не хватило.'
        elif not buy_flag:
            out_str = 'ТОРПЕДА ВНИЗ. Стакана 20 не хватило' + out_str
        elif not sale_flag:
            out_str = 'РАКЕТА ВВЕРХ. Стакана 20 не хватило' + out_str
        print('Начальная ' + str(start_price), out_str, 'Покупают лотов', buy, 'Продают лотов', sale)


def concat_dfs(frm, t, ticker):
    '''t, frm = day.month'''
    global df
    global result_df
    frm = str(frm)
    t = str(t)
    path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/%s_not_close_candle_df.csv'
    result_df = pd.DataFrame()
    for month in range(int(frm.split('.')[1]), int(t.split('.')[1]) + 1):
        for day in range(int(frm.split('.')[0]), int(t.split('.')[0]) + 1):
            try:
                df = pd.read_csv(path % (str(day) + '.' + str(month) + '_' + ticker))
            except FileNotFoundError:
                continue
            result_df = pd.concat([result_df, df])
    res_path = path % (frm + '-' + t + '_' + ticker)
    try:
        result_df.to_csv(path % (frm + '-' + t + '_' + ticker), index=False)
    except PermissionError:
        res_path += '[2] debil'
        result_df.to_csv(res_path, index=False)
    print(res_path)


def glasses_difference(i, time, step=1,):
    global dfv, df, dfv1, dfv2
    '''df = pd.read_csv('C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/13.07_YNDX_volume_df.csv')
dfc = pd.read_csv('C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/13.7_YNDX_not_close_candle_df.csv')
dfc.time = pd.to_datetime(dfc.time)
df.time = pd.to_datetime(df.time)
dfc.last_trade_ts = pd.to_datetime(dfc.last_trade_ts)
step = 150
dfvv2 = df.iloc[i + 40*step:i + 40*step + 40]
glasses_difference(122800,150)'''
    # сумма стаканов
    '''j = 0
for i in range(round(len(df)/40)):
    step = i*40
    while dfc.last_trade_ts[j] < df.time[step]:
        j += 1
    dfc.loc[j, 'qb2'] = df.quantity_buy[0+i*40:2+i*40].sum()
    dfc.loc[j, 'qs2'] = df.quantity_sale[20+i*40:22+i*40].sum()
    dfc.loc[j, 'qb5'] = df.quantity_buy[0 + i * 40:5 + i * 40].sum()
    dfc.loc[j, 'qs5'] = df.quantity_sale[20 + i * 40:25 + i * 40].sum()
    
ax, ax2, ax3, ax4 = fplt.create_plot(rows=4)
dfc['iterator'] = range(0, len(dfc))
fplt.candlestick_ochl(dfc[['iterator', 'open', 'close', 'high', 'low']], ax=ax)
fplt.plot(dfc['iterator'], dfc['close'], legend='close', ax=ax, color='white', width=2)
fplt.volume_ocv(dfc[['iterator', 'open', 'close', 'volume']], ax=ax2)  # .overlay())
fplt.plot(dfc['iterator'], dfc['qb5'], legend='qb5', ax=ax3, color='green', style='o')
fplt.plot(dfc['iterator'], dfc['qs5'], legend='qs5', ax=ax3, color='red',style='o')
fplt.plot(dfc['iterator'], dfc['qb2'], legend='qb2', ax=ax4, color='green', style='o')
fplt.plot(dfc['iterator'], dfc['qs2'], legend='qs2', ax=ax4, color='red',style='o')
fplt.show()'''
    # Скорость
    '''for i in range(1, len(dfc)):
    cdif = dfc.close[i] - dfc.close[i - 1]
    if cdif != 0:
        dfc.loc[i, 'speed'] = cdif / (dfc.last_trade_ts[i] - dfc.last_trade_ts[i - 1]).microseconds * 10 ** 6
    else:
        dfc.loc[i, 'speed'] = 0
    if dfc.speed[i] >20:
        dfc.loc[i, 'speed'] = 20
    elif dfc.speed[i] < -20:
        dfc.loc[i, 'speed'] = -20'''
    # размер свечей ОТДЕЛЬНО туду
    '''for i in range(1, len(df)):
    cdif = df.high[i] - df.low[i]
    if df.close[i] >= df.open[i]:
        df.loc[i, 'gc'] = cdif
    elif df.close[i] < df.open[i]:
        df.loc[i, 'rc'] = abs(cdif)
    df.loc[i, 'c'] = cdif
ax, ax2, ax3 = fplt.create_plot(rows=3, title='4')
fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
fplt.plot(df['time'], df['high'], legend='high', ax=ax, color='white', width=2)
fplt.plot(df['time'], df['low'], legend='low', ax=ax, color='white', width=2)
fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax2)  # .overlay())
fplt.plot(df['time'], df['gc'], legend='gc', ax=ax3, color='green',style='o')
fplt.plot(df['time'], df['rc'], legend='rc', ax=ax3, color='red',style='o')
fplt.plot(df['time'], df['c'], legend='c', ax=ax3, color='white',)
fplt.show()'''
    # выборка стаканов за минуту
    i_list = []
    # time = dfc.time.loc[1000]
    for i in range(40, len(df) + 1, 40):
        if time + timedelta(minutes=1) <= df.time[i]:
            while (df.time[i] >= time) and (i >= 40):
                i_list.append(i)
                i -= 40
            i = i_list[0]
            break
    i_list.reverse()
    # сравнение стаканов
    dfv1 = df.iloc[i:i + 40]
    dfv2 = df.iloc[i + 40*step:i + 40*step + 40]
    dfv1 = dfv1.set_index('price')
    dfv2 = dfv2.set_index('price')
    dfv = dfv2 - dfv1
    for i in dfv.index:
        if dfv.time[i] is pd.NaT:
            if i in dfv1.index:
                dfv.loc[i] = dfv1.loc[i]
                dfv.loc[i, 'quantity_buy'] *= -1
                dfv.loc[i, 'quantity_sale'] *= -1
            else:
                dfv.loc[i] = dfv2.loc[i]


def lines():
    global df
    df = pd.read_csv(
        'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/5.10-7.10_YNDX_candle_df.csv')
    df.time = pd.to_datetime(df.time)

    def approximation(x_values, y_values):
        average_x = 0
        average_y = 0
        average_xy = 0
        average_xx = 0
        n = len(x_values)
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
        return a, b,

    percent_extr = 0.10 / 100  # 0.10 / 100
    extr_up = 0
    extr_up_count = 0
    extr_dwn = df.high.loc[0]
    prev_extr = df.high.loc[0]
    price = df.high.loc[0]
    i_extr = 0
    buy = df.high.loc[0]
    i_buy = 0
    prev_i_extr = 0
    prev_buy_extr = 0
    napr = 1
    from_i = 1 # 1
    to_i = len(df)  # 216 len(df)
    for i in range(from_i, to_i):
        df.loc[i, 'medium'] = (df.high[i] + df.low[i]) / 2
        prev_price = price
        price = df.medium.loc[i]  # df.medium.loc[i]
        if napr == 1 and df.high[i] * (1 + percent_extr) < extr_up:
            if (extr_up_count == 0) and (i_extr > i_buy) and (df.medium.loc[i_extr] < buy * 1.0009):
                extr_up_count = 1
            elif (extr_up_count > 0) and (df.medium.loc[i_extr] < prev_buy_extr):
                extr_up_count += 1
            else:
                extr_up_count = 0
            napr = -1
            df.loc[i_extr, 'extr_up'] = extr_up
            df.loc[i + 1, 'extr_des'] = extr_up

            #extr_df = pd.concat([extr_df, pd.DataFrame({'time': [df.time.loc[i_extr]], 'extr_up': [extr_up]})],
            #                    ignore_index=True)
            i_extr_medium = round((prev_i_extr + i_extr) / 2)
            extr_medium = (prev_extr + extr_up) / 2
            df.loc[i_extr_medium, 'extr_medium'] = extr_medium
            prev_extr = extr_up
            prev_buy_extr = extr_up
            prev_i_extr = i_extr

            extr_up = 0
        elif napr == -1 and df.low[i] * (1 - percent_extr) > extr_dwn:
            napr = 1
            df.loc[i_extr, 'extr_dwn'] = extr_dwn
            df.loc[i + 1, 'extr_des'] = extr_dwn

            #extr_df = pd.concat([extr_df, pd.DataFrame({'time': [df.time.loc[i_extr]], 'extr_dwn': [extr_dwn]})],
            #                    ignore_index=True)
            #i_extr_medium = round((prev_i_extr + i_extr) / 2)
            # extr_medium = (prev_extr + extr_dwn)/2
            # df.loc[i_extr_medium, 'extr_medium'] = extr_medium
            prev_extr = extr_dwn
            prev_i_extr = i_extr

            extr_dwn = price * 2
        if napr == 1 and df.high[i] > extr_up:
            extr_up = df.high[i]
            i_extr = i
        elif napr == -1 and df.low[i] < extr_dwn:
            extr_dwn = df.low[i]
            i_extr = i

    extr_df_to_list = pd.concat([df[['time', 'extr_up']].dropna(), df[['time', 'extr_dwn']].dropna()])
    extr_df_to_list.fillna(0, inplace=True)
    extr_df_to_list.extr_up = extr_df_to_list.extr_up + extr_df_to_list.extr_dwn
    extr_df_to_list.drop(['extr_dwn'], axis=1, inplace=True)
    extr_df_to_list.sort_index(inplace=True)
    x_list = extr_df_to_list.index.to_list()
    y_list = extr_df_to_list.extr_up.to_list()
    points_count = 30
    line = approximation(x_list[-points_count:], y_list[-points_count:])
    a = line[0]         # [1, 70, 140, 210], [2460, 2450, 2440, 2430]
    b = line[1]
    x = [x_list[-points_count], x_list[-1]]
    y = [a*x[0]+b, a*x[1]+b]
    print('a =', a, 'b =', b)
    print(x)
    print(y)

    ax, ax2 = fplt.create_plot(rows=2, title='4')
    fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
    # fplt.plot(df['time'], df['high'], legend='high', ax=ax, color='white', width=2)
    # fplt.plot(df['time'], df['low'], legend='low', ax=ax, color='white', width=2)
    fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax2)  # .overlay())
    fplt.plot(df['time'], df['extr_up'], legend='extr_up', ax=ax, color='#ff6666', width=1, style='o')
    fplt.plot(df['time'], df['extr_dwn'], legend='extr_dwn', ax=ax, color='#6666ff', width=1, style='o')
    # fplt.plot(df['time'], df['extr_des'], legend='extr_des', ax=ax, color='gray', width=1, style='o')
    fplt.plot(df['time'], df['extr_medium'], legend='extr_medium', ax=ax, color='yellow', width=1, style='o')
    fplt.add_line((x[0], y[0]), (x[1], y[1]), color='#993', ax=ax) # , interactive=True)
    # fplt.add_text((x[1], y[1]), 'text', color='#993', ax=ax)
    fplt.show()


def ma_sma():
    df.loc[0, 'bp'] = 0
    df.loc[0, 'sp'] = 0
    for i in range(1, len(df)):
        p = df.close[i] - df.close[i - 1]
        if p >= 0:
            df.loc[i, 'bp'] = p
            df.loc[i, 'sp'] = df.sp[i - 1]
        else:
            df.loc[i, 'sp'] = abs(p)
            df.loc[i, 'bp'] = df.bp[i - 1]
    for i in range(1, len(df)):
        if i >= 5:
            df.loc[i, 'ma5bp'] = sum(df.bp[i - 5:i]) / 5
            df.loc[i, 'ma5sp'] = sum(df.sp[i - 5:i]) / 5
        if i >= 10:
            df.loc[i, 'ma10bp'] = sum(df.bp[i - 10:i]) / 10
            df.loc[i, 'ma10sp'] = sum(df.sp[i - 10:i]) / 10
        df.loc[i, 'smabp'] = sum(df.bp[0:i]) / i
        df.loc[i, 'smasp'] = sum(df.sp[0:i]) / i
    ax, ax2, ax3, ax4 = fplt.create_plot(rows=4, title='4')
    fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
    fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax2)  # .overlay())
    fplt.plot(df['time'], df['ma5bp'], legend='ma5bp', ax=ax3, color='green', )
    fplt.plot(df['time'], df['ma5sp'], legend='ma5sp', ax=ax3, color='red', )
    fplt.plot(df['time'], df['sma5bp'], legend='sma5bp', ax=ax4, color='green', )
    fplt.plot(df['time'], df['sma5sp'], legend='sma5sp', ax=ax4, color='red', )
    fplt.plot(df['time'], df['ma10bp'], legend='ma10bp', ax=ax3, color='#008000', )
    fplt.plot(df['time'], df['ma10sp'], legend='ma10sp', ax=ax3, color='#800000', )
    fplt.plot(df['time'], df['sma10bp'], legend='sma10bp', ax=ax4, color='#008000', )
    fplt.plot(df['time'], df['sma10sp'], legend='sma10sp', ax=ax4, color='#800000', )
    fplt.show()


def off():
    import os
    import winsound
    for i in range(10):
        winsound.Beep(300 + 60 * i, 500)
        time.sleep(1)
    winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    shutdown_command = "shutdown /s /t 00"
    os.system(shutdown_command)
    shutdown = "Нет"
    if shutdown == "Нет":
        exit()


class Line:
    def __init__(self,):
        self.average_x = 0
        self.average_y = 0
        self.average_xy = 0
        self.average_xx = 0

        global tilt_line
        point1, point2, k, b = approximation(data['price'].iterator.to_list()[-10:],
                                             data['price'].close.to_list()[-10:])
        tilt_line = fplt.add_line(point1, point2, color='#993', ax=ax, width=2)  # , interactive=True)

        global extr_up_line
        extr_up_df = extr_df[['iterator', 'g_extr_up']].dropna()
        if extr_up_df.__len__() > 1:
            point1, point2, k, b = approximation(extr_up_df.iterator.to_list()[-2:],
                                                 extr_up_df.g_extr_up.to_list()[-2:])
            extr_up_line = fplt.add_line(point1, point2, color='#f66', ax=ax, width=2)  # , interactive=True)
        global extr_dwn_line
        extr_dwn_df = extr_df[['iterator', 'g_extr_dwn']].dropna()
        if extr_dwn_df.__len__() > 1:
            point1, point2, k, b = approximation(extr_dwn_df.iterator.to_list()[-2:],
                                                 extr_dwn_df.g_extr_dwn.to_list()[-2:])
            extr_dwn_line = fplt.add_line(point1, point2, color='#66f', ax=ax, width=2)  # , interactive=True)

    def approximation(self, x_values, y_values):
        n = len(x_values)
        for i in range(n):
            self.average_x += x_values[i]
            self.average_y += y_values[i]
            self.average_xy += x_values[i] * y_values[i]
            self.average_xx += x_values[i] ** 2
        average_x = average_x / n
        average_y = average_y / n
        average_xy = average_xy / n
        average_xx = average_xx / n
        d_x = average_xx - average_x ** 2
        k = (average_xy - average_x * average_y) / d_x
        b = average_y - k * average_x
        x = [x_values[0], x_values[-1]]
        y = [k * x[0] + b, k * x[1] + b]
        point1 = [x[0], y[0]]
        point2 = [x[1], y[1]]
        return point1, point2, k, b

    average_x += -1 / 3 + 4 / 3
    average_y += -1 / 3 + 4 / 3
    average_xy += -(1 * 1) / 3 + (4 * 4) / 3
    average_xx += -(1 ** 2) / 3 + (4 ** 2) / 3

if __name__ == "__main__":
    print("__main__")
    # lines()
    # turnover_summ()
    # download_candles('YNDX', to_dt_string="", days_before=3)  # крайние год месяц ДЕНЬ час минута 2022 10 20 23 59
    # download_dofiga_candles('YNDX', to_dt_string="", days_before=31)  # крайние год месяц ДЕНЬ час минута 2022 10 20 23 59
    # open_not_close_candle_df('C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/6.12_FIVE_not_close_candle_df.csv')
    # concat_dfs(14.11, 16.11, 'YNDX')
