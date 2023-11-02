import os
from datetime import datetime, timedelta
from pandas import DataFrame
import math
import pandas as pd
import numpy as np
import finplot as fplt
import matplotlib.dates as mdates
import mplfinance as mpf
from tinkoff.invest import (Client, CandleInterval, RequestError, InstrumentStatus,
                            SharesResponse, InstrumentIdType, HistoricCandle, MarketDataResponse, schemas)
from tinkoff.invest.services import InstrumentsService, MarketDataService
from ta.trend import ema_indicator


token_read = os.getenv('TITOKEN_READ_ALL', 'Ключа нет')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def create_df(candles: [HistoricCandle]):
    global df
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


def create_MarketDataResponse_df(MDRs: [MarketDataResponse]):
    df = DataFrame([{
        'sbsc_c': MDR.subscribe_candles_response,
        'sbsc_or_b': MDR.subscribe_order_book_response,
        'sbsc_trades': MDR.subscribe_trades_response,
        'sbsc_info': MDR.subscribe_info_response,
        'candle': MDR.candle,
        'trade': MDR.trade,
        'orderbook': MDR.orderbook,
        'trading_status': MDR.trading_status,
        'ping': MDR.ping,
        'subscribe_last_price_response': MDR.subscribe_last_price_response,
        'last_price': MDR.last_price
    } for MDR in MDRs])
    return df


def unpack_MarketDataResponse(MDR: MarketDataResponse):
    import tinkoff.invest.schemas
    ##    for i in range(0,len(a)):
    ##	for j in a.loc[i]:
    list_response = [MDR.subscribe_candles_response,
                     MDR.subscribe_order_book_response,
                     MDR.subscribe_trades_response,
                     MDR.subscribe_info_response,
                     MDR.candle,
                     MDR.trade,
                     MDR.orderbook,
                     MDR.trading_status,
                     MDR.ping,
                     MDR.subscribe_last_price_response,
                     MDR.last_price]
    list_response_translate = []
    for response in list_response:
        if isinstance(response, tinkoff.invest.schemas.SubscribeCandlesResponse):
            list_response_translate.append(
                ['Candles ', response.candles_subscriptions[0].figi, ' ',
                 response.candles_subscriptions[0].subscription_status.name])
        elif isinstance(response, tinkoff.invest.schemas.SubscribeOrderBookResponse):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeTradesResponse):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeInfoResponse):
            list_response_translate.append(response.info_subscriptions[0].subscription_status.name)
        elif isinstance(response, tinkoff.invest.schemas.Candle):
            list_response_translate.append(create_df(response))
        elif isinstance(response, tinkoff.invest.schemas.Trade):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.OrderBook):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.TradingStatus):
            list_response_translate.append(response.trading_status.name)
        elif isinstance(response, tinkoff.invest.schemas.Ping):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeLastPriceResponse):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.LastPrice):
            list_response_translate.append(response)
        else:
            list_response_translate.append(response)
    list_response_clear = list(filter(None, list_response_translate))
    columns = ['sbsc_c', 'sbsc_or_b', 'sbsc_trades', 'sbsc_info', 'candle', 'trade', 'orderbook', 'trading_status',
               'ping', 'subscribe_last_price_response', 'last_price']
    sery = pd.Series(list_response_translate, index=columns)
    out = {'list_response_translate': list_response_translate, 'list_response_clear': list_response_clear, 'sery': sery}
    return out


def cast_money(v):
    return v.units + v.nano / 1e9  # nano - 9 нулей


def FindShareByTicker(TICKER) -> dict:
    try:
        with Client(token_read) as client:
            instruments: InstrumentsService = client.instruments
            market_data: MarketDataService = client.market_data
            try:
                shares = pd.read_csv('all_shares.csv')
            except FileNotFoundError:
                shares = DataFrame(
                    instruments.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE).instruments,
                    columns=['name', 'ticker', 'figi',  'lot', 'currency', 'class_code']
                )
                shares.to_csv('all_shares.csv', index=False)
            find_share = shares[shares['ticker'] == TICKER]
            # r = r[r['ticker'] == TICKER]['figi'].iloc[0]
            if find_share.empty: raise Exception('Нет такого тикера')
            find_share = {'name': find_share.iloc[0][0], 'ticker': find_share.iloc[0][1], 'figi': find_share.iloc[0][2],
                          'lot': find_share.iloc[0][3], 'currency': find_share.iloc[0][4], 'class_code': find_share.iloc[0][5]}
            return find_share
    except RequestError as error:
        print('ошибка find' + "\n" + str(error))


def trading_schedules() -> DataFrame:
    from tinkoff.invest import (InstrumentStatus, RequestError)
    from tinkoff.invest.services import (InstrumentsService, )
    try:
        with Client(token_read) as client:
            instruments: InstrumentsService = client.instruments
            # market_data: MarketDataService = client.market_data
            schedules = instruments.trading_schedules(from_=datetime.utcnow() + timedelta(days=2),
                                                      to=datetime.utcnow() + timedelta(days=3)).exchanges # exchange='MOEX_PLUS',
            trading_schedule = DataFrame([{
                'exchange': s.exchange,
                'is_trading_day': s.days[0].is_trading_day,
                'start_time': s.days[0].start_time,
                'end_time': s.days[0].end_time,
                'evening_start_time': s.days[0].evening_start_time,
                'evening_end_time': s.days[0].evening_end_time,
            } for s in schedules])
            return trading_schedule
    except RequestError as error:
        print('ошибка schedules' + "\n" + str(error))


def df_candles(share: dict, dt_string="2022 9 9 10 0"):
    try:
        with Client(token_read) as client:
            figi = share.get('figi')  # 'BBG006L8G4H1'
            ticker = share.get('ticker')

            try:
                # raise FileNotFoundError()
                df = pd.read_csv((ticker + '_' + 'candles.csv'))
                df.time = pd.to_datetime(df.time)
            except FileNotFoundError:
                time_format = "%Y %m %d %H %M"
                dt_object = datetime.strptime(dt_string, time_format)
                candles = client.market_data.get_candles(
                    figi=figi,
                    # from_=datetime.utcnow()-timedelta(hours = 24), to=datetime.utcnow(),
                    from_=dt_object, to=dt_object + timedelta(minutes=520),
                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                )
                df = create_df(candles.candles)
                df.to_csv((ticker + '_' + 'candles.csv'), index=False)

            # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.ema_indicator
            df['medium'] = np.nan
            df['ema'] = ema_indicator(close=df['close'], window=200)
            df['volume_diff'] = np.nan
            df['volume_diff_op'] = 1
            df['volume_diff_cl'] = 1
            df['volume_null_mark'] = 0
            df['volume_force'] = np.nan
            # df['volume_calm'] = np.nan
            df['degrees'] = np.nan
            df['volume_degrees'] = np.nan
            df['volume_min'] = np.nan
            df['degrees_close'] = np.nan
            for i in range(0, len(df.degrees)):
                df.loc[i, 'medium'] = (df.high[i] + df.low[i]) / 2
            df['derivative'] = df.medium.diff()
            df['volume_derivative'] = df.volume.diff()
            df['derivative_close'] = df.close.diff()
            for i in range(1, len(df.degrees)):  # Тангенс
                df.loc[i, 'degrees'] = math.degrees(math.atan(df.derivative[i]))
                # df.loc[i,'volume_calm'] = df.volume[i]/abs(df.degrees[i]+1)
                df.loc[i, 'degrees_close'] = math.degrees(math.atan(df.derivative_close[i]))
                df.loc[i, 'volume_degrees'] = math.degrees(math.atan(df.volume_derivative[i]))

            for i in range(1, len(df.volume_diff)):
                # .rolling(25, min_periods= 25).mean()
                if df.volume_derivative[i] >= 0:  # % роста объёма
                    df.loc[i, 'volume_diff'] = 100 - df.volume[i - 1] / df.volume[i] * 100
                    df.loc[i, 'volume_diff_op'] = 0
                else:  # % падения объёма
                    df.loc[i, 'volume_diff'] = 100 - df.volume[i] / df.volume[i - 1] * 100
                    df.loc[i, 'volume_diff_op'] = 2

            for i in range(3, len(df.volume_force)):
                # .rolling(25, min_periods= 25).mean()
                if ((df.volume_diff[i - 3] < df.volume_diff[i - 2] < df.volume_diff[i - 1] < 0) and (
                        df.volume_degrees[i - 3] < df.volume_degrees[i - 2] < df.volume_degrees[i - 1] < 0) or
                        (0 > df.volume_diff[i - 3] > df.volume_diff[i - 2] > df.volume_diff[i - 1]) and (
                                0 > df.volume_degrees[i - 3] > df.volume_degrees[i - 2] > df.volume_degrees[i - 1]) or
                        (0 < df.volume_diff[i - 3] < df.volume_diff[i - 2] < df.volume_diff[i - 1]) and (
                                0 < df.volume_degrees[i - 3] < df.volume_degrees[i - 2] < df.volume_degrees[i - 1]) or
                        (df.volume_diff[i - 3] > df.volume_diff[i - 2] > df.volume_diff[i - 1] > 0) and (
                                df.volume_degrees[i - 3] > df.volume_degrees[i - 2] > df.volume_degrees[i - 1] > 0)):
                    df.loc[i - 1, 'volume_force'] = 1
                elif ((df.volume_diff[i - 2] < df.volume_diff[i - 1] < 0) and (
                        df.volume_degrees[i - 2] < df.volume_degrees[i - 1] < 0) or
                      (0 > df.volume_diff[i - 2] > df.volume_diff[i - 1]) and (
                              0 > df.volume_degrees[i - 2] > df.volume_degrees[i - 1]) or
                      (0 < df.volume_diff[i - 2] < df.volume_diff[i - 1]) and (
                              0 < df.volume_degrees[i - 2] < df.volume_degrees[i - 1]) or
                      (df.volume_diff[i - 2] > df.volume_diff[i - 1] > 0) and (
                              df.volume_degrees[i - 2] > df.volume_degrees[i - 1] > 0)):
                    df.loc[i - 1, 'volume_force'] = 0.5
            '''
                if ((df.volume_degrees[i-3] < df.volume_degrees[i-2] < df.volume_degrees[i-1] < 0) or
                    (0 > df.volume_degrees[i-3] > df.volume_degrees[i-2] > df.volume_degrees[i-1]) or
                    (0 < df.volume_degrees[i-3] < df.volume_degrees[i-2] < df.volume_degrees[i-1]) or
                    (df.volume_degrees[i-3] > df.volume_degrees[i-2] > df.volume_degrees[i-1] > 0)):
                    df.loc[i-1,'volume_force']= 1
                elif ((df.volume_degrees[i-2] < df.volume_degrees[i-1] < 0) or
                    (0 >  df.volume_degrees[i-2] > df.volume_degrees[i-1]) or
                    (0 <  df.volume_degrees[i-2] < df.volume_degrees[i-1]) or
                    (df.volume_degrees[i-2] > df.volume_degrees[i-1] > 0)):
                    df.loc[i-1,'volume_force']= 0.5
            '''
            """
                if ((df.volume_degrees[i-3] < 0) and (df.volume_degrees[i-2]< 0) and
                    (df.volume_degrees[i-1] < 0) and (df.volume_degrees[i] > 0)):
                    df.loc[i,'volume_force']= 1
                elif ((df.volume_degrees[i-2]< 0) and (df.volume_degrees[i-1] < 0) and
                      (df.volume_degrees[i] > 0)):
                    df.loc[i,'volume_force']= 0.5
            """

            df.loc[i, 'volume_min'] = df.volume[i]
            df.volume_min = df.volume.rolling(30, min_periods=8).median()
            #                if df.volume[i]/abs(df.degrees[i]+1) > 300:
            #                    df.loc[i,'volume_calm']= 300
            #                else:
            #            for i in range(1, len(df.volume_force)):
            #                df.loc[i,'volume_calm']= df.volume[i]/abs(df.degrees[i]-1)

            # elif df.derivative[i] < 0:
            #    df.loc[i,'volume_force']= abs(df.volume[i-1] - df.volume[i])
            print(df.head(15))

            #            import matplotlib.pyplot as plt
            #            print(df[['time', 'close', 'ema']])
            #            fig = plt.figure(figsize=(10, 10), dpi= 80)
            #            ax1 = fig.add_subplot(2,1,1)
            #            ax2 = fig.add_subplot(2,1,2)
            #            df.plot(ax = ax1, x='time', y='close', marker='o')
            #            df.plot(ax=ax1, x='time', y='ema')
            #            df.plot(ax=ax1, x='time', y='high')
            #            df.plot(ax=ax1, x='time', y='low',color = 'red')
            #            ax1.grid(color = 'gray',linewidth = 1)
            #
            #            #ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
            #            df.plot.bar(ax=ax2,x='time', y='volume')
            #            plt.show()

            ax, ax2, ax3, ax4, ax5 = fplt.create_plot(ticker, rows=5)

            fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
            fplt.plot(df['time'], df['medium'], ax=ax, width=3, legend=ticker + 'medium')
            # fplt.plot(df['time'],df['close'],ax = ax,width = 3,legend=TICKER+'close')
            # fplt.plot(df['time'],df['open'], ax = ax,width = 3,legend=TICKER+'open')

            # fplt.plot(df['time'],df['close'].rolling(25, min_periods= 25).mean(),ax = ax,legend='ma-25')
            fplt.plot(df['time'], df['close'].ewm(span=10, min_periods=10, adjust=False).mean(), ax=ax, legend='ema-10',
                      color='green')
            fplt.plot(df['time'], df['close'].ewm(span=50, min_periods=50, adjust=False).mean(), ax=ax, legend='ema-50',
                      color='red')
            fplt.plot(df['time'], df['close'].ewm(span=20, min_periods=20, adjust=False).mean(), ax=ax, legend='ema-20',
                      color='brown')
            fplt.plot(df['time'], df['close'].expanding().mean(), ax=ax, legend='cma')

            # fplt.plot(df['time'],df.derivative,ax = ax2,legend='derivative', color = 'red')
            fplt.plot(df['time'], df.degrees, legend='degrees', ax=ax2)
            #  fplt.plot(df['time'], df.degrees.ewm(span=2, min_periods=2, adjust=False).mean(), ax=ax2, legend='ema-2',
            #          color='green')
            fplt.plot(df['time'], df.degrees.rolling(3).mean(), ax=ax2, legend='sma-3',
                      color='green')
            # fplt.volume_ocv(df[['time','open','close','volume_degrees']], ax=ax2)
            # fplt.plot(df['time'],df.degrees_close,ax = ax2,legend='degrees_close')
            fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax4)

            # fplt.plot(df['time'],df['degrees'].expanding().mean(),width = 3,ax = ax4,legend='cma_degr')

            fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax.overlay())
            fplt.volume_ocv(df[['time', 'open', 'close', 'volume_force']], ax=ax3)
            fplt.plot(df.time, df.volume_null_mark, ax=ax3, legend='volume_force')

            fplt.volume_ocv(df[['time', 'volume_diff_op', 'volume_diff_cl', 'volume_diff']], ax=ax5)
            fplt.plot(df.time, df.volume_null_mark, ax=ax5, legend='volume_diff')
            # fplt.volume_ocv(df[['time','open','close','volume_calm']], ax=ax4)
            # fplt.plot(df.time,df.volume_degrees,ax = ax5,legend='volume_degrees')
            fplt.plot(df.time, df.volume_min, ax=ax4, legend='volume')
            # fplt.volume_ocv(df[['time','volume_diff_op','volume_diff_cl','volume_min']], ax=ax5)

            """ah = fplt.create_plot(tiker+'_', rows = 1)
            fplt.candlestick_ochl(df[['time','open','close','high','low']], ax = ah)
            fplt.plot(df['time'],df['close'].ewm(span=50,min_periods=50,adjust=False).mean(),ax = ah,legend='50',color = 'green')
            fplt.plot(df['time'],df['close'].ewm(span=200,adjust=False).mean(),ax = ah,legend='200T')#==fplt.plot(df['time'],df['ema'],ax = ah,legend='200m')###
            fplt.plot(df['time'],df['close'].ewm(halflife = 200,min_periods=200,adjust=False).mean(),ax = ah,legend='200h')
            fplt.plot(df['time'],df['close'].ewm(alpha = 2/(200+1),min_periods=200,adjust=True).mean(),ax = ah,legend='200a')"""

            fplt.show()

            global df1
            df1 = df
    except RequestError as error:
        print('ошибка' + "\n" + str(error))


if __name__ == "__main__":
    df_candles(FindShareByTicker('YNDX'))
    # trading_schedule = trading_schedules()
    # print(trading_schedule.loc[27:34])
    # print(trading_schedule)
