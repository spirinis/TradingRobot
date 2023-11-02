import os
import pandas as pd
from pandas import DataFrame
import time
from datetime import datetime, timedelta, timezone
from tinkoff.invest import (Client, HistoricCandle, Candle, CandleInterval, Quotation, MarketDataResponse)

import threading

import finplot as fplt

token_read = os.getenv('TITOKEN_READ_ALL', 'Ключа нет')
global work

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


def cast_money(v: Quotation) -> float:
    return v.units + v.nano / 1e9  # nano - 9 нулей


def create_df(candles: list) -> DataFrame:
    if candles == [] or isinstance(candles, list) and len(candles) > 0 and isinstance(candles[0], HistoricCandle):
        df = DataFrame([{
            'time': c.time,
            'volume': c.volume,
            'open': cast_money(c.open),
            'close': cast_money(c.close),
            'high': cast_money(c.high),
            'low': cast_money(c.low),
        } for c in candles])
        return df


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
    if ((len(globals()['candle_df_%s' % ticker]) > 0)
       and is_close and (globals()['candle_df_%s' % ticker].time.iloc[-1] == candle.time)):
        globals()['candle_df_%s' % ticker] = globals()['candle_df_%s' % ticker][:-1]
    """if ((len(globals()['candle_df_%s' % ticker]) > 0)
       and (globals()['candle_df_%s' % ticker].time.iloc[-1] == candle.time)
       and (globals()['candle_df_%s' % ticker].close.iloc[-1] == candle.close)):
        globals()['candle_df_%s' % ticker] = globals()['candle_df_%s' % ticker][:-1]"""  # TODO верни
    globals()['candle_df_%s' % ticker] = pd.concat([globals()['candle_df_%s' % ticker], df])  # .iloc[1:]
    df['ticker'] = [ticker]
    return df


def df_candles(share: DataFrame, dt_string="2022 9 9 10 0"):  # 829 свечей
    from tinkoff.invest import (RequestError, )
    try:
        with Client(token_read) as client:
            figi = share.figi.iloc[0]  # 'BBG006L8G4H1'
            ticker = share.ticker.iloc[0]
            try:
                file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/%s_candle_df.csv' % ticker
                t1 = datetime.fromtimestamp(os.path.getmtime(file))
                dt1 = datetime.now() - t1
                if dt1.days == 0:
                    candle_df = pd.read_csv(file)
                    candle_df.time = pd.to_datetime(candle_df.time)
                    if ((datetime.now(timezone.utc)
                         - candle_df.time.iloc[-1].to_pydatetime() + timedelta(minutes=1)).days == 0):
                        candles = client.market_data.get_candles(
                            figi=figi,
                            from_=candle_df.time.iloc[-1] + timedelta(minutes=1),
                            to=datetime.utcnow(),
                            interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                        )
                        globals()['candle_df_%s' % ticker] = pd.concat([candle_df, create_df(candles)])
                    else:
                        raise FileNotFoundError()
                else:
                    raise FileNotFoundError()

            except FileNotFoundError:
                time_format = "%Y %m %d %H %M"
                # dt_object = datetime.strptime(to_dt_string, time_format)
                time_utc_now = datetime.utcnow()
                if time_utc_now.hour < 21:
                    dt_object = time_utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    dt_object = (time_utc_now+timedelta(hours=3)).replace(hour=0, minute=0, second=0, microsecond=0)
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
                for i in range(3):
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
                threading.Thread(target=run(ticker)).start()

                # candle_df = create_df(candles.candles)

    except RequestError as error:
        print('ошибка df_candles' + "\n" + str(error))


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
            # r = r[r['ticker'] == tiker]['figi'].iloc[0]
            if find_share.empty:
                raise Exception('Нет такого тикера')

                #{'name': find_share.iloc[0][0], 'ticker': find_share.iloc[0][1], 'figi': find_share.iloc[0][2],
                #          'lot': find_share.iloc[0][3], 'currency': find_share.iloc[0][4],
                #          'class_code': find_share.iloc[0][5]}
            return find_share
    except RequestError as error:
        print('ошибка find' + "\n" + str(error))


def FindTickerByFigi(figi: str) -> str:
    return df_of_shares[df_of_shares.figi == figi].ticker.iloc[0]


def trading_schedules() -> DataFrame:
    from tinkoff.invest import (RequestError, )
    from tinkoff.invest.services import (InstrumentsService, )
    global trading_schedule
    try:
        with Client(token_read) as client:
            instruments: InstrumentsService = client.instruments
            # market_data: MarketDataService = client.market_data
            schedules = instruments.trading_schedules(exchange='MOEX_PLUS', from_=datetime.utcnow(),
                                                      to=datetime.utcnow() + timedelta(days=1)).exchanges
            trading_schedule = DataFrame([{
                'exchange': s.exchange,
                'is_trading_day': s.days[0].is_trading_day,
                'start_time': s.days[0].start_time,
                'end_time': s.days[0].end_time.time(),
                'evening_start_time': s.days[0].evening_start_time.time(),
                'evening_end_time': s.days[0].evening_end_time.time(),
            } for s in schedules])
            return trading_schedule
    except RequestError as error:
        print('ошибка schedules' + "\n" + str(error))


def request_iterator_candles(figis: list, action: str):
    from tinkoff.invest import (MarketDataRequest, SubscribeCandlesRequest, CandleInstrument,
                                SubscriptionAction, SubscriptionInterval,)
    candle_instruments = []
    global work
    if action == 'subscribe':
        action = SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE
        work += 1
    elif action == 'unsubscribe':
        action = SubscriptionAction.SUBSCRIPTION_ACTION_UNSUBSCRIBE
        work -= 1
    for figi in figis:
        candle_instruments.append(
            CandleInstrument(
                figi=figi,
                interval=SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE)
        )
    yield MarketDataRequest(
        subscribe_candles_request=SubscribeCandlesRequest(
            waiting_close=is_close,
            subscription_action=action,
            instruments=candle_instruments
        )
    )
    i = 0
    while work > 0:
        time.sleep(5)
        # print('sleepC ', i)
        i += 5


def request_iterator_order_book(figis: list, action: str):
    from tinkoff.invest import (MarketDataRequest, SubscribeCandlesRequest, SubscribeOrderBookRequest, CandleInstrument,
                                OrderBookInstrument, SubscriptionAction, SubscriptionInterval, )
    candle_instruments = []
    order_book_instruments = []
    global work
    if action == 'subscribe':
        action = SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE
        work += 1
    elif action == 'unsubscribe':
        action = SubscriptionAction.SUBSCRIPTION_ACTION_UNSUBSCRIBE
        work -= 1
    for figi in figis:
        candle_instruments.append(
            CandleInstrument(
                figi=figi,
                interval=SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE)
        )
        order_book_instruments.append(
            OrderBookInstrument(
                figi=figi,
                depth=10
            )
        )
    yield MarketDataRequest(
        subscribe_order_book_request=SubscribeOrderBookRequest(
            subscription_action=action,
            instruments=order_book_instruments
        )
    )
    i = 0
    while work > 0:
        time.sleep(5)
        # print('sleepC ', i)
        i += 5


def unpack_MarketDataResponse(mdr: MarketDataResponse):
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
            """df_response = DataFrame({
                'ticker': FindTickerByFigi(response.figi),
                'time': response.time,
            })"""
            # list_response_translate.append(df_response)
            list_response_translate.append(df_response)
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
    list_response_clear = list(filter(lambda x: x is DataFrame or x is not None, list_response_translate))
    columns = ['sbsc_c', 'sbsc_or_b', 'sbsc_trades', 'sbsc_info', 'candle', 'trade', 'orderbook', 'trading_status',
               'ping', 'subscribe_last_price_response', 'last_price']
    sery = pd.Series(list_response_translate, index=columns)
    out = {'list_response_translate': list_response_translate, 'sery': sery,
           'list_response_clear': list_response_clear, }
    return out


def run(ticker):
    fplt.create_plot(ticker)
    load_candles = globals()['candle_df_%s' % ticker][['time', 'open', 'close', 'high', 'low']]
    plot = fplt.candlestick_ochl(load_candles)
    def update(): plot.update_data(load_candles)
    fplt.timer_callback(update, 30)
    fplt.show()


list_of_figis = list_of_tickers = []
counter_dict = {}
is_close = False
try:
    if __name__ == "__main__":
        work = 0
        # trading_schedule = DataFrame()  # trading_schedule = trading_schedules()
        # trading_schedules()
        # print('===== РАСПИСАНИЕ =====\n', trading_schedule)
        list_of_tickers = ['YNDX', 'FIVE']  # , 'OZON']  # , 'ROSN', 'SWN', ]/
        df_of_shares = DataFrame(columns=['name', 'ticker', 'figi', 'lot', 'currency', 'class_code'])
        for ticker in list_of_tickers:
            df_share = FindShareByTicker(ticker)
            df_of_shares = pd.concat([df_of_shares, df_share])  # df_of_shares.append(dict_share)  # BBG000N9MNX3
            # df_candles(df_share)  # globals()['candle_df_%s' % ticker]
            globals()['candle_df_%s' % ticker] = DataFrame({
                'time': [],
                'volume': [],
                'open': [],
                'close': [],
                'high': [],
                'low': [],
                'last_trade_ts': [],
            })
            counter_dict[ticker] = 0
        df_of_shares = df_of_shares.reset_index(drop=True)
        print('===== АКЦИИ =====\n', df_of_shares)
        list_of_figis = df_of_shares.figi.to_list()
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
        # print(trading_schedule.loc[27:34])

        # (FindShareByTicker('YNDX'))
except KeyboardInterrupt:
    with Client(token_read) as client:
        for marketdata in client.market_data_stream.market_data_stream(
                request_iterator_candles(list_of_figis, 'unsubscribe'),
                # request_iterator_order_book(list_of_figis, 'unsubscribe'),
        ):
            response = unpack_MarketDataResponse(marketdata).get('list_response_clear')
            if len(response) == 1:
                print('===== ОТВЕТ =====\n', response[0])
            else:
                print('===== ОТВЕТ =====\n', response)
            if work == 0:
                break
    if input('Сохранить массив? ') == '':
        for ticker in list_of_tickers:

            """
            ax = fplt.create_plot(ticker, rows=1)
            fplt.candlestick_ochl(globals()['candle_df_%s' % ticker][['time', 'open', 'close', 'high', 'low']], ax=ax)
            fplt.volume_ocv(globals()['candle_df_%s' % ticker][['time', 'open', 'close', 'volume']], ax=ax.overlay())
            fplt.show()
            """
            path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/%s_not_close_candle_TEST_df_ALL.csv' % ticker
            globals()['candle_df_%s' % ticker].to_csv(path, index=False)
    time.sleep(1)
