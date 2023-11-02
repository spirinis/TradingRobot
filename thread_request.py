import os
import pandas as pd
from pandas import DataFrame
import time
from datetime import datetime, timedelta, timezone
from tinkoff.invest import (
    Client,
    HistoricCandle, Candle, CandleInterval,
    Quotation,
    MarketDataResponse)
import tinkoff.invest.schemas

from threading import Thread
token_read = os.getenv('TITOKEN_READ_ALL', 'Ключа нет')
is_close = False
subs_candles = 0
subs_order_book = 0

go = True
counter_dict = {}

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


def cast_money(v: Quotation) -> float:
    return v.units + v.nano / 1e9  # nano - 9 нулей


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


def request_iterator_order_book(figis: list, action: str):
    from tinkoff.invest import (
        MarketDataRequest,
        SubscribeOrderBookRequest,
        OrderBookInstrument,
        SubscriptionAction,
    )
    order_book_instruments = []
    global subs_order_book
    if action == 'subscribe':
        action = SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE
        subs_order_book += 1
    elif action == 'unsubscribe':
        action = SubscriptionAction.SUBSCRIPTION_ACTION_UNSUBSCRIBE
        subs_order_book -= 1
    for figi in figis:
        order_book_instruments.append(
            OrderBookInstrument(
                figi=figi,
                depth=20
            )
        )
    yield MarketDataRequest(
        subscribe_order_book_request=SubscribeOrderBookRequest(
            subscription_action=action,
            instruments=order_book_instruments
        )
    )
    i = 0
    while go:
        time.sleep(1)
        i += 1
        if i >= 5:
            print('sleepOB ', i)
            i -= 5


def stream_candles():
    global go
    while go:
        try:
            with Client(token_read) as client:
                for marketdata in client.market_data_stream.market_data_stream(
                        request_iterator_candles(list_of_figis, 'subscribe')
                ):
                    response = unpack_MarketDataResponse(marketdata).get('list_response_clear')
                    print('===== ОТВЕТ =====\n', response[0])
                    '''if isinstance(response, tinkoff.invest.schemas.Ping):
                        if (response.time.hour == 20 and response.time.minute >= 50) or (response.time.hour >= 21):
                            break
                    if subs_candles == 0:
                        break'''
                print('outc')
        except Exception as exc:
            print(str(type(exc)) + ' | ' + str(exc))
            continue


def stream_order_book():
    global go
    while go:
        try:
            with Client(token_read) as client:
                for marketdata in client.market_data_stream.market_data_stream(
                        # request_iterator_candles(list_of_figis, 'subscribe'),
                        request_iterator_order_book(list_of_figis, 'subscribe'),
                ):
                    response = unpack_MarketDataResponse(marketdata).get('list_response_clear')
                    print('===== ОТВЕТ =====\n', response[0])
                    '''if isinstance(response, tinkoff.invest.schemas.Ping):
                        if (response.time.hour == 20 and response.time.minute >= 50) or (response.time.hour >= 21):
                            break
                    if subs_order_book == 0:
                        break'''
                print('outob')
        except Exception as exc:
            print(str(type(exc)) + ' | ' + str(exc) + str(exc.__traceback__))
            continue


list_of_tickers = ['YNDX', 'FIVE', 'OZON']  # , 'ROSN', 'SWN', ]/
df_of_shares = DataFrame(columns=['name', 'ticker', 'figi', 'lot', 'currency', 'class_code'])
for ticker in list_of_tickers:
    df_share = FindShareByTicker(ticker)
    df_of_shares = pd.concat([df_of_shares, df_share])  # df_of_shares.append(dict_share)  # BBG000N9MNX3
    list_of_figis = df_of_shares.figi.to_list()
    # df_candles(df_share)  # globals()['candle_df_%s' % ticker]
    globals()['volume_df_%s' % ticker] = DataFrame({
        'price': [],
        'quantity_buy': [],
        'quantity_sale': [],
        'time': [],
        'is_consistent': [],
    })

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


t1 = Thread(target=stream_candles, daemon=True)
# t2 = Thread(target=stream_order_book, daemon=True)
t1.start()
# t2.start()
# t1.join()
# t2.join()
i = 0

# go = False
'''while True:
    i += 1
    time.sleep(3)
    print('центр', i)'''

'''lass 'tinkoff.invest.exceptions.RequestError'> | (<StatusCode.RESOURCE_EXHAUSTED: (8, 'resource exhausted')>, '', Metadata(tracking_id=None, ratelimit_limit='300, 300;w=60', ratelimit_remaining=0, ratelimit_reset=24, message=None))
<class 'tinkoff.invest.exceptions.RequestError'> | (<StatusCode.RESOURCE_EXHAUSTED: (8, 'resource exhausted')>, '', Metadata(tracking_id=None, ratelimit_limit='300, 300;w=60', ratelimit_remaining=0, ratelimit_reset=24, message=None))
None MarketDataStream RESOURCE_EXHAUSTED
None MarketDataStream RESOURCE_EXHAUSTED
<class 'tinkoff.invest.exceptions.RequestError'> | (<StatusCode.RESOURCE_EXHAUSTED: (8, 'resource exhausted')>, '', Metadata(tracking_id=None, ratelimit_limit='300, 300;w=60', ratelimit_remaining=0, ratelimit_reset=24, message=None))
<class 'tinkoff.invest.exceptions.RequestError'> | (<StatusCode.RESOURCE_EXHAUSTED: (8, 'resource exhausted')>, '', Metadata(tracking_id=None, ratelimit_limit='300, 300;w=60', ratelimit_remaining=0, ratelimit_reset=24, message=None))
None MarketDataStream RESOURCE_EXHAUSTED
None MarketDataStream RESOURCE_EXHAUSTED
<class 'tinkoff.invest.exceptions.RequestError'> | (<Stat'''