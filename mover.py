import os
import pandas as pd
from pandas import DataFrame
import time
from datetime import datetime, timedelta, timezone
from tinkoff.invest import (Client, HistoricCandle, Candle, CandleInterval, Quotation, MarketDataResponse)

import threading

import finplot as fplt

token_read = os.getenv('TITOKEN_READ_ALL', 'Ключа нет')
account_id = '2054618731'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


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

    find_share = {'name': find_share.name[0], 'ticker': find_share.ticker[0], 'figi': find_share.figi[0],
                  'lot': find_share.lot[0], 'currency': find_share.currency[0],
                  'class_code': find_share.class_code[0]}
    return find_share


def current_positions() -> DataFrame:
    from tinkoff.invest import (RequestError, )
    try:
        with Client(token_read) as client:
            securities = client.operations.get_positions(account_id=account_id).securities
            shares = DataFrame([{
                'ticker': FindShareOrTicker(figi=s.figi).get('ticker'),
                'figi': s.figi,
                'balance': s.balance,
                'position_uid': s.position_uid,
            } for s in securities])
            return shares
    except RequestError as error:
        print('ошибка schedules' + "\n" + str(error))


def cast_money(v: Quotation) -> float:
    return v.units + v.nano / 1e9  # nano - 9 нулей


def append_candle_df(candle: Candle) -> DataFrame:
    global counter_list
    ticker = FindShareOrTicker(figi=candle.figi).get('ticker')
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
                'ticker': FindShareOrTicker(figi=candles_subscription.figi).get('ticker'),
                'candle_sbscr_status': candles_subscription.subscription_status.name,
            } for candles_subscription in response.candles_subscriptions])
            list_response_translate.append(df_response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeOrderBookResponse):
            df_response = DataFrame([{
                'ticker': FindShareOrTicker(figi=order_book_subscription.figi).get('ticker'),
                'order_book_sbscr_status': order_book_subscription.subscription_status.name,
            } for order_book_subscription in response.order_book_subscriptions])
            list_response_translate.append(df_response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeTradesResponse):
            list_response_translate.append(response)
        elif isinstance(response, tinkoff.invest.schemas.SubscribeInfoResponse):
            # list_response_translate.append(response.info_subscriptions[0].subscription_status.name)
            df_response = DataFrame([{
                'ticker': FindShareOrTicker(figi=info_subscription.figi).get('ticker'),
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


list_of_figis = list_of_tickers = counter_list = []
counter_dict = {}
is_close = False
work = 0
try:
    if __name__ == "__main__":
        list_of_tickers = ['YNDX', 'FIVE']  # , 'OZON']  # , 'ROSN', 'SWN', ]/
        df_of_shares = DataFrame(columns=['name', 'ticker', 'figi', 'lot', 'currency', 'class_code'])
        for ticker in list_of_tickers:
            df_share = pd.DataFrame(FindShareOrTicker(ticker=ticker), index=[0])
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
            path = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/' \
                   '%s_not_close_candle_TEST_df_ALL.csv' % ticker
            globals()['candle_df_%s' % ticker].to_csv(path, index=False)
    time.sleep(1)
