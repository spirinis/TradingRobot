import os
import time

import proga
from tinkoff.invest import (
    CandleInstrument,
    InfoInstrument,
    Client,
    MarketDataRequest,
    SubscribeCandlesRequest,
    SubscribeInfoRequest,
    SubscriptionAction,
    SubscriptionInterval,
)

token_read = os.getenv('TITOKEN_READ_ALL', 'Ключа нет')


def main():
    def request_iterator_candles():
        yield MarketDataRequest(
            subscribe_candles_request=SubscribeCandlesRequest(
                waiting_close=True,
                subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                instruments=[
                    CandleInstrument(
                        figi="BBG006L8G4H1",
                        interval=SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE,
                    )
                ],
            )
        )
        i = 0
        while True:
            time.sleep(5)
            print('sleepC ', i)
            i += 5

    def request_iterator_info():
        yield MarketDataRequest(
            subscribe_info_request=SubscribeInfoRequest(
                subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                instruments=[
                    InfoInstrument(
                        figi="BBG006L8G4H1"
                    )
                ],
            )
        )
        i = 0
        while True:
            time.sleep(5)
            print('sleepI ', i)
            i += 5

    with Client(token_read) as client:
        for marketdata in client.market_data_stream.market_data_stream(
            #request_iterator_candles(),
            request_iterator_info()
        ):
            print(proga.unpack_MarketDataResponse(marketdata).get('list_response_clear'))


if __name__ == "__main__":
    main()
