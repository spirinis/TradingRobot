import pandas as pd
from pandas import DataFrame
import finplot as fplt
import numpy as np

ticker = 'YNDX'
file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/28-30.09_YNDX_candle_df.csv'
# file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/5-7.10_YNDX_candle_df.csv'
df = pd.read_csv(file)
df.time = pd.to_datetime(df.time)


df['medium'] = np.nan
df['decision'] = np.nan
df['up_bar'] = np.nan
df['down_bar'] = np.nan
df['cost'] = np.nan
df['status'] = np.nan
df['mistake'] = np.nan
df['buy'] = np.nan
df['sale'] = np.nan
cost = 0  # 1725.0
percent = 0.18/100  # без комиссии
up_bar = cost * (1.0009+percent)
down_bar = cost * (0.9991-percent)
decision = 0
summ_profit = profit = 0
summ_percent_profit = percent_profit = 0
buy = cost
time_buy = 0
counter = 0

df.loc[0, 'medium'] = (df.high[0] + df.low[0]) / 2


def extremum(dfc, percent=0.09):
    df['extr_up'] = np.nan
    df['extr_dwn'] = np.nan
    df['extr_des'] = np.nan
    percent = percent / 100
    napr = 1
    extr_up = 0
    extr_dwn = dfc.loc[0]*2
    i_extr = 0
    cost = dfc.loc[0]
    for i in range(0, len(dfc)):
        prev_cost = cost
        cost = dfc.loc[i]
        if napr == 1 and cost * (1 + percent) < extr_up:
            napr = -1
            df.loc[i_extr, 'extr_up'] = extr_up
            df.loc[i+1, 'extr_des'] = extr_up
            extr_up = 0
        elif napr == -1 and cost * (1 - percent) > extr_dwn:
            napr = 1
            df.loc[i_extr, 'extr_dwn'] = extr_dwn
            df.loc[i + 1, 'extr_des'] = extr_dwn
            extr_dwn = cost * 2
        if napr == 1 and cost > extr_up:
            extr_up = cost
            i_extr = i
        elif napr == -1 and cost < extr_dwn:
            extr_dwn = cost
            i_extr = i
    """ax, ax2 = fplt.create_plot('YNDX' + str(percent), rows=2)
    fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
    fplt.plot(df['time'], df['extr_up'], legend='extr_up', ax=ax, color='red', width=1, style='o')
    fplt.plot(df['time'], df['extr_dwn'], legend='extr_dwn', ax=ax, color='blue', width=1, style='o')
    fplt.plot(df['time'], df['extr_des'], legend='extr_des', ax=ax, color='gray', width=1, style='o')
    fplt.plot(df['time'], df['medium'], legend='medium', ax=ax, color='orange', width=0.5)
    fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax2)
    fplt.show()"""


for i in range(1, len(df)):
    df.loc[i, 'medium'] = (df.high[i] + df.low[i]) / 2

    if cost > 0:  # продаём
        if df.high.loc[i-1] * (1.0009+percent) > up_bar:  #
            up_bar = df.high.loc[i-1] * 1.0009
        else:
            if decision > -1:
                decision = -1
            up_bar = np.nan
            down_bar = df.high.loc[i-1] * 0.9991
        df.loc[i-1, 'up_bar'] = up_bar
        df.loc[i-1, 'down_bar'] = down_bar
    else:  # покупаем
        if (df.low.loc[i-1] * (0.9991-percent) >= down_bar):  # and (napr):
            decision = 1
            down_bar = np.nan
            up_bar = df.low.loc[i - 1] * 1.0009
        else:
            down_bar = df.low.loc[i - 1] * 0.9991
        df.loc[i-1, 'up_bar'] = up_bar
        df.loc[i-1, 'down_bar'] = down_bar

    if decision == 1:  # покупка
        #  cost = buy = df.low.loc[i-1]
        cost = buy = df.medium.loc[i]
        time_buy = df.loc[i, 'time']
        time_buy = str(time_buy.day) + '-' + str(time_buy.hour) + ':' + str(time_buy.minute)
        #  df.loc[i, 'cost'] = cost*1.0015  # TODO
        df.loc[i, 'cost'] = cost
        df.loc[i, 'status'] = cost * 0.995
        df.loc[i, 'buy'] = cost
        decision = 2
    elif decision == 2:  # графика после покупки
        df.loc[i, 'cost'] = df.high.loc[i - 1] * 0.9985
        df.loc[i, 'status'] = df.high.loc[i - 1] * 0.9985 * 0.995
    elif decision <= -1:  # в продаже
        #  cost = df.high.loc[i - 1] * 0.9985
        mistake = df.medium.loc[i] * 0.0002*abs(decision + 1)
        cost = df.medium.loc[i]
        profit = round(cost - buy - (cost + buy)*0.0004, 2)
        percent_profit = round(cost/(buy/100) - 100 - 0.09, 2)
        if percent_profit > 0:  # продажа
            summ_profit += profit
            summ_percent_profit += percent_profit
            df.loc[i, 'cost'] = cost
            df.loc[i, 'status'] = cost * 0.995
            df.loc[i, 'sale'] = cost
            cost = 0
            decision = 0
            counter += 1
            time_sale = df.loc[i, 'time']
            time_sale = str(time_sale.day)+'-'+str(time_sale.hour)+':'+str(time_sale.minute)
            print(profit, str(percent_profit) + '% ' + time_buy + ' -> ' + time_sale)
        else:  # в минусе в продаже
            decision -= 1
            # df.loc[i, 'status'] = df.medium.loc[i - 1] * 0.994
            df.loc[i, 'mistake'] = cost


print(percent*100, counter, round(summ_profit, 2), str(round(summ_percent_profit, 2))+'%')
extremum(df.medium)
ax, ax2 = fplt.create_plot('YNDX'+str(percent), rows=2)
fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
#  fplt.plot(df['time'], df['medium'], width=3, legend=ticker + 'medium', ax=ax)
fplt.plot(df['time'], df['up_bar'], legend='up_bar', ax=ax, color='green')
fplt.plot(df['time'], df['down_bar'], legend='down_bar', ax=ax, color='red')
fplt.plot(df['time'], df['cost'], legend='cost', ax=ax, color='orange')
fplt.plot(df['time'], df['status'], legend='status', ax=ax, color='blue', width=2)
fplt.plot(df['time'], df['mistake'], legend='mistake', ax=ax, color='red', width=2)
fplt.plot(df['time'], df['buy'], legend='buy', ax=ax, color='blue', width=2, style='^')
fplt.plot(df['time'], df['sale'], legend='sale', ax=ax, color='red', width=2, style='v')
fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax2)

fplt.plot(df['time'], df['extr_up'], legend='extr_up', ax=ax, color='#ff7777', width=1, style='o')
fplt.plot(df['time'], df['extr_dwn'], legend='extr_dwn', ax=ax, color='#7777ff', width=1, style='o')
fplt.plot(df['time'], df['extr_des'], legend='extr_des', ax=ax, color='gray', width=1, style='o')
fplt.plot(df['time'], df['medium'], legend='medium', ax=ax, color='orange', width=0.5)

fplt.show()
"""
with Client(token_read) as client:
    a= client.market_data.get_last_prices(figi=('BBG006L8G4H1',))
a
GetLastPricesResponse(last_prices=[LastPrice(figi='BBG006L8G4H1', price=Quotation(units=1997, nano=400000000),
time=datetime.datetime(2022, 10, 14, 15, 45, 17, 330, tzinfo=datetime.timezone.utc),
instrument_uid='10e17a87-3bce-4a1f-9dfc-720396f98a3c')])
"""
"""разница между последними экстемумами
последним экстемумом и ценой
?несколько свечей назад и ценой? то НЕТ"""