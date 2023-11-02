import pandas as pd

import finplot as fplt
import numpy as np

# file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/28-30.09_YNDX_candle_df.csv'  # рост 11%
# file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/5-7.10_YNDX_candle_df.csv'  # плиты, падение
# file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/18-20.10_YNDX_candle_df.csv'  # волатильное падение
# file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/28.8-27.9_YNDX_candle_df.csv'  # 31 день
file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/close/28.9-28.10_YNDX_candle_df.csv'  # 31 день рост 19%
df = pd.read_csv(file)
df.time = pd.to_datetime(df.time)


ticker = 'YNDX'

df['medium'] = np.nan
df['decision'] = np.nan
df['up_bar'] = np.nan
df['down_bar'] = np.nan
df['cost'] = np.nan
df['status'] = np.nan
df['mistake'] = np.nan
df['buy'] = np.nan
df['sale'] = np.nan
df['bad_sale'] = np.nan
df['extr_up'] = np.nan
df['extr_dwn'] = np.nan
df['extr_des'] = np.nan
df['extr_medium'] = np.nan
cost = 0
extr_df = pd.DataFrame()
extr_df['time'] = np.nan
extr_df['extr_up'] = np.nan
extr_df['extr_dwn'] = np.nan
extr_df['medium'] = np.nan

jump_buy_percent = 0.03/100  # 0.08  без комиссии ?0.18?
jump_sale_percent = 0.05/100  # 0.05 без комиссии
gap_sale_percent = 0.03/100  # 0.05

percent_extr = 0.10 / 100
diff_extr = 0  # 0.07
stop_loss_percent = 0.33 / 100  # TODO

# up_bar = cost * (1.0009+percent)
# down_bar = cost * (0.9991-percent)np.nan
up_bar = np.nan
down_bar = np.nan
decision = 0
bad_sale_flag = False
summ_profit = profit = 0
summ_percent_profit = percent_profit = 0
buy = cost
i_buy = 0
time_buy = 0
counter = 0
des_name = ''

df.loc[0, 'medium'] = (df.high[0] + df.low[0]) / 2
price = df.medium.loc[0]
napr = 1
extr_up = 0
extr_up_count = 0
extr_dwn = df.medium.loc[0]*2
i_extr = 0
prev_extr = price
prev_i_extr = 0
prev_buy_extr = 0


def jump_des():
    global up_bar, down_bar, decision, des_name
    des_name = 'jd'
    if cost > 0:  # продаём
        if df.medium.loc[i - 1] * (1.0009 + jump_sale_percent) <= up_bar:
            if decision > -1:
                decision = -1
            up_bar = np.nan
            down_bar = df.medium.loc[i - 1] * 0.9991
        else:
            up_bar = df.medium.loc[i - 1] * 1.0009
        df.loc[i - 1, 'up_bar'] = up_bar
        df.loc[i - 1, 'down_bar'] = np.nan  # down_bar
    else:  # покупаем
        if ((df.medium.loc[i - 1] * (0.9991 - jump_buy_percent) >= down_bar) and (not (napr == 1)
           or (round(price / (df.loc[i_extr, 'extr_dwn'] / 100) - 100, 2) >= diff_extr))
           and not (df.loc[i - 1, 'extr_up'] > 0)):  # TODO решение покупка
            decision = 1
            down_bar = np.nan
            up_bar = df.medium.loc[i - 1] * 1.0009
        else:
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = np.nan  # up_bar
        df.loc[i - 1, 'down_bar'] = down_bar


def cost_sale_des():  # s%p = 14.54%
    global up_bar, down_bar, decision, des_name
    des_name = 'csd'
    if cost > 0:  # продаём
        # если предыдущее + комиссия меньше максимума + комиссия + зазор
        if df.medium.loc[i - 1] * (1.0009 + gap_sale_percent) <= up_bar:
            if decision > -1:
                decision = -1
            # up_bar = np.nan
            down_bar = df.medium.loc[i - 1] * 0.9991
        else:
            if df.medium.loc[i - 1] * 1.0009 > up_bar:
                up_bar = df.medium.loc[i - 1] * 1.0009
        df.loc[i - 1, 'up_bar'] = up_bar
        df.loc[i - 1, 'down_bar'] = np.nan  # down_bar
    else:  # покупаем
        # если предыдущее-комиссия-зазор больше препредыдущего-комиссия,
        # и если не возрастает или разница с нижн экстр больше порога
        # и если предыдущее не верх экстр
        if ((df.medium.loc[i - 1] * (0.9991 - jump_buy_percent) >= down_bar) and
                (not (napr == 1) or (round(price / (df.loc[i_extr, 'extr_dwn'] / 100) - 100, 2) >= diff_extr))
                and not (df.loc[i - 1, 'extr_up'] > 0)):  # TODO решение покупка
            decision = 1
            down_bar = np.nan
            up_bar = df.medium.loc[i - 1] * 1.0009
        else:
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = np.nan  # up_bar
        df.loc[i - 1, 'down_bar'] = down_bar


def cost_sale_des_cancelled_sale():  # s%p = 17.42%
    global up_bar, down_bar, decision, des_name
    des_name = 'csdcs'
    if cost > 0:  # продаём
        # если предыдущее + комиссия меньше максимума + комиссия + зазор
        if df.medium.loc[i - 1] * (1.0009 + gap_sale_percent) <= up_bar:
            if decision > -1:
                decision = -1
            # up_bar = np.nan

        else:
            if df.medium.loc[i - 1] * 1.0009 > up_bar:
                up_bar = df.medium.loc[i - 1] * 1.0009
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = up_bar
        df.loc[i - 1, 'down_bar'] = np.nan  # down_bar
    if (cost == 0) or (decision == -1):  # покупаем
        # если предыдущее-комиссия-зазор больше препредыдущего-комиссия,
        # и если не возрастает или разница с нижн экстр больше порога
        # и если предыдущее не верх экстр
        if ((df.medium.loc[i - 1] * (0.9991 - jump_buy_percent) >= down_bar)
                and not (df.loc[i - 1, 'extr_up'] > 0)):  # TODO решение покупка
            if decision == -1:
                decision = 0
            else:
                decision = 1
            down_bar = np.nan
            up_bar = df.medium.loc[i - 1] * 1.0009
        else:
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = np.nan  # up_bar
        df.loc[i - 1, 'down_bar'] = down_bar
    if decision == -1:  # если всёже
        down_bar = df.medium.loc[i - 1] * 0.9991


def cost_sale_medextr_des_cancelled_sale():
    global up_bar, down_bar, decision, des_name
    des_name = 'csmedcs'
    if cost > 0:  # продаём
        # если предыдущее + комиссия меньше максимума + комиссия + зазор
        # и если над средней экстремума
        if (df.medium.loc[i - 1] * (1.0009 + gap_sale_percent) <= up_bar)\
                and (df.medium.loc[i - 1] > extr_medium):
            if decision > -1:
                decision = -1
            # up_bar = np.nan

        else:
            if df.medium.loc[i - 1] * 1.0009 > up_bar:
                up_bar = df.medium.loc[i - 1] * 1.0009
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = up_bar
        df.loc[i - 1, 'down_bar'] = np.nan  # down_bar
    if (cost == 0) or (decision == -1):  # покупаем
        # если предыдущее-комиссия-зазор больше препредыдущего-комиссия,
        # УБРАЛ и если не возрастает или разница с нижн экстр больше порога
        # и если предыдущее не верх экстр
        # и если под средней экстремума
        if ((df.medium.loc[i - 1] * (0.9991 - jump_buy_percent) >= down_bar)
                and not (df.loc[i - 1, 'extr_up'] > 0)
                and (df.medium.loc[i - 1] < extr_medium)):  # TODO решение покупка
            if decision == -1:
                decision = 0
            else:
                decision = 1
            down_bar = np.nan
            up_bar = df.medium.loc[i - 1] * 1.0009
        else:
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = np.nan  # up_bar
        df.loc[i - 1, 'down_bar'] = down_bar
    if decision == -1:  # если всёже
        down_bar = df.medium.loc[i - 1] * 0.9991


def cost_sale_medextr_des_cancelled_sale_out():
    global up_bar, down_bar, decision, des_name, bad_sale_flag
    des_name = 'csmedcso'
    if cost > 0:  # продаём
        # если предыдущее + комиссия меньше максимума + комиссия + зазор
        # и если над средней экстремума
        if (df.medium.loc[i - 1] * (1.0009 + gap_sale_percent) <= up_bar)\
                and (df.medium.loc[i - 1] > extr_medium):
            if decision > -1:
                decision = -1
            # up_bar = np.nan

        else:
            if df.medium.loc[i - 1] * 1.0009 > up_bar:
                up_bar = df.medium.loc[i - 1] * 1.0009
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = up_bar
        df.loc[i - 1, 'down_bar'] = np.nan  # down_bar
    if (cost == 0) or (decision == -1):  # покупаем
        # если предыдущее-комиссия-зазор больше препредыдущего-комиссия,
        # УБРАЛ и если не возрастает или разница с нижн экстр больше порога
        # и если предыдущее не верх экстр
        # и если под средней экстремума
        if ((df.medium.loc[i - 1] * (0.9991 - jump_buy_percent) >= down_bar)
                and not (df.loc[i - 1, 'extr_up'] > 0)
                and (df.medium.loc[i - 1] < extr_medium)):  # TODO решение покупка
            if decision == -1:
                decision = 0
            else:
                decision = 1
            down_bar = np.nan
            up_bar = df.medium.loc[i - 1] * 1.0009
        else:
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = np.nan  # up_bar
        df.loc[i - 1, 'down_bar'] = down_bar
    if decision == -1:  # если всёже
        down_bar = df.medium.loc[i - 1] * 0.9991
    if (cost > 0) and (extr_up_count >= 0) and (df.medium.loc[i - 1] * 1.0009 < buy)\
            and (i-i_buy > (60*8+49 + 60*4-3+49)):
        bad_sale_flag = True
        decision = -1
        df.loc[i - 1, 'up_bar'] = up_bar
        df.loc[i - 1, 'down_bar'] = np.nan
        down_bar = df.medium.loc[i - 1] * 0.9991


def cost_sale_des_cancelled_buy():  # s%p = 9.46%
    global up_bar, down_bar, decision, des_name
    des_name = 'csdcb'
    if cost == 0:  # покупаем
        # если предыдущее-комиссия-зазор больше препредыдущего-комиссия,
        # и если не возрастает или разница с нижн экстр больше порога
        # и если предыдущее не верх экстр
        if ((df.medium.loc[i - 1] * (0.9991 - jump_buy_percent) >= down_bar)
                and not (df.loc[i - 1, 'extr_up'] > 0)):  # TODO решение покупка
            decision = 1
            down_bar = np.nan
        else:
            down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = np.nan  # up_bar
        df.loc[i - 1, 'down_bar'] = down_bar
    if (cost > 0) or (decision == 1):  # продаём
        # если предыдущее + комиссия меньше максимума + комиссия + зазор
        if df.medium.loc[i - 1] * (1.0009 + gap_sale_percent) <= up_bar:
            if decision != 1:
                decision = -1
            else:
                decision = 0
        else:
            if df.medium.loc[i - 1] * 1.0009 > up_bar:
                up_bar = df.medium.loc[i - 1] * 1.0009
        down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = up_bar
        df.loc[i - 1, 'down_bar'] = np.nan  # down_bar

    if decision == 1:  # если всёже
        up_bar = df.medium.loc[i - 1] * 1.0009


def cost_sale_nedes_twice():
    global up_bar, down_bar, decision, des_name
    des_name = 'csndt'
    if cost > 0:  # продаём
        # если предыдущее + комиссия меньше максимума + комиссия + зазор
        if df.medium.loc[i - 1] * (1.0009 + gap_sale_percent) <= up_bar:
            if decision > -1:
                decision = -1
            # up_bar = np.nan

        else:
            if df.medium.loc[i - 1] * 1.0009 > up_bar:
                up_bar = df.medium.loc[i - 1] * 1.0009

    if (cost == 0) or (decision == -1):  # покупаем
        # если предыдущее-комиссия-зазор больше препредыдущего-комиссия,
        # и если не возрастает или разница с нижн экстр больше порога
        # и если предыдущее не верх экстр
        if ((df.medium.loc[i - 1] * (0.9991 - jump_buy_percent) >= down_bar)
                and not (df.loc[i - 1, 'extr_up'] > 0)):  # TODO решение покупка
            if decision == -1:
                decision = 0
            else:
                decision = 1
            down_bar = np.nan

        else:
            down_bar = df.medium.loc[i - 1] * 0.9991

    if decision == -1:  # если всёже
        down_bar = df.medium.loc[i - 1] * 0.9991
        df.loc[i - 1, 'up_bar'] = up_bar
        df.loc[i - 1, 'down_bar'] = np.nan  # down_bar
    elif decision == 1:
        up_bar = df.medium.loc[i - 1] * 1.0009
        df.loc[i - 1, 'up_bar'] = np.nan  # up_bar
        df.loc[i - 1, 'down_bar'] = down_bar


to_i = len(df)
# to_i = round(len(df)/2) + 100
# to_i = 500
for i in range(1, to_i):
    df.loc[i, 'medium'] = (df.high[i] + df.low[i]) / 2
    prev_price = price
    price = df.medium.loc[i]
    if napr == 1 and price * (1 + percent_extr) < extr_up:
        if (extr_up_count == 0) and (i_extr > i_buy) and (df.medium.loc[i_extr] < buy * 1.0009):
            extr_up_count = 1
        elif (extr_up_count > 0) and (df.medium.loc[i_extr] < prev_buy_extr):
            extr_up_count += 1
        else:
            extr_up_count = 0
        napr = -1
        df.loc[i_extr, 'extr_up'] = extr_up
        df.loc[i+1, 'extr_des'] = extr_up

        extr_df = pd.concat([extr_df, pd.DataFrame({'time': [df.time.loc[i_extr]], 'extr_up': [extr_up]})],
                            ignore_index=True)
        i_extr_medium = round((prev_i_extr + i_extr)/2)
        extr_medium = (prev_extr + extr_up)/2
        df.loc[i_extr_medium, 'extr_medium'] = extr_medium
        prev_extr = extr_up
        prev_buy_extr = extr_up
        prev_i_extr = i_extr

        extr_up = 0
    elif napr == -1 and price * (1 - percent_extr) > extr_dwn:
        napr = 1
        df.loc[i_extr, 'extr_dwn'] = extr_dwn
        df.loc[i + 1, 'extr_des'] = extr_dwn

        extr_df = pd.concat([extr_df, pd.DataFrame({'time': [df.time.loc[i_extr]], 'extr_dwn': [extr_dwn]})],
                            ignore_index=True)
        i_extr_medium = round((prev_i_extr + i_extr)/2)
        # extr_medium = (prev_extr + extr_dwn)/2
        # df.loc[i_extr_medium, 'extr_medium'] = extr_medium
        prev_extr = extr_dwn
        prev_i_extr = i_extr

        extr_dwn = price * 2
    if napr == 1 and price > extr_up:
        extr_up = price
        i_extr = i
    elif napr == -1 and price < extr_dwn:
        extr_dwn = price
        i_extr = i

    # TODO                                        medium          open |
    # jump_des()                                # s%p = 19.43%  19.05% |
    # cost_sale_des()                           # s%p = 14.54%  12.96% |
    # cost_sale_des_cancelled_sale()            # s%p = 17.42%  21.59% |
    cost_sale_medextr_des_cancelled_sale_out()
    # cost_sale_medextr_des_cancelled_sale()    # s%p = 19.79%  17.45% |
    # cost_sale_des_cancelled_buy()             # s%p =  9.46%   7.38% |
    # cost_sale_nedes_twice()
    # TODO

    """if buy > df.medium.loc[i - 1] * (1 + stop_loss_percent):   # продаём в минус
        cost = df.medium.loc[i]
        profit = round(cost - buy - (cost + buy) * 0.0004, 2)
        percent_profit = round(cost / (buy / 100) - 100 - 0.09, 2)
        summ_profit += profit
        summ_percent_profit += percent_profit
        df.loc[i, 'cost'] = cost
        df.loc[i, 'status'] = cost * 0.995
        df.loc[i, 'bad_sale'] = cost
        buy = 0
        cost = 0
        decision = 0
        counter += 1
        time_sale = df.loc[i, 'time']
        time_sale = str(time_sale.day) + '-' + str(time_sale.hour) + ':' + str(time_sale.minute)
        print(profit, str(percent_profit) + '% ' + time_buy + ' -> ' + time_sale + '---------------')
        # up_bar = np.nan
        down_bar = df.medium.loc[i - 1] * 0.9991"""
    if decision == 1:  # покупка
        i_buy = i
        extr_up_count = 0
        cost = buy = df.open.loc[i]
        # cost = buy = df.medium.loc[i]
        time_buy = df.loc[i, 'time']
        time_buy = str(time_buy.day) + '-' + str(time_buy.hour) + ':' + str(time_buy.minute)
        df.loc[i, 'cost'] = cost
        df.loc[i, 'status'] = cost * 0.995
        df.loc[i, 'buy'] = cost
        decision = 2
    elif decision == 2:  # графика после покупки
        df.loc[i, 'cost'] = df.high.loc[i] * 0.9985
        df.loc[i, 'status'] = df.high.loc[i] * 0.9985 * 0.995
    elif decision <= -1:  # в продаже
        #  cost = df.high.loc[i - 1] * 0.9985
        cost = df.open.loc[i]
        # cost = df.medium.loc[i]
        #  mistake = df.medium.loc[i] * 0.0002*abs(decision + 1)
        profit = round(cost - buy - (cost + buy)*0.0004, 2)
        percent_profit = round(cost/(buy/100) - 100 - 0.09, 2)
        if (percent_profit > 0) or bad_sale_flag:  # продажа
            summ_profit += profit
            summ_percent_profit += percent_profit
            df.loc[i, 'cost'] = cost
            df.loc[i, 'status'] = cost * 0.995
            if not bad_sale_flag:
                df.loc[i, 'sale'] = cost
                bad_sale_marker = ''
            else:
                df.loc[i, 'bad_sale'] = cost
                bad_sale_marker = ' ----------'
            bad_sale_flag = False
            buy = 0
            cost = 0
            decision = 0
            counter += 1
            time_sale = df.loc[i, 'time']
            time_sale = str(time_sale.day)+'-'+str(time_sale.hour)+':'+str(time_sale.minute)
            print(profit, str(percent_profit) + '% ' + str(round(summ_percent_profit, 2)) + '% ' + time_buy + ' -> ' + time_sale + bad_sale_marker)
        else:  # в минусе в продаже
            # decision -= 1
            decision = 0
            # df.loc[i, 'status'] = df.medium.loc[i - 1] * 0.994
            df.loc[i, 'mistake'] = cost

hold_percent_profit = round(df.medium.iloc[to_i-1]/(df.medium.iloc[0]/100) - 100 - 0.09, 2)
if buy > 0:
    position = round(df.medium.iloc[to_i-1]/(buy/100) - 100 - 0.04, 2)
else:
    position = np.nan
print(des_name, 'jump_buy_percent = ' + str(jump_buy_percent*100), 'jump_sale_percent = ' + str(jump_sale_percent*100),
      'gap_sale_percent = ' + str(gap_sale_percent*100),
      'diff_extr = ' + str(diff_extr), '\ndeal count = ' + str(counter), 'summ_profit = ' + str(round(summ_profit, 2)),
      'position = ' + str(round(position, 2))+'%', 'summ_percent_profit = ' + str(round(summ_percent_profit, 2))+'%',
      'hold_percent_profit = ' + str(hold_percent_profit)+'%')
ax, ax2 = fplt.create_plot('YNDX '+des_name+' jbp:'+str(jump_buy_percent*100)+' jsp:'+str(jump_sale_percent*100)+' gsp:'
                           + str(gap_sale_percent*100)+' dex:'+str(round(diff_extr, 2))+' s%p:'
                           + str(round(summ_percent_profit, 2)), rows=2)
fplt.candlestick_ochl(df[['time', 'open', 'close', 'high', 'low']], ax=ax)
#  fplt.plot(df['time'], df['medium'], width=3, legend=ticker + 'medium', ax=ax)
fplt.plot(df['time'], df['up_bar'], legend='up_bar', ax=ax, color='green')
fplt.plot(df['time'], df['down_bar'], legend='down_bar', ax=ax, color='red')
# fplt.plot(df['time'], df['cost'], legend='cost', ax=ax, color='orange')
fplt.plot(df['time'], df['status'], legend='status', ax=ax, color='blue', width=2)
fplt.plot(df['time'], df['mistake'], legend='mistake', ax=ax, color='red', width=2)
fplt.plot(df['time'], df['buy'], legend='buy', ax=ax, color='blue', width=2, style='^')
fplt.plot(df['time'], df['sale'], legend='sale', ax=ax, color='red', width=2, style='v')
fplt.plot(df['time'], df['bad_sale'], legend='bad_sale', ax=ax, color='#FF00FF', width=2, style='v')
fplt.volume_ocv(df[['time', 'open', 'close', 'volume']], ax=ax2)

fplt.plot(df['time'], df['extr_up'], legend='extr_up', ax=ax, color='#ff6666', width=1, style='o')
fplt.plot(df['time'], df['extr_dwn'], legend='extr_dwn', ax=ax, color='#6666ff', width=1, style='o')
fplt.plot(df['time'], df['extr_des'], legend='extr_des', ax=ax, color='gray', width=1, style='o')
fplt.plot(df['time'], df['medium'], legend='medium', ax=ax, color='orange', width=0.5)

#  fplt.plot(extr_df['time'], extr_df['extr_up'], legend='extr_upppp', ax=ax, color='#ff6666', style='o')
fplt.plot(df['time'], df['extr_medium'], legend='extr_medium', ax=ax, color='yellow', width=1, style='o')

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
"""продажа: найти максимум просадки до верхнего экстремума, = коэфф выхода"""
"""ЧЁБЛИН: СТАВЛЮ ПЛАНКИ НА ПРЕДИДУЩИЙ ПОСЛЕ РЕШЕНИЯ"""
'''под и над средней от экстремумов'''
