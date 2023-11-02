import pandas as pd

import finplot as fplt
import numpy as np
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/11.11_YNDX_not_close_candle_df.csv'
# file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/11.11_FIVE_not_close_candle_df.csv'
# file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/11.11_OZON_not_close_candle_df.csv'
file = 'C:/Users/IGOR/PycharmProjects/InvestPythonProject/spectator1 data/2022/14.11-16.11_YNDX_not_close_candle_df.csv'
df = pd.read_csv(file)
df.time = pd.to_datetime(df.time)
df.last_trade_ts = pd.to_datetime(df.last_trade_ts)
df['iterator'] = range(0, len(df))


ticker = 'YNDX'
min_price_increment = 0.2

df['medium'] = np.nan
df['decision'] = np.nan
df['up_bar'] = np.nan
df['down_bar'] = np.nan
df['cost'] = np.nan
df['buy'] = np.nan
df['sale'] = np.nan
df['bad_sale'] = np.nan
df['dif_volume'] = np.nan

df['extr_up'] = np.nan
df['extr_dwn'] = np.nan
df['extr_des'] = np.nan

df['chisel'] = np.nan
df['corridor'] = 0
df['up_chisel'] = 0
df['dwn_chisel'] = 0

df['g_extr_up'] = np.nan
df['g_extr_dwn'] = np.nan
df['g_extr_des'] = np.nan
df['g_extr_medium'] = np.nan
cost = 0
df['t_extr_up'] = np.nan
df['t_extr_dwn'] = np.nan
df['t_extr_des'] = np.nan

df['mawp'] = np.nan
df['ma3'] = np.nan
df['ma7'] = np.nan
df['ma15'] = np.nan
df['ma20'] = np.nan
df['ma206'] = np.nan
df['degrees'] = np.nan
df['degrees_ma20'] = np.nan
df['der_degrees'] = np.nan
df['degrees_degrees'] = np.nan
df['null_mark'] = 0

jump_buy_percent = 0.03/100  # 0.08  без комиссии ?0.18?
jump_sale_percent = 0.05/100  # 0.05 без комиссии
gap_sale_percent = 0.03/100  # 0.05

g_percent_extr = 0.10 / 100
diff_extr = 0  # 0.07
stop_loss_percent = 0.33 / 100  # TODO

# up_bar = cost * (1.0009+percent)
# down_bar = cost * (0.9991-percent)np.nan
up_bar = np.nan
down_bar = np.nan
decision = 0
bad_sale_flag = False
panic_flag = False
was_bad_sale_flag = False
summ_profit = profit = 0
summ_percent_profit = percent_profit = 0
buy = cost
sale = df.close.loc[0]*2
i_buy = 0
time_buy = 0
counter = 0
des_name = ''

df.loc[0, 'medium'] = (df.high[0] + df.low[0]) / 2
price = df.medium.loc[0]

napr = 1
max_after_deal = 0
min_after_deal = df.close.loc[0]*2
i_max_after_deal = 0
i_min_after_deal = 0
extr_up = 0
extr_up_count = 0
extr_dwn = df.close.loc[0]*2
i_extr_up = 0
i_extr_dwn = 0
prev_extr = price
extr_medium = price*2
prev_i_extr = 0
prev_buy_extr = 0

g_napr = 1
g_extr_up = 0
g_extr_up_count = 0
g_extr_dwn = df.medium.loc[0]*2
g_i_extr = 0
g_prev_extr = price
g_extr_medium = price*2
g_prev_i_extr = 0
g_prev_buy_extr = 0

t_napr = 1
t_extr_up = 0
t_extr_dwn = df.medium.loc[0]*2
t_i_extr = 0
t_prev_extr = price
t_prev_i_extr = 0

close_differance_list = [0]
medium_speed_list = [0]
speed_list = [0]
noise_list = [0]
degrees_differance_list = [0]
degrees_summ3_list = [0, 0]
chisel_list = []
chisel_dict = {}
i_corridor_start = 2
up_chisel = 0
dwn_chisel = 0
degrees_incr_time = 0
degrees_decr_time = 0
waves_list = []
wave_period_list = []
ideal_sum_percent_profit = 0

g_i_time = 0
g_i_time_count = 0

prev_close = 0
prev_close_perc_differance = 0


def first_open_des():
    # время(i), процент движения от минимакса => подтверждение сделки <= ?время, процент после максимина?
    # ну например подтверждать если >0.25 * 1м(20т) >
    #  мб > время падения > время подтверждения
    # > процент падения < время подтверждения
    global close_differance_list, extr_up, extr_dwn, i_extr_up, i_extr_dwn, decision, des_name, bad_sale_flag
    des_name = 'fod'
    close_differance_list.append(round(df.close.loc[i-1]/(df.close.loc[i-2]/100) - 100, 2))
    if df.close.loc[i-1] >= extr_up:
        extr_up = df.close.loc[i-1]
        i_extr_up = i-1
    if df.close.loc[i-1] <= extr_dwn:
        extr_dwn = df.close.loc[i-1]
        i_extr_dwn = i-1
    extr_up_profit = sum(close_differance_list[i_extr_up:i - 1]) * 100
    extr_dwn_profit = sum(close_differance_list[i_extr_dwn:i - 1])*100
    if cost > 0:  # продаём
        if (extr_dwn_profit > 10) and (extr_dwn_profit > (i - 1 - i_extr_dwn) / 2):  #
            decision = -1
    elif cost == 0:  # покупаем
        if (extr_up_profit < -10) and (abs(extr_up_profit) > (i - 1 - i_extr_up)):  #
            decision = 1


def first_degrees_open_des():
    global decision, des_name, bad_sale_flag
    des_name = 'fdod'
    if cost > 0:  # продаём
        if df.degrees[i] > 11:  #
            decision = -1
    elif cost == 0:  # покупаем
        if df.degrees[i] < -11:  #
            decision = 1

    """if df.high.loc[i] > t_extr_up:
            t_extr_up = df.high.loc[i]
            t_i_extr = i
            t_napr = 1
        elif df.low.loc[i] > t_extr_dwn:
            t_extr_dwn = df.low.loc[i]
            t_i_extr = i
            t_napr = -1
        else:
            if i > t_i_extr + : # + t это экстремум? максимум? а комиссия? доход
                if t_napr == 1:
                    decision = 1
                elif t_napr == -1:
                    decision = -1"""


def percent_chisel_open_des():
    #
    global up_chisel, dwn_chisel, i_corridor_start, chisel_list, chisel_dict, prev_close_perc_differance, prev_close, buy, sale, cost, i, max_after_deal,\
        min_after_deal, i_max_after_deal, i_min_after_deal, panic_flag, was_bad_sale_flag, close_differance_list, extr_up, extr_dwn,\
        i_extr_up, i_extr_dwn, decision, des_name, bad_sale_flag
    des_name = 'pchod'

    close_dif_ma = round(sum(close_differance_list[i-11:i - 1]), 2)
    df.loc[i-1, 'close_dif_ma'] = close_dif_ma

    close = df.close[i - 1]

    #  ========== КОРИДОР =========
    #if (abs(close_perc_differance) < 0.09):
    #    df.loc[i-1, 'corridor'] = df.loc[i-2, 'corridor'] + 1
    if abs(round(sum(close_differance_list[-i_corridor_start:]), 2)) <= 0.03: # TODO терпит галочку в начале
        df.loc[i - 1, 'corridor'] = df.loc[i - 2, 'corridor'] + 1
        i_corridor_start += 1
        if close_perc_differance > 0:
            up_chisel += 1
            df.loc[i - 1, 'up_chisel'] = up_chisel
        elif close_perc_differance < 0:
            dwn_chisel += 1
            df.loc[i - 1, 'dwn_chisel'] = dwn_chisel
        else:
            df.loc[i - 1, 'up_chisel'] = up_chisel
            df.loc[i - 1, 'dwn_chisel'] = dwn_chisel
    else:
        i_corridor_start = 2
        up_chisel = 0
        dwn_chisel = 0
    #  ========== МАКСИМУМ после сделки =========
    if close > max_after_deal:
        max_after_deal = close
        i_max_after_deal = i - 1
    if close < min_after_deal:
        min_after_deal = close
        i_min_after_deal = i - 1

    """
    #  слишком большой скачок или не то направление движения не интересуют решение
    if close_perc_differance >= 0.09:
        return
    if close_perc_differance <= -0.09:
        return
    if ((cost == 0) and (close_perc_differance > 0)) or ((cost > 0) and (close_perc_differance < 0)):
        return
    """

    #  ========== ДОЛБЁЖКА =========
    #  {цена:[процент, удары, нули, единицы]}
    #  df({цена: [процент, удары, нули, единицы]}, соседи[процент, удары])
    #  чистка если прожолжение движения
    if (prev_close_perc_differance * close_perc_differance > 0) and [prev_close] in chisel_list:  # спорно
        chisel_dict[prev_close][0] -= prev_close_perc_differance
        chisel_dict[prev_close][1] -= 1
    #  чистка для окна в +- 0.08%
    for chisel_cost in chisel_list:
        if (chisel_cost > round(close * 1.0008, 2)) or (chisel_cost < round(close * 0.9992, 2)):
            chisel_list.remove(chisel_cost)
            chisel_dict.pop(chisel_cost)
    #  ЗАПИСЬ
    if close in chisel_dict:
        chisel_dict[close][0] = round(chisel_dict[close][0] + close_perc_differance, 2)
        if close_perc_differance != 0:
            if abs(close_perc_differance) == 0.01:
                chisel_dict[close][1] += 1
                chisel_dict[close][3] += 1
            else:
                chisel_dict[close][1] += 1
        else:
            chisel_dict[close][2] += 1
    else:
        chisel_list.append(close)
        if close_perc_differance != 0:
            if abs(close_perc_differance) == 0.01:
                chisel_dict.update([(close, [close_perc_differance, 1, 0, 1])])
            else:
                chisel_dict.update([(close, [close_perc_differance, 1, 0, 0])])
        else:
            chisel_dict.update([(close, [close_perc_differance, 0, 1, 0])])
        if len(chisel_list) == 21:
            chisel_dict.pop(chisel_list.pop(0))
    #  соседи
    neighbors = [0, 0]
    for j in [1, 2]:
        close_incr = round(close + j * min_price_increment, 2)
        close_decr = round(close - j * min_price_increment, 2)
        if close_incr in chisel_dict:
            bang = chisel_dict[close_incr][0]
            if ((bang < 0) and (cost == 0)) or ((bang > 0) and (cost > 0)):
                neighbors[0] = round(neighbors[0] + bang, 2)
                neighbors[1] += chisel_dict[close_incr][1]
        if close_decr in chisel_dict:
            bang = chisel_dict[close_decr][0]
            if ((bang < 0) and (cost == 0)) or ((bang > 0) and (cost > 0)):
                neighbors[0] = round(neighbors[0] + bang, 2)
                neighbors[1] += chisel_dict[close_decr][1]

    # df.loc[i-1, 'chisel'] = (chisel_dict[close], neighbors)

    """ des1
    if (cost > 0):  # продаём
        if (round(max_after_deal/(close/100)-100, 2) <= 0.01):
            # с соседями 3 удара И с соседями набили 0.15
            # ИЛИ удары с соседями + удержание 5
            if ((chisel_dict[close][1] + neighbors[1] >= 3) and (round(chisel_dict[close][0] + neighbors[0], 2) >= 0.15)
                    or (chisel_dict[close][1] + neighbors[1] + chisel_dict[close][2] >= 5)):  #
                decision = -1
                df.loc[i - 1, 'decision'] = -1
                chisel_list = []
                chisel_dict = {}
        if df.loc[i - 1, 'corridor'] >= 35:
            bad_sale_flag = True
            decision = -1
            chisel_list = []
            chisel_dict = {}

            was_bad_sale_flag = True

            max_after_deal = close
            i_max_after_deal = i - 1
        '''if min_after_deal < round(buy * 0.9975, 2):
            panic_flag = True
        if (panic_flag is True) and (round((close/(min_after_deal/100) - 100), 2) >= round(0.5*(buy/(min_after_deal/100) - 100), 2)):
            bad_sale_flag = True
            decision = -1
            chisel_list = []
            chisel_dict = {}'''
    # ПОКУПКА если
    elif (cost == 0) and (round(sum(close_differance_list[i_max_after_deal:]), 2) <= -0.08) and (df.loc[i - 1, 'corridor'] >= 20)\
            and ((was_bad_sale_flag is False) or (round(sum(close_differance_list[i_max_after_deal:]), 2) <= -0.40)):  # покупаем
        # 3 удара И с соседями набили -0.07
        # ИЛИ 8 ударов с соседями И с соседями набили -0.20
        # ИЛИ планка 5
        if (((chisel_dict[close][1] >= 3) and (round(chisel_dict[close][0] + neighbors[0], 2) <= -0.07))
                or ((chisel_dict[close][1] >= 2) and (round(chisel_dict[close][0] + neighbors[0], 2) <= -0.07))
                or ((chisel_dict[close][1] + neighbors[1] >= 8) and (round(chisel_dict[close][0] + neighbors[0], 2) <= -0.20))  # 2горбый холм
                or ((chisel_dict[close][2] >= 5))):  # удержание
            decision = 1
            df.loc[i - 1, 'decision'] = 1
            chisel_list = []
            chisel_dict = {}

            was_bad_sale_flag = False

    prev_close = close
    prev_close_perc_differance = close_perc_differance
    """
    """
    if (cost > 0):  # продаём
        if ((chisel_dict[close][1] >= 3)):  #
            decision = -1
            df.loc[i - 1, 'decision'] = -1
            chisel_list = []
            chisel_dict = {}
    elif (cost == 0):  # покупаем
        if ((chisel_dict[close][1] >= 3)):  # удержание
            decision = 1
            df.loc[i - 1, 'decision'] = 1
            chisel_list = []
            chisel_dict = {}
    """
    if (cost > 0):  # продаём
        if ((round((close/(max_after_deal/100) - 100), 2) <= -0.05) and (round((close/(buy/100) - 100), 2) >= 0.20)):  # round(sum(close_differance_list[i-11:i - 1]), 2)
            decision = -1
            # df.loc[i - 1, 'decision'] = -1
            chisel_list = []
            chisel_dict = {}
        elif ((round((close/(max_after_deal/100) - 100), 2) <= -4.00)):  # round(sum(close_differance_list[i-11:i - 1]), 2)
            bad_sale_flag = True
            decision = -1
            # df.loc[i - 1, 'decision'] = -1
            chisel_list = []
            chisel_dict = {}
    elif (cost == 0):  # покупаем
        if ((round((close/(min_after_deal/100) - 100), 2) >= 0.05)):  # удержание
            decision = 1
            # df.loc[i - 1, 'decision'] = 1
            chisel_list = []
            chisel_dict = {}


def stupid_open_des():
    global decision, des_name, bad_sale_flag
    des_name = 'sod'
    if cost > 0:  # продаём
        if g_extr_up == 0:  #
            decision = -1
    elif cost == 0:  # покупаем
        if g_extr_dwn == price * 2:  #
            decision = 1
t = 6

to_i = len(df)
# to_i = round(len(df)/2) + 100
# to_i = 10000
df.loc[1, 'dif_volume'] = (df.volume[1] - df.volume[0])
for i in range(2, to_i-1):
    """if i >= 3+1:
        df.loc[i, 'ma3'] = sum(df.close[i-3+1:i+1])/3
        if i >= 7+1:
            df.loc[i, 'ma7'] = sum(df.close[i-7+1:i+1])/7"""
    """
    if i >= 15 + 1:
        df.loc[i, 'ma15'] = sum(df.close[i - 15 + 1:i + 1]) / 15
        if i >= 20 + 1:
            df.loc[i, 'ma20'] = sum(df.close[i - 20 + 1:i + 1]) / 20
    if i >= 206:
        df.loc[i, 'ma206'] = sum(df.close[(i-206):i]) / 206
    else:
        df.loc[i, 'ma206'] = sum(df.close[:i]) / i"""
    if (((df.time[i-1].hour == 15) and (df.time[i-1].minute == 39)) and ((df.time[i].hour == 15) and (df.time[i].minute == 45)))\
       or (((df.time[i-1].hour == 15) and (df.time[i-1].minute == 49)) and ((df.time[i].hour == 16) and (df.time[i].minute == 4)))\
       or (((df.time[i-1].hour == 20) and (df.time[i-1].minute == 49)) and ((df.time[i].hour == 6) and (df.time[i].minute == 59))):
        df.loc[i - 1, 'border'] = df.close[i - 1]
        df.loc[i, 'border'] = df.close[i]
    '''if (df.last_trade_ts[i] - df.last_trade_ts[i-1]).seconds/60 > 3:
        df.loc[i - 1, 'border'] = df.close[i - 1] * 0.9900
        df.loc[i, 'border'] = df.close[i] * 1.0100
        print(df.last_trade_ts[i-1], df.last_trade_ts[i], 3)
    elif (df.last_trade_ts[i] - df.last_trade_ts[i-1]).seconds/60 > 2:
        df.loc[i - 1, 'border'] = df.close[i - 1] * 0.9950
        df.loc[i, 'border'] = df.close[i] * 1.0050
        print(df.last_trade_ts[i - 1], df.last_trade_ts[i], 2)'''
    if len(wave_period_list) >= 1:
        wave_period = round(sum(wave_period_list[-12:]) / 3)  # if g_i_time_count > 0: round(4 * g_i_time / g_i_time_count)
        #  df.loc[i, 'mawp'] = sum(df.close[i - wave_period + 1:i + 1]) / wave_period

    close_perc_differance = round(df.close.loc[i - 1] / (df.close.loc[i - 2] / 100) - 100, 2)
    close_differance_list.append(close_perc_differance)
    df.loc[i, 'dif_volume'] = (df.volume[i] - df.volume[i-1])
    df.loc[i, 'ma_norm_volume'] = round(sum(df.volume[i - 10 - 1:i - 1])/2/10, 2)
    # df.loc[i, 'medium'] = (df.high[i] + df.low[i]) / 2
    # df.loc[i, 'degrees'] = math.degrees(math.atan(df.ma30[i] - df.ma30[i - 1]))
    # df.loc[i, 'degrees_ma20'] = math.degrees(math.atan(df.ma20[i] - df.ma20[i - 1]))
    # df.loc[i, 'der_degrees'] = (df.ma30[i] - df.ma30[i - 1]) - (df.ma30[i-1] - df.ma30[i - 2])
    # df.loc[i, 'degrees_degrees'] = math.degrees(math.atan(df.der_degrees[i]))
    """if df.degrees.loc[i - 1] != 0:
        if df.degrees.loc[i - 1] <= df.degrees.loc[i]:
            degrees_perc_shange = abs(round(df.degrees.loc[i] / (df.degrees.loc[i - 1] / 100) - 100, 2))
        else:
            degrees_perc_shange = -abs(round(df.degrees.loc[i] / (df.degrees.loc[i - 1] / 100) - 100, 2))
    else:
        if df.degrees.loc[i - 1] <= df.degrees.loc[i]:
            degrees_perc_shange = 100
        else:
            degrees_perc_shange = -100"""

    degrees_shange = df.degrees.loc[i] - df.degrees.loc[i - 1]




    if degrees_shange < 0:
        degrees_decr_time += 1
        degrees_incr_time = 0
    elif degrees_shange > 0:
        degrees_incr_time += 1
        degrees_decr_time = 0
    if (df.degrees[i-1] >= 0) and (df.degrees[i] <= 0) or (df.degrees[i-1] <= 0) and (df.degrees[i] >= 0):
        degrees_incr_time = 0
        degrees_decr_time = 0
    degrees_differance_list.append(degrees_shange)
    degrees_summ3_list.append(sum(degrees_differance_list[i-3+1:i+1]))
    if degrees_decr_time == 3:  # TODO
        df.loc[i, 'degrees_sale'] = df.degrees.loc[i]  # RED
        degrees_decr_time = 0
    elif degrees_incr_time == 3:
        df.loc[i, 'degrees_buy'] = df.degrees.loc[i]  # BLUE
        degrees_incr_time = 0

    prev_price = price
    price = df.close.loc[i]

    """====== ЭКСТРЕМУМ ЗАЗОРА ======"""
    if g_napr == 1 and df.close.loc[i] * (1 + g_percent_extr) < g_extr_up:
        if (g_extr_up_count == 0) and (g_i_extr > i_buy) and (df.close.loc[g_i_extr] < buy * 1.0009):
            g_extr_up_count = 1
        elif (g_extr_up_count > 0) and (df.close.loc[g_i_extr] < g_prev_buy_extr):
            g_extr_up_count += 1
        else:
            g_extr_up_count = 0
        g_napr = -1
        df.loc[g_i_extr_up, 'g_extr_up'] = g_extr_up
        df.loc[i+1, 'g_extr_des'] = g_extr_up

        # g_i_extr_medium = round((g_prev_i_extr + g_i_extr)/2)
        # g_extr_medium = (g_prev_extr + g_extr_up)/2
        # df.loc[g_i_extr_medium, 'g_extr_medium'] = g_extr_medium
        waves_list.append(round(g_extr_up / (g_prev_extr / 100) - 100, 2))
        ideal_sum_percent_profit += waves_list[-1]
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
        df.loc[i + 1, 'g_extr_des'] = g_extr_dwn

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

        g_extr_dwn = price * 2
    if g_napr == 1 and df.close.loc[i] > g_extr_up:
        g_extr_up = df.close.loc[i]
        g_i_extr_up = i
    elif g_napr == -1 and df.close.loc[i] < g_extr_dwn:
        g_extr_dwn = df.close.loc[i]
        g_i_extr_dwn = i

    """====== ВРЕМЕННОЙ ЭКСТРЕМУМ ======"""
    """if t_napr == 1 and (i > t_i_extr + t) and df.close.loc[i] < t_extr_up:
        t_napr = -1
        t_prev_extr = t_extr_up
        t_prev_i_extr = t_i_extr
        df.loc[t_i_extr, 't_extr_up'] = t_extr_up
        df.loc[i, 't_extr_des'] = t_extr_up
        t_extr_up = 0
    elif t_napr == -1 and (i > t_i_extr + t) and df.close.loc[i] > t_extr_dwn:
        t_napr = 1
        t_prev_extr = t_extr_dwn
        t_prev_i_extr = t_i_extr
        df.loc[t_i_extr, 't_extr_dwn'] = t_extr_dwn
        df.loc[i, 't_extr_des'] = t_extr_dwn
        t_extr_dwn = price * 2
    if t_napr == 1 and df.close.loc[i] >= t_extr_up:
        t_extr_up = df.close.loc[i]
        t_i_extr = i
    elif t_napr == -1 and df.close.loc[i] <= t_extr_dwn:
        t_extr_dwn = df.close.loc[i]
        t_i_extr = i"""

    # TODO                                        medium          open |
    # first_open_des()
    # first_degrees_open_des()
    percent_chisel_open_des()
    # stupid_open_des()
    # TODO

    if i >= 15:
        medium_speed_list.append(round(sum(map(abs, list(filter(lambda num: num != 0, close_differance_list[i-15:i]))))/15, 3))
        speed_list.append(abs(round((df.close.loc[i-15] / (df.close.loc[i-1] / 100) - 100)/15, 3)))
    else:
        medium_speed_list.append(round(sum(map(abs, list(filter(lambda num: num != 0, close_differance_list[:i]))))/i, 3))
        speed_list.append(abs(round((df.close.loc[0] / (df.close.loc[i - 1] / 100) - 100)/15, 3)))

    if i >= 50:
        try:
            noise_list.append(max(set(map(abs, list(filter(lambda num: num != 0, close_differance_list[i-100:i])))), key=close_differance_list[i-100:i].count))
        except ValueError:  # все нули
            noise_list.append(0)
    else:
        noise_list.append(max(set(map(abs, list(filter(lambda num: num != 0, close_differance_list[-100:i])))), key=close_differance_list[i-100:i].count))

    """if buy > df.medium.loc[i - 1] * (1 + stop_loss_percent):   # продаём в минус
        cost = df.medium.loc[i]
        profit = round(cost - buy - (cost + buy) * 0.0004, 2)
        percent_profit = round(cost / (buy / 100) - 100 - 0.09, 2)
        summ_profit += profit
        summ_percent_profit += percent_profit
        df.loc[i, 'cost'] = cost
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
        if df.close.loc[i] <= df.close.loc[i-1]:
            i_buy = i
            extr_up_count = 0
            cost = buy = df.close.loc[i]
            # cost = buy = df.medium.loc[i]
            time_buy = df.loc[i, 'last_trade_ts']
            time_buy = str(time_buy.day) + '-' + str(time_buy.hour) + ':' + str(time_buy.minute)
            df.loc[i, 'cost'] = cost
            df.loc[i, 'buy'] = cost
            df.loc[i, 'decision'] = df.close.loc[i]
            decision = 2

            max_after_deal = 0
            min_after_deal = df.close.loc[i] * 2

            extr_up = 0
            extr_dwn = df.close.loc[0] * 2
        else:
            df.loc[i, 'decision'] = df.close.loc[i-1]
            continue
    elif decision == 2:  # графика после покупки
        df.loc[i, 'cost'] = df.high.loc[i] * 0.9985
    elif decision <= -1:  # в продаже
        if df.close.loc[i] >= df.close.loc[i - 1]:
            #  cost = df.high.loc[i - 1] * 0.9985
            cost = df.close.loc[i]
            # cost = df.medium.loc[i]
            profit = round(cost - buy - (cost + buy)*0.0004, 2)
            percent_profit = round(cost/(buy/100) - 100 - 0.09, 2)
            if (percent_profit > 0) or bad_sale_flag:  # продажа
                summ_profit += profit
                summ_percent_profit += percent_profit
                df.loc[i, 'cost'] = cost
                if not bad_sale_flag:
                    df.loc[i, 'sale'] = cost
                    bad_sale_marker = ''
                else:
                    df.loc[i, 'bad_sale'] = cost
                    bad_sale_marker = ' ----------'
                df.loc[i, 'decision'] = df.close.loc[i]
                bad_sale_flag = False
                panic_flag = False
                buy = 0
                sale = cost
                cost = 0
                decision = 0

                max_after_deal = 0
                min_after_deal = df.close.loc[0] * 2

                extr_up = 0
                extr_dwn = df.close.loc[0] * 2

                counter += 1
                time_sale = df.loc[i, 'last_trade_ts']
                time_sale = str(time_sale.day)+'-'+str(time_sale.hour)+':'+str(time_sale.minute)+':'+str(time_sale.second)
                i_time_sale = str(i_buy) + ' -> ' + str(i)
                print(profit, str(percent_profit) + '% ' + time_buy + ' -> ' + time_sale + '  ' + i_time_sale + bad_sale_marker)
        else:
            df.loc[i, 'decision'] = df.close.loc[i-1]
            continue

hold_percent_profit = round(df.close.iloc[to_i-1]/(df.close.iloc[0]/100) - 100 - 0.09, 2)
medium_wave = sum(map(abs, waves_list))/len(waves_list)
ideal_sum_percent_profit *= 0.9991

if buy > 0:
    position = round(df.close.iloc[to_i-1]/(buy/100) - 100 - 0.04, 2)
else:
    position = np.nan
print('medium_wave_percent = ' + str(round(medium_wave, 2))+'%',
      'ideal_sum_percent_profit = ' + str(round(ideal_sum_percent_profit, 2))+'%',
      'wave_period = ' + str(wave_period))
print(des_name, 'jump_buy_percent = ' + str(jump_buy_percent*100), 'jump_sale_percent = ' + str(jump_sale_percent*100),
      'gap_sale_percent = ' + str(gap_sale_percent*100),
      'diff_extr = ' + str(diff_extr), '\ndeal count = ' + str(counter), 'summ_profit = ' + str(round(summ_profit, 2)),
      'position = ' + str(round(position, 2))+'%', 'summ_percent_profit = ' + str(round(summ_percent_profit, 2))+'%',
      'hold_percent_profit = ' + str(hold_percent_profit)+'%')
ax, ax2, ax3 = fplt.create_plot('YNDX '+des_name+' jbp:'+str(jump_buy_percent*100)+' jsp:'+str(jump_sale_percent*100)+' gsp:'
                           + str(gap_sale_percent*100)+' dex:'+str(round(diff_extr, 2))+' s%p:'
                           + str(round(summ_percent_profit, 2)), rows=3)
fplt.candlestick_ochl(df[['iterator', 'open', 'close', 'high', 'low']], ax=ax)
#  fplt.plot(df['time'], df['medium'], width=3, legend=ticker + 'medium', ax=ax)
fplt.plot(df['iterator'], df['up_bar'], legend='up_bar', ax=ax, color='green')
fplt.plot(df['iterator'], df['border'], legend='session_border', ax=ax, width=4, color='blue')
fplt.plot(df['iterator'], df['down_bar'], legend='down_bar', ax=ax, color='red')
# fplt.plot(df['iterator'], df['cost'], legend='cost', ax=ax, color='orange')

fplt.plot(df['iterator'], df['buy'], legend='buy', ax=ax, color='blue', width=2, style='^')
fplt.plot(df['iterator'], df['sale'], legend='sale', ax=ax, color='red', width=2, style='v')
fplt.plot(df['iterator'], df['bad_sale'], legend='bad_sale', ax=ax, color='#FF00FF', width=2, style='v')
fplt.plot(df['iterator'], df['decision'], legend='deal_decision', ax=ax, color='#00FF00', width=1, style='o')

fplt.plot(df['iterator'], df['g_extr_up'], legend='g_extr_up', ax=ax, color='#ff6666', width=1, style='o')
fplt.plot(df['iterator'], df['g_extr_dwn'], legend='g_extr_dwn', ax=ax, color='#6666ff', width=1, style='o')
fplt.plot(df['iterator'], df['g_extr_des'], legend='g_extr_des', ax=ax, color='gray', width=1, style='o')
fplt.plot(df['iterator'], df['medium'], legend='medium', ax=ax, color='orange', width=0.5)

fplt.plot(df['iterator'], df['close'], legend='close', ax=ax, color='white', width=2)
# fplt.plot(df['iterator'], df['g_extr_medium'], legend='g_extr_medium', ax=ax, color='yellow', width=1, style='o')

# fplt.plot(df['iterator'], df['ma15'], legend='ma15', ax=ax, color='green', width=2)
# fplt.plot(df['iterator'], df['ma7'], legend='ma7', ax=ax, color='red', width=2)
# fplt.plot(df['iterator'], df['mawp'], legend='mawp', ax=ax, width=2)
# fplt.plot(df['iterator'], df['ma206'], legend='ma206', ax=ax, width=2)

# fplt.volume_ocv(df[['iterator', 'open', 'close', 'volume']], ax=ax2)
# fplt.volume_ocv(df[['iterator', 'open', 'close', 'dif_volume']], ax=ax3)
fplt.volume_ocv(df[['iterator', 'open', 'close', 'volume']], ax=ax3)
# fplt.plot(df['iterator'], df['ma_norm_volume'], ax=ax3)
# fplt.plot(df['iterator'], close_differance_list, legend='%', ax=ax2)
# fplt.plot(df['iterator'], df['degrees'], legend='degrees ma206', ax=ax2)
# fplt.plot(df['iterator'], df['degrees_ma20'], legend='degrees ma20', ax=ax2)
# fplt.plot(df['iterator'], df['close_dif_ma'], legend='close_dif_ma', ax=ax2)

fplt.plot(df['iterator'], noise_list, legend='noise', ax=ax2)
fplt.plot(df['iterator'], medium_speed_list, legend='medium_speed', ax=ax2)
fplt.plot(df['iterator'], speed_list, legend='speed', ax=ax2)
# fplt.plot(df['iterator'], df['corridor'], legend='corridor', ax=ax2)
fplt.plot(df['iterator'], df['null_mark'], ax=ax2)
# fplt.plot(df['iterator'], df['degrees_buy'], color='#6666ff', width=1, style='o', ax=ax2)
# fplt.plot(df['iterator'], df['degrees_sale'], color='#ff6666', width=1, style='o', ax=ax2)


# fplt.plot(df['iterator'], df['der_degrees'], legend='der_degrees', ax=ax3)
# fplt.plot(df['iterator'], degrees_summ3_list, legend='summ3', ax=ax3)
# fplt.plot(df['iterator'], df['null_mark'], ax=ax3)
# fplt.plot(df['iterator'], df['up_chisel'], legend='up_chisel', color='green', ax=ax3)
# fplt.plot(df['iterator'], df['dwn_chisel'], legend='dwn_chisel', color='red', ax=ax3)

# fplt.plot(df['iterator'], df['degrees_degrees'], legend='degrees_degrees', ax=ax4)
# fplt.plot(df['iterator'], df['null_mark'], ax=ax4)


fplt.show()


"""
with Client(token_read) as client:
    a= client.market_data.get_last_prices(figi=('BBG006L8G4H1',))
a
GetLastPricesResponse(last_prices=[LastPrice(figi='BBG006L8G4H1', price=Quotation(units=1997, nano=400000000),
time=datetime.datetime(2022, 10, 14, 15, 45, 17, 330, tzinfo=datetime.timezone.utc),
instrument_uid='10e17a87-3bce-4a1f-9dfc-720396f98a3c')])
"""
"""разница между последними экстремумами
последним экстремумом и ценой
?несколько свечей назад и ценой? то НЕТ"""
"""продажа: найти максимум просадки до верхнего экстремума, = коэфф выхода"""
"""ЧЁБЛИН: СТАВЛЮ ПЛАНКИ НА ПРЕДИДУЩИЙ ПОСЛЕ РЕШЕНИЯ"""
'''под и над средней от экстремумов'''
'''ВЫЙТИ ЕСЛИ ДОЛГО В КОРИДОРЕ'''