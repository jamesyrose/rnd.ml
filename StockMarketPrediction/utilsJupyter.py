from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Activation, Dropout, Reshape, RepeatVector
from keras.layers import Input, Flatten, Embedding, Concatenate, LSTM
from keras.layers.merge import add, multiply
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from py_vollib.black_scholes_merton  import black_scholes_merton
from batchnorm import BatchNormalization
import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def CNN_RNN_Cell(features, nb_filter, filter_length, dilation, l2_layer_reg, m_pool):
    def f(_input):
        residual = _input

        layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length,
                           dilation_rate=dilation,
                           activation='linear', padding='causal', use_bias=False,
                           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                              seed=42), kernel_regularizer=l2(l2_layer_reg))(_input)

        layer_out = Activation('tanh')(layer_out)

        skip_out = Conv1D(features, 1, activation='linear', use_bias=False,
                          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                             seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_in = Conv1D(features, 1, activation='linear', use_bias=False,
                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                               seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_out = multiply([residual, network_in])

        return network_out, skip_out

    return f


def TCN(layers, features, m_pool=True):
    # Temporal Convolution RWA
    def func(a):
        for i in range(layers):
            # the internals are similar to google's wavenet
            dilation = 2 ** (layers - i)
            a = BatchNormalization()(a)
            a, b = CNN_RNN_Cell(features, 32, 32, dilation, 0.001, m_pool)(a)

        b = BatchNormalization()(b)
        conv_output = Conv1D(1, 1, activation='linear',
                             kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
                             kernel_regularizer=l2(0.001))(b)
        conv_output = BatchNormalization()(conv_output)

        hidden_features = BatchNormalization()(conv_output)
        return hidden_features

    return func


def Classifier(seq_len, features, layers, output_dim, embedding_input_size, day_embedding_size=7):
    x = Input(shape=(seq_len, features))
    _class = Input(shape=(1,))
    _day = Input(shape=(1,))

    h = TCN(layers, features, m_pool=False)(x)
    h = TCN(layers, features, m_pool=False)(h)

    embedded_class = Embedding(embedding_input_size + 1, 10)(_class)
    embedded_class = Reshape((int(embedded_class.shape[2]), int(embedded_class.shape[1])))(embedded_class)

    embedded_day = Embedding(day_embedding_size + 1, 3)(_day)
    embedded_day = Reshape((int(embedded_day.shape[2]), int(embedded_day.shape[1])))(embedded_day)

    h = Concatenate(axis=1)([h, embedded_class, embedded_day])

    attention = Dense(int(h.shape[2]), activation='softmax')(h)
    h = multiply([h, attention])  # might be worth trying an addition-based merge

    flat = Flatten()(h)
    output = Dense(output_dim, activation='sigmoid', name='classifier')(flat)

    # model = Model(inputs=[x,stock_condition], outputs=output)
    model = Model(inputs=[x, _class, _day], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer="nadam")
    model.summary()
    return model


def get_symbols():
    filenames = os.listdir('./data')
    return [filename.replace('.csv', '') for filename in filenames if filename.endswith('.csv')]


def feature_extraction(data):
    data = relative_strength_index(data, n=14)
    data = macd(data, n_fast=12, n_slow=26)
    # for x in np.array([2, 5, 10, 30, 60, 120]):
    #     data = relative_strength_index(data, n=x)
    #    data = stochastic_oscillator_d(data, n=x)
    #   data = momentum(data, n=x)
    #  data = bollinger_bands(data, n=x)
    # data = macd(data, n_fast=x, n_slow=2*x)
    return data


def compute_prediction_int(df, n):
    """
        Split into brea, neutral, & bull categories
    """
    df = df.shift(-n)['Close'] / df['Close'] - 1
    df[df >= 0] = 1
    df[df < 0] = 0
    return df.iloc[:-n]


def compute_prediction_reg(df, n):
    pred = df.shift(-n)['Close'] / df['Close'] - 1
    pred = pred.iloc[:-n]
    return pred


def quantize(df):
    features = list(df)
    for feature in features:
        intervals = pd.cut(df[feature], 255)
        df[feature] = [float(interval.left) for interval in intervals]
    return df


def normalize(df):
    roll = df.rolling(30)
    df = (df - roll.mean()) / (roll.max() - roll.min()) * 2 - 1
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def windows(df, window_length):
    return df.rolling(window_length)


def prepare_data(df, horizon):
    # need to restrict the timestamps to intraday values
    # we would prefer to not have to worry about the after values prices as well
    if df.empty:
        return None
    data = df
    # data = feature_extraction(df) # .replace([np.inf, -np.inf], np.nan).dropna()
    # data = df
    pred = compute_prediction_int(data, n=horizon)
    reg = compute_prediction_reg(data, n=horizon)
    price = df['Close']
    dates = df.index

    data = normalize(data)
    next_data = data.copy()
    data = data.iloc[:-horizon]
    pred = pred[:-horizon]
    reg = reg[:-horizon]
    if data.empty:
        return None
    # data = quantize(data)
    # next_data = quantize(next_data)
    data['pred'] = pred
    data['reg'] = reg
    data['price'] = price
    # data['date'] = dates
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(), next_data[-horizon:]


def plot_histograms(corr, conf, bins=30, name=None, norm_hist=False):
    plt.figure(figsize=(10, 10))
    right = []
    wrong = []
    for j in range(len(conf)):
        _ = conf[j]
        x = corr[j]
        if x:
            right.append(_)
        else:
            wrong.append(_)

    sns.distplot(right, kde=True, bins=bins, norm_hist=norm_hist, label='Correct')
    sns.distplot(wrong, kde=True, bins=bins, norm_hist=norm_hist, label='Incorrect')
    if name:
        plt.xlabel(name)
    else:
        plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    for i in range(len(targets)):
        target = targets[i]
        if target == 0:
            predictions[i] = 1 - predictions[i]
    predictions = np.array(predictions)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    ce = np.log(predictions + 1e-9)
    return ce


def get_x_y(df_batch, timeseries):
    X = []
    Y = []
    Class = []
    for i in range(len(df_batch) - timeseries):
        x = df_batch.iloc[i:i + timeseries]
        y = x['reg'].iloc[-1]
        price = x['price'].iloc[-1]
        date = x.iloc[-1].index
        _class = x['pred'].iloc[-1]
        x = x.drop(['reg', 'pred', 'price'], axis=1)
        Class.append(_class)
        X.append(np.array(x))
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    Class = np.array(Class)
    return X, Y, Class


def get_option_price(price, strike, time):
    S = price  # underlying asset price
    K = strike  # strike
    q = .05  # annualized dividend
    t = time  # time to expiration in years
    r = 0.01  # risk-free interest rate
    sigma = 3  # IV
    p_calc = black_scholes_merton('p', S, K, t, r, sigma, q)
    c_calc = black_scholes_merton('c', S, K, t, r, sigma, q)
    if c_calc < .01:
        c_calc = .01
    if p_calc < .01:
        p_calc = .01

    return {'call': c_calc, 'put': p_calc}


def ceil(x, base):
    return base * math.ceil(x / base)


def floor(x, base):
    return base * math.floor(x / base)


def get_chart(df, index, batch_size, next_values):
    prev_values = [df.shift(-i - 1).loc[index].price for i in range(30)]
    next_values = [df.shift(i).loc[index].price for i in range(batch_size)]
    prev_values += [None for _ in next_values]
    to_add = len(prev_values) - len(next_values)
    next_values = [None for i in range(to_add)] + next_values
    x = range(len(next_values))
    plt.plot(x, prev_values, "o")
    plt.plot(x, next_values, "x")
    plt.show()


def bull_put_spread(price, upper, lower, time):
    upper_ops = get_option_price(price, upper, time)
    lower_ops = get_option_price(price, lower, time)
    cost = lower_ops['put'] - upper_ops['put']
    if abs(cost) < .01:
        try:
            cost = .01 * cost / abs(cost)
        except:
            cost = .01
    return cost


def bear_call_spread(price, upper, lower, time):
    upper_ops = get_option_price(price, upper, time)
    lower_ops = get_option_price(price, lower, time)
    cost = upper_ops['call'] - lower_ops['call']
    if abs(cost) < .01:
        try:
            cost = .01 * cost / abs(cost)
        except:
            cost = .01
    return cost


def index_to_dt(index):
    """ '2019-04-18' """
    segments = index.split('-')
    year = int(segments[0])
    month = int(segments[1])
    day = int(segments[2])
    dt = datetime.datetime(year=year, month=month, day=day)
    return dt


def df_to_weekday(df):
    """ '2019-04-18' """
    indicies = df.index
    segments = [index.split('-') for index in indicies]
    dt = [datetime.datetime(year=int(seg[0]), month=int(seg[1]), day=int(seg[2])) for seg in segments]
    return [_.weekday() for _ in dt]


def bull_put_spread(price, upper, lower, time):
    upper_ops = get_option_price(price,upper,time)
    lower_ops = get_option_price(price,lower,time)
    cost = lower_ops['put'] - upper_ops['put']
    if abs(cost)< .01:
        try:
            cost = .01 * cost/abs(cost)
        except:
            cost = .01
    return cost


def bear_call_spread(price, upper, lower, time):
    upper_ops = get_option_price(price,upper,time)
    lower_ops = get_option_price(price,lower,time)
    cost = upper_ops['call'] - lower_ops['call']
    if abs(cost)< .01:
        try:
            cost = .01 * cost/abs(cost)
        except:
            cost = .01
    return cost


def bear_put_spread(price, upper, lower, time):
    upper_ops = get_option_price(price,upper,time)
    lower_ops = get_option_price(price,lower,time)
    cost = - lower_ops['put'] + upper_ops['put']
    if abs(cost)< .01:
        try:
            cost = .01 * cost/abs(cost)
        except:
            cost = .01
    return cost


def bull_call_spread(price, upper, lower, time):
    upper_ops = get_option_price(price,upper,time)
    lower_ops = get_option_price(price,lower,time)
    cost = - upper_ops['call'] + lower_ops['call']
    if abs(cost)< .01:
        try:
            cost = .01 * cost/abs(cost)
        except:
            cost = .01
    return cost


def index_to_dt(index):
    """ '2019-04-18' """
    segments = index.split('-')
    year = int(segments[0])
    month = int(segments[1])
    day = int(segments[2])
    dt = datetime.datetime(year=year,month=month,day=day)
    return dt


def is_day_of_week(dt, day_of_week):
    dow = ['monday','tuesday','wednesday','thursday','friday'].index(day_of_week)
    return dt.weekday() == dow


def relative_strength_index(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= len(df) - 1:
        UpMove = df.iloc[i + 1]['High'] - df.iloc[i]['High']
        DoMove = df.iloc[i]['Low'] - df.iloc[i + 1]['Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI))
    df = df['RSI_' + str(n)] = RSI
    return df


def macd(df, n_fast, n_slow):
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df
