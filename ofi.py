import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

h = pd.Timedelta(seconds=20) #Set 20 seconds as the given time interval. This can be set to any value, and will have no effect on the time complexity

m = 10 #count of levels
bid_px = [0] * m
ask_px = [0] * m
bid_sz = [0] * m
ask_sz = [0] * m
bid_order_flow, ask_order_flow = None, None
pre_bof, pre_aof = None, None #prefix sums of bid and ask order flows
n = 0 #count of the number of rows
pre_bid_sz, pre_ask_sz = None, None

def load_data(filepath):
    return pd.read_csv(filepath)

def compute_order_flows(df):
    global n, bid_order_flow, ask_order_flow, pre_bof, pre_aof, pre_bid_sz, pre_ask_sz
    '''Precomputes all order flows'''
    for i in range(m):
        bid_px[i] = df['bid_px_' + f"{i:02}"]
        ask_px[i] = df['ask_px_' + f"{i:02}"]
        bid_sz[i] = df['bid_sz_' + f"{i:02}"]
        ask_sz[i] = df['ask_sz_' + f"{i:02}"]
    n = len(bid_px[0])
    bid_order_flow, ask_order_flow = [[0] * n] * m, [[0] * n] * m
    pre_bof, pre_aof = [[0] * n] * m, [[0] * n] * m
    pre_bid_sz, pre_ask_sz = [[0] * (n+1)] * m,  [[0] * (n+1)] * m
    for i in range(m):
        for j in range(1, n):
            if bid_px[i][j] == bid_px[i][j-1]:
                bid_order_flow[i][j] = bid_sz[i][j] - bid_sz[i][j-1]
            elif bid_px[i][j] > bid_px[i][j-1]:
                bid_order_flow[i][j] = bid_sz[i][j]
            else:
                bid_order_flow[i][j] = -bid_sz[i][j]
            pre_bof[i][j] = pre_bof[i][j-1] + bid_order_flow[i][j]

            if ask_px[i][j] == ask_px[i][j-1]:
                ask_order_flow[i][j] = ask_sz[i][j] - ask_sz[i][j-1]
            elif ask_px[i][j] > ask_px[i][j-1]:
                ask_order_flow[i][j] = -ask_sz[i][j]
            else:
                ask_order_flow[i][j] = ask_sz[i][j]

            pre_aof[i][j] = pre_aof[i][j-1] + ask_order_flow[i][j]

            pre_bid_sz[i][j] = pre_bid_sz[i][j-1] + bid_sz[i][j-1]
            pre_ask_sz[i][j] = pre_ask_sz[i][j-1] + ask_sz[i][j-1]

def compute_best_level_ofi(df, h):
    '''Computes ofi of the best level of i for each interval, and adds/updates it to df, using h as the time interval'''

    df['ofi_best_level'] = df['symbol']
    df.at[0, 'ofi_best_level'] = None
    prev = 0
    for i in range(1, n):
        cur_time = pd.Timestamp(df['ts_event'][i], tz='UTC')
        while cur_time > h+pd.Timestamp(df['ts_event'][prev+1], tz='UTC'):
            prev += 1
        cur_best_level_ofi = pre_bof[0][i] - pre_bof[0][prev] - pre_aof[0][i] + pre_aof[0][prev]
        df.at[i, 'ofi_best_level'] = cur_best_level_ofi
    return df

def compute_deep_level_ofi(df, h):
    '''Computes ofi of using multiple levels of i for each interval, and adds/updates it to df, using h as the time interval'''
    prev = 0
    avg_order_book_depth = [0] * n
    for i in range(1, n):
        cur_time = pd.Timestamp(df['ts_event'][i], tz='UTC')
        while cur_time > h+pd.Timestamp(df['ts_event'][prev+1], tz='UTC'):
            prev += 1
        div = i - prev
        if cur_time <= h+pd.Timestamp(df['ts_event'][prev], tz='UTC'):
            div += 1
        div *= 2
        for j in range(0, m):
            num = pre_bid_sz[j][i+1] - pre_bid_sz[j][prev+1] + pre_ask_sz[j][i+1] - pre_ask_sz[j][prev+1]
            avg_order_book_depth[i] += num
        avg_order_book_depth[i] /= m * div
    cur = 'multi_level_ofi'
    df[cur] = df['symbol']
    df.at[0, cur] = None
    prev = 0
    for i in range(1, n):
        cur_time = pd.Timestamp(df['ts_event'][i], tz='UTC')
        while cur_time > h+pd.Timestamp(df['ts_event'][prev+1], tz='UTC'):
            prev += 1
        cur_ofi = pre_bof[0][i] - pre_bof[0][prev] - pre_aof[0][i] + pre_aof[0][prev]
        df.at[i, cur] = cur_ofi / avg_order_book_depth[i]
    return df

def compute_integrated_ofi(df, h):
    '''Computes integrated ofi and adds/updates it to df, using h as the time interval. Assumes deep ofi is in df'''
    features = []
    for i in range(10):
        features += [
            f'bid_px_0{i}', f'ask_px_0{i}',
            f'bid_sz_0{i}', f'ask_sz_0{i}',
            f'bid_ct_0{i}', f'ask_ct_0{i}'
        ]
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    first_principal_vector = pca.components_[0]
    norm = 0
    for x in first_principal_vector:
        norm += abs(x)
    df['integrated_ofi'] = df['symbol']
    df.at[0, 'integrated_ofi'] = None
    for i in range(1, n):
        integrated_ofi = 0
        for j in range(m):
            integrated_ofi = first_principal_vector[j] * df.at[i, 'multi_level_ofi']
        df.at[i, 'integrated_ofi'] = integrated_ofi / norm
    return df

def save_results(df, output_filepath):
    df.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    df = load_data('first_25000_rows.csv')
    compute_order_flows(df)
    df = compute_best_level_ofi(df, h) #adds best level ofi
    df = compute_deep_level_ofi(df, h) #adds multi level ofi
    df = compute_integrated_ofi(df, h) #adds integrated ofi
    save_results(df, 'first_25000_rows_with_ofi.csv')
