coins = list(map(lambda u: u + "-USD", ["XRP", "BTC", "ETH", "SOL", "ZEC", "ADA", "LTC", "LINK"]))

import websocket
import json
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):

    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoding = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        self.decoding = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, input_size)
        )

    def forward(self, x):
        return self.decoding(self.encoding(x))
    

class DataFeed(threading.Thread):

    def __init__(self, depths, lr=0.001):
        threading.Thread.__init__(self)
        self.orderbook = {}
        self.sync = False

        self.autoencoder = {coin:{depth:AutoEncoder(depth*4) for depth in depths} for coin in coins}
        self.loss_function = {coin:{depth:nn.MSELoss() for depth in depths} for coin in coins}
        self.optimizer = {coin:{depth:optim.Adam(self.autoencoder[coin][depth].parameters(), lr=lr) for depth in depths} for coin in coins}
        self.anomaly = {coin:{depth:0 for depth in depths} for coin in coins}

    def run(self):
        conn = websocket.create_connection('wss://ws-feed.exchange.coinbase.com')
        msg = {'type':'subscribe','product_ids':coins,'channels':['level2_batch']}
        conn.send(json.dumps(msg))
        while True:
            resp = json.loads(conn.recv())
            if 'type' in resp.keys():
                if resp['type'] == 'snapshot':
                    self.orderbook[resp['product_id']] = {'bids':{float(price):float(volume) for price, volume in resp['bids']},
                                                          'asks':{float(price):float(volume) for price, volume in resp['asks']}}
                    if len(coins) == len(self.orderbook.keys()):
                        self.sync = True
                
                if resp['type'] == 'l2update':
                    ticker = resp['product_id']
                    for (side, price, volume) in resp['changes']:
                        price, volume = float(price), float(volume)
                        if side == 'buy':
                            if volume == 0:
                                if price in self.orderbook[ticker]['bids'].keys():
                                    del self.orderbook[ticker]['bids'][price]
                            else:
                                self.orderbook[ticker]['bids'][price] = volume
                        else:
                            if volume == 0:
                                if price in self.orderbook[ticker]['asks'].keys():
                                    del self.orderbook[ticker]['asks'][price]
                            else:
                                self.orderbook[ticker]['asks'][price] = volume


    def extract_book(self, ticker, depth=5):
        obook = self.orderbook[ticker]
        bid = list(sorted(obook['bids'].items(), reverse=True))[:depth]
        ask = list(sorted(obook['asks'].items()))[:depth]
        bid, ask = np.array(bid), np.array(ask)
        bp, bv = bid[:, 0], bid[:, 1]
        ap, av = ask[:, 0], ask[:, 1]
        mid = 0.5*(bp[0] + ap[0])
        bp = (bp - mid)/mid
        ap = (ap - mid)/mid
        bv = bv / sum(bv)
        av = av / sum(av)
        return bp.tolist() + ap.tolist() + bv.tolist() + av.tolist()
    
    def zero_out(self, depths):
        self.anomaly = {coin:{depth:0 for depth in depths} for coin in coins}

fig = plt.figure()
ax = fig.add_subplot(111)

depths = [5, 10, 15, 20, 25, 30, 35, 40]

feed = DataFeed(depths)
feed.start()

epochs = 40

Anomaly = []

while True:
    if feed.sync == True:
        for crypto in coins:
            temp = []
            for size in depths:
                inputs = feed.extract_book(crypto, depth=size)
                the_input = torch.tensor(inputs, dtype=torch.float32)
                feed.zero_out(depths)
                for epoch in range(epochs):
                    the_output = feed.autoencoder[crypto][size](the_input)
                    loss = feed.loss_function[crypto][size](the_output, the_input)
                    feed.optimizer[crypto][size].zero_grad()
                    loss.backward()
                    feed.optimizer[crypto][size].step()
                    feed.anomaly[crypto][size] += loss.item()
                temp.append(feed.anomaly[crypto][size])
            Anomaly.append(temp)
        
        cov = np.cov(Anomaly, rowvar=False)
        sd = np.sqrt(np.diag(cov)).reshape(-1, 1)
        sdv = sd @ sd.T
        corr = cov / sdv
        for i in range(len(corr)):
            corr[i][i] = np.nan

        ax.cla()
        ax.set_title('Crypto Orderbook Anomalies')
        ax.set_xticks(range(len(coins)))
        ax.set_xticklabels(coins)
        ax.set_yticks(range(len(coins)))
        ax.set_yticklabels(coins)
        mm, nn = corr.shape
        x, y = np.meshgrid(range(mm), range(nn))
        ax.contourf(x, y, corr, cmap='jet_r')
        plt.pause(0.001)
                
    else:
        print('Still Need Some Coins: ', len(coins) - len(feed.orderbook.keys()))


plt.show()
feed.join()