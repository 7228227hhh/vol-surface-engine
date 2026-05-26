"""
直接从币安公开API获取期权数据（无需API key）
"""


import pandas as pd
import numpy as np
from datetime import datetime
import json
import requests
from curl_cffi import requests as rq# 一个支持 TLS/JA3 指纹模拟的异步 HTTP 客户端。简单说，能让你的 Python 请求伪装成真实的浏览器
import websockets
import threading
import asyncio
import time
import sys
from websockets_proxy import  Proxy,proxy_connect
# 1. 获取期权标记价格（含IV和Greeks）
mark_url = "https://eapi.binance.com/eapi/v1/mark"#请注意，mark_url返回的所有字段都是理论值，币安综合了买卖双方以及做市商的报价后，利用bs模型从市场里面反推出来的greeks和iv，仅供参考，不代表实际成交）
#params = {"underlying": "BTCUSDT"}
info_url= "https://eapi.binance.com/eapi/v1/exchangeInfo"
info_resp=requests.get(info_url)
print(info_resp.json()['optionSymbols'][0].keys())
exchange_info=info_resp.json()
"""加一个查询币安支持的底层资产"""
unique_underlying=set()
for symbol in exchange_info['optionSymbols']:
    unique_underlying.add(symbol['underlying'])
print("币安支持的所有底层期权资产：")
for u in sorted(unique_underlying):
    print(f" - {u}")
static_data={}
for opt in exchange_info['optionSymbols']:
    static_data[opt['symbol']]={
        'strike': float(opt['strikePrice']),
        'expiryDate': opt['expiryDate'],  # 到期日
        'OptionType': opt['side'],
        'underlying': opt['underlying']
    }
print("正在获取币安期权数据...")
headers={ 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
mark_resp = requests.get(mark_url,headers=headers,timeout=10)
mark_data = mark_resp.json()

print(f"获取到 {len(mark_data)} 个期权合约：")
print("字典列表")
for key in mark_data[0].keys():
    print(f"{key}")
print('\n第一条数据实例：')
print(json.dumps(mark_data[0], indent=2, ensure_ascii=False))#indent表示分行且缩进，ensure_ascii参数控制中文字符能不能显示


# 2. 转换成DataFrame
records = []
now = datetime.now()

for item in mark_data:
    # 到期日转换
    symbol=item['symbol']
    if symbol not in static_data:
        continue
    static = static_data[symbol]
    expiry_ms=int(static['expiryDate'])
    expiry_date=datetime.fromtimestamp(expiry_ms/1000)
    days_left=(expiry_date - now).total_seconds()/86400#精确到天的小数
    if days_left < 1:
        continue



    records.append({
        'symbol': symbol,
        'strike': static['strike'],
        'expiry_date': expiry_date,
        'days_left': days_left,
        'T': days_left / 365.0,
        'implied_vol': float(item.get('markIV', 0)),  # 币安给的是markIV
        'delta': float(item.get('delta', 0)),
        'gamma': float(item.get('gamma', 0)),
        'vega': float(item.get('vega', 0)),
        'theta': float(item.get('theta', 0)),
        'option_type': static['OptionType'][0],  # 'C' or 'P'
        'mark_price': float(item.get('markPrice', 0))
    })

df = pd.DataFrame(records)
print(f"清洗后: {len(df)} 个合约")

# 3. 获取现货价格，我用一共尝试了三个方法，第一个是不模拟浏览器，直接用requests库拉取请求，被拒绝，第二个是用curl_cffi里面的requests库模拟浏览器拉取请求，仍然被拒绝，最后使用了websocket订阅的模式，成功。
"""def get_spot_from_symbol(symbol,quote='USDT'):
    PROXIES = {
        'http': 'http://127.0.0.1:1080',
        'https': 'http://127.0.0.1:1080'
    }
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        #'Accept': 'application/json, text/plain, */*',
    }
    underlying=symbol.split('-')[0]+quote
    spot_url = "https://api.binance.com/api/v3/ticker/price"
    spot_resp = requests.get(
        spot_url,
        params={"symbol": underlying},
        headers=HEADERS,   # 过墙
        proxies=PROXIES,
        #impersonate='chrome120',
        #timeout=10
    )
    return float(spot_resp.json()['price'])"""
BTC_current_spot_price = None
ETH_current_spot_price = None
BNB_current_spot_price = None
XRP_current_spot_price = None
DOGE_current_spot_price = None
SOL_current_spot_price = None
spot_price_ready = threading.Event()
loop = None


def start_websocket_thread():
    """在后台线程启动asyncio事件循环"""

    def run_loop():
        global loop
        #创建事件循环
        loop = asyncio.new_event_loop()
        #激活事件循环
        asyncio.set_event_loop(loop)
        #运行事件循环
        loop.run_until_complete(subscribe_spot_price())

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

"""try...except可以用来捕获异常，然后决定下一步动作，代码里没用到，但是可以添加的其实"""
async def subscribe_spot_price():
    """订阅实时价格"""
    uri = "wss://stream.binance.com:9443/ws"
    proxy = Proxy.from_url("http://127.0.0.1:1080")

    async with proxy_connect(uri,proxy=proxy) as ws:#async with和普通with用法类似，由于websockets.connect(uri)订阅是一个异步过程，所以为了方便不用手动关闭，就用自动管理
        # 订阅成交数据
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": ["btcusdt@trade","ethusdt@trade","bnbusdt@trade","solusdt@trade","xrpusdt@trade","dogeusdt@trade"],
            "id": 1
        }
        await ws.send(json.dumps(subscribe_msg))
        print("WebSocket: 已订阅合约实时价格")

        # 接收消息
        async for message in ws:#这里因为ws是异步过程返回的是异步可迭代对象（async iterable），所以不能用普通for，需要用async for
            data = json.loads(message)

            if 'p' in data:
                global BTC_current_spot_price,ETH_current_spot_price,BNB_current_spot_price,XRP_current_spot_price,DOGE_current_spot_price,SOL_current_spot_price
                if data['s']=='BTCUSDT':
                    BTC_current_spot_price = float(data['p'])
                elif data['s']=='ETHUSDT':
                    ETH_current_spot_price = float(data['p'])
                elif data['s']=='BNBUSDT':
                    BNB_current_spot_price = float(data['p'])
                elif data['s']=='XRPUSDT':
                    XRP_current_spot_price = float(data['p'])
                elif data['s']=='DOGEUSDT':
                    DOGE_current_spot_price = float(data['p'])
                elif data['s']=='SOLUSDT':
                    SOL_current_spot_price = float(data['p'])
                if (BTC_current_spot_price is not None and
                        ETH_current_spot_price is not None and
                        BNB_current_spot_price is not None and
                        XRP_current_spot_price is not None and
                        DOGE_current_spot_price is not None and
                        SOL_current_spot_price is not None):
                    if not spot_price_ready.is_set():
                        spot_price_ready.set()


def get_spot_price(symbol,timeout=5):

    """获取最新现货价格"""
    global BTC_current_spot_price,ETH_current_spot_price,BNB_current_spot_price,XRP_current_spot_price,DOGE_current_spot_price,SOL_current_spot_price

    if not symbol.endswith('USDT'):
        symbol = f"{symbol.upper()}USDT"
    else:
        symbol = symbol.upper()

        # 启动 WebSocket（只启动一次）
    if not spot_price_ready.is_set():
        start_websocket_thread()
        print("等待WebSocket连接...")
        spot_price_ready.wait(timeout)

        # 根据 symbol 返回对应价格
    if symbol[:3] == 'BTC':
        price = BTC_current_spot_price
    elif symbol[:3] == 'ETH':
        price = ETH_current_spot_price
    elif symbol[:3] == 'BNB':
        price = BNB_current_spot_price
    elif symbol[:3] == 'XRP':
        price = XRP_current_spot_price
    elif symbol[:3] == 'DOG':
        price = DOGE_current_spot_price
    elif symbol[:3] == 'SOL':
        price = SOL_current_spot_price
    else:
        print(f"不支持的交易对: {symbol}")
        return None

    if price is not None:
        print(f"获取到 {symbol} 价格: {price}")
        return price
    else:
        wait_count = 0
        while price is None and wait_count < 50:
            time.sleep(1)
            wait_count += 1
            # 重新获取 price
            if symbol[:3] == 'BTC':
                price = BTC_current_spot_price
            elif symbol[:3] == 'ETH':
                price = ETH_current_spot_price
            elif symbol[:3] == 'BNB':
                price = BNB_current_spot_price
            elif symbol[:3] == 'XRP':
                price = XRP_current_spot_price
            elif symbol[:3] == 'DOG':
                price = DOGE_current_spot_price
            elif symbol[:3] == 'SOL':
                price = SOL_current_spot_price

        if price is not None:
            print(f"获取到 {symbol} 价格: {price}")
            return price
        else:
            print(f"错误: 等待50秒后仍未获取到 {symbol} 价格")
            sys.exit(1)  # 程序退出
# 4. 计算log_moneyness
df['spot'] = df['symbol'].apply(get_spot_price)
df['log_moneyness'] = np.log(df['strike'] / df['spot'])
# 5. 查看数据
print("\n数据预览:")
print(df.head(10))

print("\n按到期日统计:")
print(df.groupby('expiry_date').agg({
    'symbol': 'count',
    'implied_vol': ['min', 'mean', 'max']
}))
df.to_csv('data/data.csv', index=False)





