# IBKR Paper 接入

平台默认只接受 IBKR 纸账户端口：TWS Paper `7497` 或 IB Gateway Paper
`4002`。真实账户端口 `7496` / `4001` 会被程序拒绝。

## 1. 安装官方 TWS API

IBKR 不支持从 PyPI 获取 `ibapi`。请从 [IBKR 官方 TWS API 下载页](https://interactivebrokers.github.io/)
下载 Mac/Unix 版本并接受其许可条款。解压后，在本项目虚拟环境中安装：

```bash
cd ~/IBJts/source/pythonclient
"/Users/mi_qi88sohu.com/Documents/宝钢股份/.venv/bin/python" -m pip install .
```

确认安装：

```bash
"/Users/mi_qi88sohu.com/Documents/宝钢股份/.venv/bin/python" -m pip show ibapi
```

## 2. 登录纸账户并启用 API

1. 启动 Trader Workstation，选择 **Paper Trading** 登录；也可以登录 IB Gateway Paper。
2. 在 TWS 的 API / Settings 中启用 Socket 客户端。
3. TWS Paper Socket Port 使用 `7497`；IB Gateway Paper 使用 `4002`。
4. 保留 “localhost only” 或仅信任 `127.0.0.1`。
5. 若要在纸账户测试下单，关闭 Read-Only API；平台仍会拒绝真实账户端口。

登录凭据和双重认证只在 IBKR 软件中完成，不要填写到项目文件或 Streamlit 页面。

## 3. 诊断与连接

```bash
cd "/Users/mi_qi88sohu.com/Documents/宝钢股份"
source .venv/bin/activate
python3 diagnose.py AAPL
./run.sh
```

进入“模拟交易与风控”，在侧边栏选择 **IBKR Paper**，再点击“连接 IBKR Paper”。

IBKR API 行情授权与 TWS 屏幕行情可能不同。没有相应订阅时，程序会申请延迟行情；
网站的策略历史数据仍默认使用 Yahoo Finance，提交订单前应以 IBKR 报价复核。
