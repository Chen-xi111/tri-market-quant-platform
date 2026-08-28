# 富途 OpenD 低延迟行情接入

网站只通过本机 `127.0.0.1:11111` 读取行情，不会收集或保存富途密码。

1. 安装并启动富途官方 OpenD。
2. 在 OpenD 中登录富途账户。
3. 确认 API 监听地址为 `127.0.0.1:11111`。
4. 在网站侧边栏“行情 API”选择“富途 OpenD”。

终端诊断：

```bash
cd "/Users/mi_qi88sohu.com/Documents/宝钢股份/systems/us"
../../.venv/bin/python diagnose.py AAPL
```

美股报价实时性取决于富途账户实际拥有的行情权限。网站会显示 OpenD 返回的
`update_time` 和数据年龄；若 OpenD 中断，会自动回退东方财富和 Yahoo。

刷新建议：

- 富途 OpenD、IBKR 或东方财富单股快照：5 秒。
- Yahoo：15 秒或 30 秒；高频重试不能消除其源端延迟。
- 自动刷新只重跑实时盯盘片段，不会反复执行回测。
