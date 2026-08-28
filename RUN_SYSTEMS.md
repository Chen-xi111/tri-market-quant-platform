# 三市场统一平台启动说明

推荐从项目根目录使用一个命令启动统一入口和三个隔离的市场服务：

```bash
./run_all.sh
```

浏览器入口为 `http://localhost:8500`。按一次 `Control-C` 会同时安全停止四个
Streamlit 进程。运行日志写入 `.runtime/logs/`，不会进入 Git 仓库。

三套应用现在使用独立源码目录、行情缓存、Streamlit 进程与端口：

- A 股：`systems/cn`，默认 `http://localhost:8503`
- 美股：`systems/us`，默认 `http://localhost:8501`
- 港股：`systems/hk`，默认 `http://localhost:8502`

A 股应用顶部包含两个互相隔离的专区：原“个股策略工作台”和新增“全市场资金雷达”。
资金雷达在服务运行期间于每个工作日 15:20 自动扫描，也可以手动执行：

```bash
cd "/Users/mi_qi88sohu.com/Documents/宝钢股份/systems/cn"
../../.venv/bin/python run_fund_flow_scan.py
```

## 分别启动

需要单独调试某个市场时，可以在不同 Terminal 中分别运行：

```bash
cd "/Users/mi_qi88sohu.com/Documents/宝钢股份"
./run_cn.sh
```

```bash
cd "/Users/mi_qi88sohu.com/Documents/宝钢股份"
./run_us.sh
```

```bash
cd "/Users/mi_qi88sohu.com/Documents/宝钢股份"
./run_hk.sh
```

用 `Control-C` 只会停止当前 Terminal 对应的系统，不会关闭另外两套。

如果默认端口已被其他程序占用，可以把新端口作为第一个参数传入：

```bash
./run_cn.sh 8513
./run_us.sh 8511
./run_hk.sh 8512
```

原来的 `./run.sh` 保留为美股启动命令的兼容别名。以后不要直接在项目根目录运行
`streamlit run dashboard.py`，否则会绕开这套隔离目录。

三套程序共用项目根目录的 `.venv` Python 环境，但源码、市场配置、交易会话、
模拟账户、行情缓存和界面状态互相独立。
