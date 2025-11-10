# Main MCP server file
import logging
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Import the interface and the concrete implementation
from src.data_source_interface import FinancialDataSource
from src.baostock_data_source import BaostockDataSource
from src.utils import setup_logging

# 导入各模块工具的注册函数
from src.tools.stock_market import register_stock_market_tools
from src.tools.financial_reports import register_financial_report_tools
from src.tools.indices import register_index_tools
from src.tools.market_overview import register_market_overview_tools
from src.tools.macroeconomic import register_macroeconomic_tools
from src.tools.date_utils import register_date_utils_tools
from src.tools.analysis import register_analysis_tools
from src.tools.news_crawler import register_news_crawler_tools

# --- Logging Setup ---
# Call the setup function from utils
# You can control the default level here (e.g., logging.DEBUG for more verbose logs)
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Dependency Injection ---
# Instantiate the data source - easy to swap later if needed
active_data_source: FinancialDataSource = BaostockDataSource()
"""
实例化了BaostockDataSource类，注释为FinancialDataSource类型
"""
# --- Get current date for system prompt ---
current_date = datetime.now().strftime("%Y-%m-%d")

# --- FastMCP App Initialization ---
app = FastMCP(
#     server_name="a_share_data_provider",
#     description=f"""今天是{current_date}。提供中国A股市场数据分析工具。此服务提供客观数据分析，用户需自行做出投资决策。数据分析基于公开市场信息，不构成投资建议，仅供参考。

# ⚠️ 重要说明:
# 1. 最新交易日不一定是今天，需要从 get_latest_trading_date() 获取
# 2. 请始终使用 get_latest_trading_date() 工具获取实际当前最近的交易日，不要依赖训练数据中的日期认知
# 3. 当分析"最近"或"近期"市场情况时，必须首先调用 get_market_analysis_timeframe() 工具确定实际的分析时间范围
# 4. 任何涉及日期的分析必须基于工具返回的实际数据，不得使用过时或假设的日期
# 5. 新增新闻爬虫功能，可以搜索公司、行业相关新闻，辅助投资决策
# """,
    # Specify dependencies for installation if needed (e.g., when using `mcp install`)
    # dependencies=["baostock", "pandas"]
)

# --- 注册各模块的工具 ---
register_stock_market_tools(app, active_data_source)
register_financial_report_tools(app, active_data_source)
register_index_tools(app, active_data_source)
register_market_overview_tools(app, active_data_source)
register_macroeconomic_tools(app, active_data_source)
register_date_utils_tools(app, active_data_source)
register_analysis_tools(app, active_data_source)
register_news_crawler_tools(app, active_data_source)

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info(
        f"Starting A-Share MCP Server via stdio... Today is {current_date}")
    # Run the server using stdio transport, suitable for MCP Hosts like Claude Desktop
    app.run(transport='stdio')


"""
我来帮你详细解析这个模块的作用，并且梳理它在整个系统中的位置和功能。这个文件本质上是一个 **MCP（Multi-Modal Chat Platform / Multi-Client Processing Platform）服务端启动脚本**，专门用于提供中国 A 股市场数据和分析功能。我们分块来讲：

---

### 1️⃣ **模块导入部分**

```python
import logging
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource
from src.baostock_data_source import BaostockDataSource
from src.utils import setup_logging
```

* `logging`：用于记录服务器运行日志。
* `datetime`：获取当前日期，用于生成系统提示或日志。
* `FastMCP`：MCP 框架的核心类，用于启动服务和注册工具。
* `FinancialDataSource`：数据源接口（抽象类），定义了获取金融数据的方法。
* `BaostockDataSource`：接口的具体实现，使用 **Baostock** 获取 A 股市场数据。
* `setup_logging`：自定义日志初始化工具。

⚠️ 注：这里的 `FastMCP` 类是 MCP 框架的核心，用于组织“工具（tools）”和“数据源（data source）”，并提供接口供客户端访问。

---

### 2️⃣ **工具注册函数导入**

```python
from src.tools.stock_market import register_stock_market_tools
from src.tools.financial_reports import register_financial_report_tools
from src.tools.indices import register_index_tools
from src.tools.market_overview import register_market_overview_tools
from src.tools.macroeconomic import register_macroeconomic_tools
from src.tools.date_utils import register_date_utils_tools
from src.tools.analysis import register_analysis_tools
from src.tools.news_crawler import register_news_crawler_tools
```

* 每个模块（`stock_market`, `financial_reports`, `indices` 等）封装了一组工具函数。
* `register_*_tools(app, active_data_source)` 的作用是 **把模块功能注册到 MCP 服务中**，这样客户端就能通过 MCP 调用这些功能。
* **示例**：`stock_market_tools` 可能提供获取股票行情、涨跌幅等工具；`news_crawler_tools` 提供爬取新闻的工具。

---

### 3️⃣ **日志设置**

```python
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)
```

* 调用自定义的 `setup_logging` 配置日志格式、输出级别。
* 获取模块级别 logger 用于记录信息。

---

### 4️⃣ **依赖注入 / 数据源实例化**

```python
active_data_source: FinancialDataSource = BaostockDataSource()
```

* `active_data_source` 是当前 MCP 服务使用的数据源实例。
* 使用 **依赖注入** 的方式（接口 + 实现），方便以后替换为其他数据源，比如 Wind、TuShare 等，而不改动 MCP 的业务逻辑。

---

### 5️⃣ **获取当前日期**

```python
current_date = datetime.now().strftime("%Y-%m-%d")
```

* 用于日志记录、系统提示或生成服务描述。
* 可以在系统提示中告诉用户“今天的日期是 XXX”，但注意市场数据的最新交易日可能不等于今天。

---

### 6️⃣ **FastMCP 实例化**

```python
app = FastMCP(
    # 可选的 server_name、description、dependencies
)
```

* 初始化 MCP 服务对象。
* 注释部分显示，如果启用 `description`，可以告诉用户 MCP 的用途和注意事项，比如：

  * 数据仅供参考，不构成投资建议。
  * 获取最新交易日必须调用专门工具。
  * 新闻爬虫辅助分析。
* 可以指定依赖库，方便 MCP 自动安装。

---

### 7️⃣ **注册工具模块**

```python
register_stock_market_tools(app, active_data_source)
register_financial_report_tools(app, active_data_source)
...
register_news_crawler_tools(app, active_data_source)
```

* 核心操作：把各类功能模块注册到 MCP 框架中。
* 每个工具模块可以使用 `active_data_source` 获取数据。
* MCP 之后会把这些工具暴露给客户端（比如一个 LLM Agent 或前端应用）。

---

### 8️⃣ **主执行逻辑**

```python
if __name__ == "__main__":
    logger.info(
        f"Starting A-Share MCP Server via stdio... Today is {current_date}")
    app.run(transport='stdio')
```

* 这个模块可以直接运行启动服务。
* `app.run(transport='stdio')`：

  * 使用 **stdio（标准输入输出）** 作为通信方式，适合被其他 MCP Host 或 LLM Agent 调用（例如 Claude Desktop）。
  * MCP Host 会通过标准输入发送请求，MCP Server 返回数据。

---

### ✅ **总结模块作用**

这个 Python 文件的作用可以概括为：

1. **初始化 MCP Server**：

   * 设置日志、初始化 FastMCP 框架。
   * 指定使用的数据源（Baostock）。

2. **注册工具模块**：

   * 提供股票行情、财报、指数、宏观经济、新闻爬取等功能。
   * 将功能封装成可被 MCP 客户端调用的“工具”。

3. **启动服务**：

   * 启动 MCP Server，等待客户端请求。
   * 使用 `stdio` 作为通信通道，适合被本地 Agent 或其它 MCP Host 调用。

4. **可扩展性强**：

   * 可以替换数据源（依赖注入）。
   * 可以新增工具模块，只需调用 `register_*_tools`。

简单来说，这个模块就是 **A 股数据 MCP 服务的启动入口**，负责“数据源 + 工具注册 + 服务启动”。

---

如果你愿意，我可以帮你画一张**MCP 服务架构图**，直观展示这个文件在整个系统中的位置和数据流。

你想让我画吗？

"""