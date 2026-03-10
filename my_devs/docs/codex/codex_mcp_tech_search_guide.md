# Codex MCP 简明说明

这台机器现在只装了两个 MCP：

- `Context7`：查官方技术文档
- `Playwright`：打开网页、读取网页内容

这份文档只回答 4 个问题：

1. 现在装了什么
2. 怎么装
3. 怎么用
4. 出问题怎么查

更新时间：`2026-03-09`

---

## 1. 现在装了什么

执行：

```bash
codex mcp list
```

当前结果应类似：

```text
Name        Command  Args                                                 Env  Cwd  Status   Auth
context7    npx      -y @upstash/context7-mcp                             -    -    enabled  Unsupported
playwright  docker   run -i --rm --init mcr.microsoft.com/playwright/mcp  -    -    enabled  Unsupported
```

作用非常简单：

- `Context7`：你要我先查官方文档时用它
- `Playwright`：你给我网页链接，希望我直接进去看时用它

---

## 2. 它们分别怎么用

### 2.1 `Context7` 是干什么的

适合：

- 查 API 文档
- 查框架文档
- 查库的推荐用法
- 查版本相关说明

你可以直接这样说：

```text
用 Context7 查官方文档，再回答我。不要凭记忆，给我来源链接。
```

常见例子：

```text
用 Context7 查 FastAPI 官方文档，告诉我 lifespan 推荐怎么写。
```

```text
用 Context7 查 TensorRT 官方 API 文档，告诉我 Python 推理流程怎么写；查不到就直接说查不到。
```

### 2.2 `Playwright` 是干什么的

适合：

- 打开网页
- 读取标题、正文、按钮文字
- 关闭弹窗
- 点击页面元素
- 查看网页实际可见内容

你可以直接这样说：

```text
用 Playwright 打开这个网页，帮我读内容。
```

常见例子：

```text
用 Playwright 打开这个网页，告诉我页面标题、主要内容和关键信息。
```

```text
用 Playwright 打开这个 Microsoft Store 页面，告诉我应用名、开发者、分类、是否免费。
```

### 2.3 两个一起用

如果你既要查文档，又要看网页，可以这样说：

```text
先用 Context7 查官方文档，再用 Playwright 打开这个网页核对，最后给我结论。
```

---

## 3. 我是怎么装的

### 3.1 安装 `Context7`

执行：

```bash
codex mcp add context7 -- npx -y @upstash/context7-mcp
```

为了避免第一次启动太慢，我补了超时配置：

```toml
[mcp_servers.context7]
command = "npx"
args = ["-y", "@upstash/context7-mcp"]
startup_timeout_sec = 60
```

### 3.2 安装 `Playwright`

我最后装的是 **Docker 版**，不是 `npx` 版。

原因：这台机器没有本地 `Chrome/Chromium`，`npx` 版虽然能启动，但实际开网页时浏览器起不来。

先拉镜像：

```bash
docker pull mcr.microsoft.com/playwright/mcp
```

再添加 MCP：

```bash
codex mcp add playwright -- docker run -i --rm --init mcr.microsoft.com/playwright/mcp
```

对应配置是：

```toml
[mcp_servers.playwright]
command = "docker"
args = ["run", "-i", "--rm", "--init", "mcr.microsoft.com/playwright/mcp"]
```

---

## 4. 你以后怎么用

最简单的方式就是先进入：

```bash
codex
```

然后按下面这个规则说话：

- 查文档：明确说 `Context7`
- 看网页：明确说 `Playwright`
- 查不到：要求我直接说查不到，不要猜

我最推荐你以后固定使用这几句：

### 4.1 查文档模板

```text
先用 Context7 查官方文档，再回答我。不要凭记忆，给我来源链接。
```

### 4.2 看网页模板

```text
用 Playwright 打开这个网页，先处理弹窗，再提取页面标题、主要内容和关键信息。读不到就直接说读不到，不要猜。
```

### 4.3 严格模式模板

```text
先查再答。文档走 Context7，网页走 Playwright。查不到的地方直接写“未查到/不确定”，不要猜。
```

---

## 5. 我已经实际验证过什么

### 5.1 `Context7`

已经实测过：

- 能被 `Codex` 正常调用
- 能查官方技术文档
- 能返回文档链接和 API 说明

### 5.2 `Playwright`

已经实测过：

- 能打开 `https://example.com/`
- 能读到页面标题和 H1
- 能打开 Microsoft Store 页面
- 能处理弹窗并继续读取页面内容

也就是说，这两个都不是“只配上了”，而是“已经用过了”。

---

## 6. 常用命令

查看当前 MCP 列表：

```bash
codex mcp list
```

查看 `Context7` 配置：

```bash
codex mcp get context7
```

查看 `Playwright` 配置：

```bash
codex mcp get playwright
```

删除 `Context7`：

```bash
codex mcp remove context7
```

删除 `Playwright`：

```bash
codex mcp remove playwright
```

---

## 7. 如果以后要重装

重装 `Context7`：

```bash
codex mcp add context7 -- npx -y @upstash/context7-mcp
```

重装 `Playwright`：

```bash
docker pull mcr.microsoft.com/playwright/mcp
codex mcp add playwright -- docker run -i --rm --init mcr.microsoft.com/playwright/mcp
```

---

## 8. 常见问题

### 8.1 为什么 `Context7` 要加超时

因为第一次用 `npx` 启动可能比较慢，所以我加了：

```toml
startup_timeout_sec = 60
```

### 8.2 为什么 `Playwright` 用 Docker 版

因为这台机器没有本地浏览器内核，Docker 版更稳，开网页更直接。

### 8.3 如果 `Playwright` 打不开网页怎么办

先检查 Docker：

```bash
docker --version
```

再拉一次镜像：

```bash
docker pull mcr.microsoft.com/playwright/mcp
```

再检查 MCP 是否还在：

```bash
codex mcp list
```

### 8.4 如果我不写 `Context7` 或 `Playwright` 行不行

可以，但不如明确说稳定。

最简单的原则就是：

- 要文档，点名 `Context7`
- 要网页，点名 `Playwright`

---

## 9. 最后一段总结

你现在不用记很多东西，只要记住下面这几句：

- 查官方技术文档：`Context7`
- 直接打开网页去看：`Playwright`
- 这两个现在都已经装好了
- 我已经实际用过它们，不是纸面配置

你以后直接这样说就够了：

```text
查文档就用 Context7，看网页就用 Playwright。查不到就直接告诉我查不到，不要猜。
```

---

## 10. 相关文件

当前文档：

```text
my_devs/docs/codex/codex_mcp_tech_search_guide.md
```

当前全局配置：

```text
~/.codex/config.toml
```

