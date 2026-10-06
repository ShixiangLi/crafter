# Crafter 智能体算法研究

所有智能体统一使用 BALROG 的环境、观测、动作校验和评估指标，模型请求通过同一个 OpenAI 兼容客户端发送。外层仅负责实验配置与算法接入，不修改 `BALROG/` 源码。

## 运行

```bash
# 原生 BALROG naive：默认 qwen3:8b，10 回合，每回合最多 2000 步
./run.sh

# 短回合检查
./run.sh --episodes 1 --max-steps 3

# 同一评估流程下切换算法
./run.sh --agent react
./run.sh --agent planner
./run.sh --agent ours
./run.sh --agent spring

# 也可以显式选择算法配置
./run.sh --config configs/planner.yaml

# 查看生效配置，不调用模型、不生成实验结果
./run.sh --agent ours --dry-run
```

`run.sh` 使用项目 `.venv`，调用 `scripts/evaluate.py`。默认使用 Ollama，运行前确保服务已启动且模型已安装；也可通过配置连接远程 API。

`run.sh` 自动为本次进程及其子进程设置本机代理绕过（`127.0.0.1`、`localhost`、`::1`），合并并保留已有 `NO_PROXY` / `no_proxy` 规则。不会修改系统代理或影响其他正在运行的实验。直接执行 Python 入口时不经过此设置。

| 方法 | 实现 |
| --- | --- |
| `naive` | 直接使用 BALROG 原生 NaiveAgent |
| `react` | ReAct：保留观测、推理与动作历史，根据真实反馈调整决策 |
| `planner` | 外层基线：默认每 10 步更新目标计划，再选择动作 |
| `spring` | 官方 9 节点问答 DAG、游戏手册和最近两步观测 |
| `ours` | 新算法入口，目前与 planner 相同，尚未实现新方法 |

## 通用 API 配置

所有智能体固定读取同一个 `configs/default.yaml`，其中统一设置模型/API、温度、生成上限和评测预算。`configs/` 中其余 YAML 只选择智能体及算法参数，不能覆盖这些公共设置。默认接口配置：

```yaml
api:
  base_url: http://127.0.0.1:11434/v1
  model: qwen3:8b
  api_key_env: null
```

`base_url` 填完整的兼容 API 根地址；客户端不会自动补 `/v1`。无需认证时 `api_key_env: null`，需要认证时填写保存密钥的环境变量名。密钥只在请求初始化时读取，不写入配置快照或结果；指定的变量未设置或为空时，实验开始前报错。不要在 YAML 中写实际密钥或使用 `${oc.env:...}` 将密钥插入配置。

使用 DeepSeek 时，只需将 **`configs/default.yaml` 中的 `api` 段**改成下面的配置，所有智能体立即共用它（模型名按 [DeepSeek 官方文档](https://api-docs.deepseek.com/)及账户可用模型填写）：

```yaml
api:
  base_url: https://api.deepseek.com
  model: deepseek-flash
  api_key_env: DEEPSEEK_API_KEY
```

```bash
# 在当前终端安全输入密钥（不回显、不写入 shell 历史）
read -rsp 'DeepSeek API Key: ' DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY

# 同一模型下比较算法；先用相同种子进行短试验
./run.sh --agent naive --episodes 1 --max-steps 3 --seed 0
./run.sh --agent react --episodes 1 --max-steps 3 --seed 0
./run.sh --agent spring --episodes 1 --max-steps 3 --seed 0
```

远程 API 不使用 `--gpu`；混用时启动器会报错。`--model` 仍可覆盖模型，末尾 `client.base_url=...`、`client.model_id=...` 和 `client.api_key_env=...` 仍具有最高优先级。对比算法时保持这些覆盖参数一致；常规切换模型统一修改公共 `api` 配置。接口采用 Chat Completions，复用 BALROG 的生成、重试和 Token 统计；不额外实现各供应商的专有接口。

## 按 GPU 启动独立 Ollama

```bash
# 两个终端分别使用 GPU 0 和 GPU 1
./run.sh --gpu 0 --agent react
./run.sh --gpu 1 --agent spring

# 只检查配置，不启动服务或模型
./run.sh --gpu 1 --agent spring --dry-run
```

`--gpu N` 使用 `nvidia-smi` 的 GPU 编号，通过 GPU UUID 绑定独立 Ollama，端口为 `11500 + N`。首次自动启动，后续仅复用本脚本记录且绑定匹配的进程；端口被其他服务占用时会报错，不重启或停止原有服务。实现依据 [Ollama GPU 选择说明](https://docs.ollama.com/gpu)。

不传 `--gpu` 时保持原来的服务配置。`--gpu` 不能与 `client.base_url` 同时指定。独立服务优先使用显式 `OLLAMA_MODELS`；未指定时，按目标模型查找当前用户及常见系统目录中可读的已有模型，否则使用当前用户的 `~/.ollama/models`。不同 GPU 服务共用该目录，无需逐 GPU 下载。系统用户的目录若没有读取权限，不会修改权限或强行读取。

启动实验前会查询目标服务确认模型存在；缺失时立即报错并给出下载命令，不会创建失败实验再重试五次。例如 `OLLAMA_HOST=127.0.0.1:11501 ollama pull qwen3:8b`，只需准备一次，所有使用同一目录的服务即可访问。脚本不会自动下载模型。若显式切换目录而已有独立服务仍使用旧目录，脚本会提示在服务空闲时停止该 PID 后重跑，不会擅自重启服务。

服务在实验退出后保留，便于复用；日志、PID 和启动锁位于 `outputs/ollama/gpu_N.*`。需要停止时，确认对应实验已结束，再执行例如 `kill "$(cat outputs/ollama/gpu_1.pid)"`。已有服务若也在使用同一张 GPU，仍会共享算力；指定不同 GPU 的独立服务才能分开队列。同一个 `--gpu N` 的实验会共用同一服务。

## 配置与种子

`configs/default.yaml` 统一模型/API、推理设置和评测预算；其余配置仅允许 `agent`、`history_length`、`replan_interval`，ReAct 还可设置 `max_thoughts`。例如 `configs/planner.yaml` 可在 `agent: planner` 后配置重规划间隔。算法配置包含模型/API 或评测参数时会报错。

`--agent react` 自动加载 `configs/react.yaml`；`naive` 对应 `configs/balrog_baseline.yaml`，也是不传参数时的默认选择。`--config` 用于显式指定算法配置；同时传入 `--agent` 时两者必须一致。请通过这两个选项选择算法，不能通过 Hydra 的 `agent.type` 切换，以免混入另一算法的参数。

其余参数优先级为：末尾 BALROG Hydra 覆盖参数 > 命令行选项 > 算法配置 > 公共配置 > 上游默认配置。模型/API 和评测参数从公共配置读取，可通过命令行临时覆盖。

```bash
./run.sh --agent planner --model qwen3:8b --episodes 1 --max-steps 200 --seed 0
./run.sh --agent planner agent.replan_interval=5 agent.max_text_history=8
./run.sh --agent ours client.generate_kwargs.temperature=0.0 client.generate_kwargs.max_tokens=1024
```

`--seed` 同时设置 Crafter 构造种子与评估器种子；同一次运行中的所有回合会使用相同地图，不能把它们当作不同地图样本。比较方法时，对每个方法分别运行相同的一组种子，例如：

```bash
for agent in naive planner ours; do
  for seed in 0 1 2; do
    ./run.sh --agent "$agent" --episodes 1 --seed "$seed"
  done
done
```

不指定种子时沿用上游随机行为。环境种子不等于模型采样种子；当前 BALROG 兼容客户端没有转发模型采样 seed。使用 Hydra 手动设种子时，需同时设置 `envs.env_kwargs.seed` 和 `envs.crafter_kwargs.seed`。

公共配置统一所有方法的默认推理设置：温度 1.0、生成上限 8192、纯文本输入、单工作进程。其他方法默认使用 16 条历史，SPRING 固定使用最近两步完整观测。比较方法时请保持模型与预算一致，并报告实际 token 消耗。

每次启动都会创建新的结果目录，当前不支持断点续跑。

## 代码结构

```text
run.sh                       启动入口与 GPU 服务管理
configs/default.yaml         所有算法共用的模型/API、推理和评测设置
configs/*.yaml               各算法选择与参数
scripts/evaluate.py          执行 BALROG 评估并汇总结果
scripts/ollama_models.py     GPU 服务的模型目录选择与预检
scripts/analyze.py           为已有试验生成或重新生成分析报告
utils/recording.py           外层观测记录，保留原生评估流程
utils/statistics.py          结果读取、成就成功率和 Crafter score
utils/visualization.py       回合图、试验汇总图和 HTML 报告
agents/factory.py           原生 NaiveAgent 与自定义智能体的统一接入
agents/*_agent.py           各方法的接入入口，导出 modules 中的实现
modules/common/config.py    两个脚本共用的配置加载和校验
modules/common/api_client.py 兼容 API 认证，复用 BALROG 请求逻辑
modules/common/base_agent.py JSON 决策、提示词缓存和 LLMResponse 适配
modules/react/policy.py     ReAct 历史、独立思考与动作选择
modules/spring/policy.py    SPRING 观测记忆、问答调用与成本记录
modules/spring/graph.py     SPRING 问题 DAG 与动作匹配
modules/planner/            policy.py、planner.py 与 executor.py
modules/ours/policy.py      新算法实现入口，目前继承 Planner
prompts/react/react.txt     ReAct 提示词
prompts/planner/            planner.txt 与 executor.txt
prompts/spring/             官方问题、手册、来源版本和许可
BALROG/                     上游环境、客户端、历史和评估实现
outputs/results/            实验结果
```

算法实现和专用组件放在 `modules/<方法>/`，跨方法共享的功能放在 `modules/common/`；依赖从接入层指向方法实现，再指向公共功能，方法实现不反向导入 `agents` 或 `scripts`。各方法的配置与提示词分别放在 `configs/<方法>.yaml` 和 `prompts/<方法>/`。

开发新算法时，主要修改 `modules/ours/policy.py`，按需在同目录添加组件；`agents/our_agent.py` 仅作为注册接入入口。实现 BALROG 接口：

- `reset()`：清理每回合状态。
- `act(observation, prev_action=None)`：接收 BALROG 原始观测，返回 `LLMResponse`。

`observation['text']` 包含 `long_term_context` 和 `short_term_context`。不要直接读取环境隐藏状态。动作后的可见反馈在下一次 `act()` 中收到；上游没有调用外层 `observe()` 的接口。

自定义基类提供 `begin_step()`、`ask()`、`finish_step()`。一次行动中的规划和执行调用都会计入返回响应的 token 总数。动作字符串由 BALROG 校验；解析失败会作为无效候选交给 BALROG，执行其默认 Noop。计划、动作理由和解析错误写入原生 CSV 的 `Reasoning` 列。

## 结果

```text
outputs/results/<运行编号>/
├── config.yaml                  完整生效配置
├── eval.log                     运行日志
├── status.json                  预期、完成、缺失回合数和顶层错误
├── summary.json                 BALROG 总汇总
├── traces/crafter/default/       每回合状态与模型调用的 JSONL 记录
├── analysis/
│   ├── report.html               可在浏览器打开的完整分析报告
│   ├── overview.png              试验汇总图
│   ├── analysis_summary.json     成就成功率、Crafter score 和完成情况
│   └── episodes/crafter/default/ 每回合六面板分析图
└── crafter/
    ├── crafter_summary.json      进度、标准误、步数与 token
    └── default/
        ├── default_run_00.json   每回合成就、种子、参数与成本
        └── default_run_00.csv    动作、理由、观测、奖励和结束标志
```

BALROG 的 `progression_percentage` 为平均解锁成就数 / 22 × 100；它不是 Crafter score。

缺失回合不会自动进入 BALROG 成绩均值；运行不完整时启动器返回非零状态，并记录 `status.json`。分析结果时同时检查回合完整性。CSV 的 Action 是模型提交的候选动作；非法候选实际执行 Noop，回合 JSON 的 `failed_candidates` 记录非法候选。

试验正常结束或因异常结束后，自动生成 `analysis/report.html`。每回合图包含生存状态、成就里程碑、当前资源持有量、请求与执行动作、每次模型生成的 Token、生成耗时。状态从环境返回的 `info` 和可见观测中记录，包含第 0 步初始状态；不传给智能体。ReAct 独立思考和 SPRING 问答分别记录，因此环境步与模型调用的横轴不同。耗时包含客户端内部重试等待；Token 只包括成功返回的响应，内部失败尝试的用量可能无法取得。记录逐行落盘，异常回合保留已有数据；强制终止后可手动生成报告。

汇总只使用完整回合计算 22 项成就成功率和 [Crafter 官方分数](https://github.com/danijar/crafter/blob/main/analysis/common.py)：`exp(mean(log(1 + 成功率百分数))) - 1`。没有完整回合时分数标为不可用，部分回合单独展示；同一种子的重复回合不等于独立地图样本。图中的钻石成功指解锁 `collect_diamond`。这是当前评估回合的分数，与原论文的训练预算和统计范围需分别说明。

已有试验可重新分析，不调用模型：

```bash
.venv/bin/python -B scripts/analyze.py outputs/results/<运行编号>
```

历史结果没有保存的生命、库存、成就时间点和调用数据会标注为未记录。报告和 PNG 可离线查看；共享完整报告时保留 `analysis/` 目录中的图片。绘图使用 Matplotlib。分析或绘图失败不会覆盖原始结果，错误写入 `eval.log`。

## 检查配置

```bash
./run.sh --agent react --dry-run
./run.sh --config configs/planner.yaml --dry-run
```

`--dry-run` 只校验并打印生效配置，不调用模型、启动服务或生成结果，也不验证 API 连通性。实际连接可用前面的单回合、少步数命令检查；远程 API 会产生相应调用费用。

## SPRING

```bash
./run.sh --config configs/spring.yaml --episodes 1 --max-steps 3 --seed 0
```

每个环境步执行 9 次问答，按依赖传递答案后选择一个原子动作，全部调用成本计入 BALROG 结果。节点问答、耗时、结束原因和资源哈希记录在 CSV 的 `Reasoning` 列；执行中也逐节点写入 `eval.log`。

这是基于官方策略的 BALROG 适配实现，动作匹配失败按官方规则回退 Do；默认模型与观测和原论文不同。来源、许可和完整差异见 [SPRING 说明](prompts/spring/README.md)。

## ReAct

```bash
./run.sh --agent react --episodes 1 --max-steps 3 --seed 0
```

ReAct 允许自主稀疏思考：输出 `{"rationale":"推理"}` 时只更新上下文，继续调用模型；输出 `{"action":"Do"}` 时才推进一个游戏步，简单动作可直接输出，也可附带简短理由。`configs/react.yaml` 的 `max_thoughts` 默认 3，最多进行 3 次独立思考，再请求动作；全部调用的 Token 一并计入结果。日志记录全部 `thoughts`、`llm_calls` 及 `protocol: sparse-thought-v1`。

下一步会看到历史思考、提交动作及环境的真实反馈；每份观测保留当时的库存、生命状态与周围环境。历史窗口默认 16 步（包含当前观测），按完整观测/决策对截断，每回合重置。示例和协议固定放在首条指令中，历史只保存实际轨迹。提示词包含两个本项目编写的 Crafter 示例。

这是 [ReAct](https://react-lm.github.io/) 在 Crafter 上的适配。原论文没有 Crafter 标准实现；JSON 协议、历史窗口和思考上限是本项目的适配。来源版本、保留机制及完整差异见 [ReAct 说明](prompts/react/README.md)。旧版固定一次调用合并理由和动作，新旧结果应按协议版本区分。
