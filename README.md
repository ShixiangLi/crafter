# Crafter 智能体算法研究

所有智能体统一使用 BALROG 的环境、观测、模型客户端、动作校验和评估指标。外层仅负责实验配置与算法接入，不修改 `BALROG/` 源码。

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

# 也可以选择实验配置
./run.sh --config experiments/planner.yaml

# 查看生效配置，不调用模型、不生成实验结果
./run.sh --agent ours --dry-run
```

`run.sh` 使用项目 `.venv`，调用 `scripts/evaluate.py`。运行模型前，确保 Ollama 已启动且模型已安装。所有方法通过 BALROG 客户端访问 Ollama 的 `/v1` 兼容接口。

`run.sh` 自动为本次进程及其子进程设置本机代理绕过（`127.0.0.1`、`localhost`、`::1`），合并并保留已有 `NO_PROXY` / `no_proxy` 规则。不会修改系统代理或影响其他正在运行的实验。直接执行 Python 入口时不经过此设置。

| 方法 | 实现 |
| --- | --- |
| `naive` | 直接使用 BALROG 原生 NaiveAgent |
| `react` | ReAct：保留观测、推理与动作历史，根据真实反馈调整决策 |
| `planner` | 外层基线：默认每 10 步更新目标计划，再选择动作 |
| `spring` | 官方 9 节点问答 DAG、游戏手册和最近两步观测 |
| `ours` | 新算法入口，目前与 planner 相同，尚未实现新方法 |

## 按 GPU 启动独立 Ollama

```bash
# 两个终端分别使用 GPU 0 和 GPU 1
./run.sh --gpu 0 --agent react
./run.sh --gpu 1 --agent spring

# 只检查配置，不启动服务或模型
./run.sh --gpu 1 --agent spring --dry-run
```

`--gpu N` 使用 `nvidia-smi` 的 GPU 编号，通过 GPU UUID 绑定独立 Ollama，端口为 `11500 + N`。首次自动启动，后续仅复用本脚本记录且绑定匹配的进程；端口被其他服务占用时会报错，不重启或停止原有服务。实现依据 [Ollama GPU 选择说明](https://docs.ollama.com/gpu)。

不传 `--gpu` 时保持原来的服务配置。`--gpu` 不能与 `client.base_url` 同时指定。独立服务使用当前用户的模型目录（或继承 `OLLAMA_MODELS`）；若模型不在其中，先执行例如 `OLLAMA_HOST=127.0.0.1:11501 ollama pull qwen3:8b`。不会自动下载模型。

服务在实验退出后保留，便于复用；日志、PID 和启动锁位于 `outputs/ollama/gpu_N.*`。需要停止时，确认对应实验已结束，再执行例如 `kill "$(cat outputs/ollama/gpu_1.pid)"`。已有服务若也在使用同一张 GPU，仍会共享算力；指定不同 GPU 的独立服务才能分开队列。同一个 `--gpu N` 的实验会共用同一服务。

## 配置与种子

公共设置位于 `configs/default.yaml`，实验配置位于 `experiments/`。优先级为：末尾 BALROG Hydra 覆盖参数 > 命令行选项 > 实验配置 > 公共配置 > 上游默认配置。

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

已移除旧 `runner` 切换、`--seeds`、`num_ctx`、独立 Ollama JSON 客户端和 research 输出格式；统一使用 `--episodes`、`--seed` 及 BALROG 参数。本启动器每次创建新运行，不支持断点续跑。

## 代码结构

```text
run.sh                    启动入口
scripts/evaluate.py       配置组装，调用 BALROG EvaluatorManager 与汇总函数
agents/factory.py         扩展上游工厂；原生方法委托给 BALROG
agents/base_agent.py      自定义方法共享的 JSON 决策与 LLMResponse 适配
agents/react_agent.py     ReAct 推理、动作、反馈闭环
agents/planner_agent.py   规划基线
agents/our_agent.py       新算法入口
agents/spring_agent.py    SPRING 问答策略
modules/spring.py         问题图遍历与动作匹配
prompts/spring/           官方问题、手册、来源版本和许可
modules/planner.py        计划更新
modules/executor.py       按计划选择原子动作
prompts/                  自定义算法提示词
configs/、experiments/    公共设置与实验选择
BALROG/                   上游环境、客户端、历史和评估实现
outputs/results/          实验结果
tests/                   接入契约与真实环境集成测试
```

开发新算法时，主要修改 `agents/our_agent.py`，按需添加算法组件。实现 BALROG 接口：

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
└── crafter/
    ├── crafter_summary.json      进度、标准误、步数与 token
    └── default/
        ├── default_run_00.json   每回合成就、种子、参数与成本
        └── default_run_00.csv    动作、理由、观测、奖励和结束标志
```

BALROG 的 `progression_percentage` 为平均解锁成就数 / 22 × 100；它不是 Crafter score。既有 `outputs/results/balrog/` 数据保留，可按相同原生格式读取。

缺失回合不会自动进入 BALROG 成绩均值；运行不完整时启动器返回非零状态，并记录 `status.json`。分析结果时同时检查回合完整性。CSV 的 Action 是模型提交的候选动作；非法候选实际执行 Noop，回合 JSON 的 `failed_candidates` 记录非法候选。

## 验证

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

测试使用真实 BALROG/Crafter 环境与模拟模型响应，不需要连接 Ollama，也不代表模型性能测试。

## SPRING

```bash
./run.sh --config experiments/spring.yaml --episodes 1 --max-steps 3 --seed 0
```

每个环境步执行 9 次问答，按依赖传递答案后选择一个原子动作，全部调用成本计入 BALROG 结果。节点问答、耗时、结束原因和资源哈希记录在 CSV 的 `Reasoning` 列；执行中也逐节点写入 `eval.log`。

这是基于官方策略的 BALROG 适配实现，默认模型、观测和无效动作回退与原论文不同。来源、许可和完整差异见 [SPRING 说明](prompts/spring/README.md)。

## ReAct

```bash
./run.sh --agent react --episodes 1 --max-steps 3 --seed 0
```

ReAct 每步一次调用，输出 `{"rationale":"简短推理","action":"Do"}`，简单动作可省略 `rationale`。下一步会看到历史推理、提交动作及环境的真实反馈；每份观测保留当时的库存、生命状态与周围环境。历史窗口由 `agent.max_text_history` 控制（默认 16，包含当前观测），按完整观测/决策对截断，不使用 `agent.max_cot_history` 再删除推理。每回合重置。

提示词包含两个自编的短交互示例，覆盖制作成功、直接行动和交互失败后的调整，相关状态变化由真实 Crafter 环境测试验证。示例不作为当前回合经历；不添加虚构环境反馈，不引入额外规划器或跨回合反思模块。

这是 [ReAct](https://react-lm.github.io/) 在 Crafter 上的适配：使用 JSON 表达 Thought/Action、有限历史窗口和项目模型配置。原论文没有 Crafter 标准实现，不能把本项目结果视为原论文成绩复现。旧版 ReAct 仅记录简短理由；新旧结果应按代码版本区分。
