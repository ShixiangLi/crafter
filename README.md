# Crafter 智能体算法研究

所有智能体统一使用 BALROG 的环境、观测、模型客户端、历史管理、动作校验和评估指标。外层仅负责实验配置与算法接入，不修改 `BALROG/` 源码。

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

# 也可以选择实验配置
./run.sh --config experiments/planner.yaml

# 查看生效配置，不调用模型、不生成实验结果
./run.sh --agent ours --dry-run
```

`run.sh` 使用项目 `.venv`，调用 `scripts/evaluate.py`。运行模型前，确保 Ollama 已启动且模型已安装。所有方法通过 BALROG 客户端访问 Ollama 的 `/v1` 兼容接口。

| 方法 | 实现 |
| --- | --- |
| `naive` | 直接使用 BALROG 原生 NaiveAgent |
| `react` | 外层轻量基线：简短理由和一个动作 |
| `planner` | 外层基线：默认每 10 步更新目标计划，再选择动作 |
| `ours` | 新算法入口，目前与 planner 相同，尚未实现新方法 |

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

公共配置统一所有方法的默认推理设置：温度 1.0、生成上限 8192、16 条历史、纯文本输入、单工作进程。比较方法时请保持模型与预算一致，并报告实际 token 消耗。

已移除旧 `runner` 切换、`--seeds`、`num_ctx`、独立 Ollama JSON 客户端和 research 输出格式；统一使用 `--episodes`、`--seed` 及 BALROG 参数。本启动器每次创建新运行，不支持断点续跑。

## 代码结构

```text
run.sh                    启动入口
scripts/evaluate.py       配置组装，调用 BALROG EvaluatorManager 与汇总函数
agents/factory.py         扩展上游工厂；原生方法委托给 BALROG
agents/base_agent.py      自定义方法共享的 JSON 决策与 LLMResponse 适配
agents/react_agent.py     简短理由基线
agents/planner_agent.py   规划基线
agents/our_agent.py       新算法入口
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
