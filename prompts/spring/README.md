# SPRING 来源与适配范围

本目录用于复现 **SPRING 的问答策略**，不是论文成绩的严格复现。

## 固定来源与许可

- Yue Wu 等，SPRING: Studying the Paper and Reasoning to Play Games：[论文](https://arxiv.org/abs/2305.15486)。
- 问题、依赖关系、消息结构、动作匹配参照 [Holmeswww/SPRING](https://github.com/Holmeswww/SPRING/blob/c0369b9127ab9ec63797797a3952be9e119334e1/GPT_Actor_Chat.ipynb)，提交 `c0369b9127ab9ec63797797a3952be9e119334e1`。Copyright (c) 2023 Yue Wu，MIT，全文见 `LICENSE.SPRING.txt`。
- 游戏知识来自官方笔记本使用的 [Microsoft SmartPlay Crafter 手册](https://github.com/microsoft/SmartPlay/blob/9dec5f2247da0dc88a47fe3daf3fb4fa02629b89/src/smartplay/crafter/assets/crafter_ctxt.pkl)，提交 `9dec5f2247da0dc88a47fe3daf3fb4fa02629b89`。SmartPlay 项目按 CC BY 4.0 发布，全文见 `LICENSE.SmartPlay.txt`。
- `sources.json` 保存来源版本、原始资源哈希、转换说明和本地文件哈希。

`questions.json` 保留原问题文字（包括拼写）、依赖顺序，为原来的两个隐式根节点增加空依赖列表。`manual.txt` 从原 pickle 的单个 Unicode 字符串静态提取，并应用 SmartPlay 环境中的五处字符串替换：移除 LaTeX 格式要求，统一四个移动方向的名称。不执行 pickle、不需要安装 SmartPlay，运行时不联网下载资源。

该手册是官方提供的知识提取结果，不是人工重新总结的规则。本实现保留其中可能不准确的内容（例如蜘蛛、隧道等描述），避免在复现时悄悄增强知识；如要修正规则，应另作实验变量。

## 保留的机制

每个环境步重新求解完整的 9 节点问题 DAG：

1. 上一个动作是什么？
2. 上一个动作是否成功？
3. 可见对象、资源和交互要求。
4. 对象交互要求是否满足？
5. 三个子任务及优先级。
6. 首要子任务的要求和第一步。
7. 五个候选动作及要求、优先级。
8. 候选动作要求是否满足？
9. 选择最佳可执行动作。

严格按依赖求解，每个节点只看到直接父节点的问答、固定手册和最近两步完整观测；不将所有前序回答堆进提示词，不跨环境步缓存答案。依赖优先的确定性遍历替代原代码的集合迭代，以便复现实验。中间回答是自由文本，不使用现有 BaseAgent 的 JSON 解析。

## BALROG 适配差异

- 继续使用 BALROG 的局部文本观测、环境和客户端，不切换 SmartPlay。每份观测合并 `long_term_context` 与 `short_term_context`，加入上次提交动作；不读取环境隐藏状态。
- 最近两步观测是 SPRING 固定的策略记忆，不受通用 `history_length` 控制；不使用其他基线的 16 步对话历史。
- 沿用官方按动作列表顺序、不区分大小写的子串匹配。未匹配时按论文 §2.2 和官方笔记本回退 Do；日志保留原回答、解析失败标识与回退动作。最终回答提到多个动作时，仍沿用官方取列表中首个匹配的行为。
- 沿用 BALROG 回合初始化，不复制官方交互循环中额外执行的动作。观测步编号从 0 连续递增，不复制笔记本的 `i+i` 编号。
- 每回合清空观测和推理状态。每步通常 9 次模型调用；所有节点的 token 合计写入原生结果，每节点问答、耗时、结束原因写入 CSV 的 Reasoning 和 eval.log。
- 默认使用项目配置的 Qwen/Ollama、2000 步上限与原生进度指标，不宣称复现 GPT-4 的论文分数。更换模型、预算和观测设置需单独报告。

启动：`./run.sh --agent spring --episodes 1 --max-steps 3 --seed 0`。

## 本次协议核验

对照论文 §2.2 和上述固定提交，9 个问题、直接父节点上下文、自由文本、两步观测及动作顺序均已核对。手册与官方静态资源转换后逐字一致。修正了早期版本的 Noop 回退；新版本使用官方 Do 回退，历史结果需按代码版本区分。论文作者的 `llm_api.get_query` 未随项目发布，因此项目继续统一使用公共模型与推理参数。
