# ReAct 来源与 Crafter 适配

原论文：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/html/2210.03629v3)，重点对照第 2、4 节。

作者项目：[ysymyth/ReAct](https://github.com/ysymyth/ReAct)，本次核验固定提交 `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`：

- [ALFWorld](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/alfworld.ipynb)：独立 `think:` 轮次后继续调用，思考只更新轨迹。
- [WebShop](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/WebShop.ipynb)：`think[...]` 不改变环境会话状态。

保留的机制：模型自主决定是否思考；思考用于分解目标、跟踪进展、切换子目标和处理真实反馈；思考不推进游戏；动作执行后的观察来自环境。模型也可直接行动。每步保存全部思考与动作，累加所有模型调用的 Token，每回合重置。

必要适配：原论文的任务是 HotpotQA、FEVER、ALFWorld、WebShop，没有 Crafter 实验。这里采用 BALROG 文本观察和 JSON：仅 `rationale` 表示独立思考，`action` 表示游戏动作，可同时包含简短理由。`react.txt` 的 Crafter 示例由本项目编写，沿用真实游戏规则，并非作者原始示例。

`configs/react.yaml` 中 `max_thoughts: 3` 表示每个环境步允许至多 3 次独立思考，再请求动作。轮次上限、默认 16 步历史窗口和公共模型配置是本项目的预算控制，不是论文实验参数。超过思考上限仍无动作、坏 JSON 或无效动作会保留错误证据并交给 BALROG 校验。控制器的思考确认消息不是新的环境观察。

旧版本固定一次调用同时输出理由和动作；当前版本支持独立思考，日志使用 `protocol: sparse-thought-v1` 标识，并记录 `thoughts` 与 `llm_calls`。比较新旧结果时应区分协议版本。原生 Naive 与 ReAct 的比较属于智能体基线对比；严格隔离思考效果还需要相同示例、移除思考的 Act 对照。这里不宣称复现原论文分数。
