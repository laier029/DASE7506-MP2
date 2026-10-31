# Model Z Verifier 中文设计指南

## 1. 当前结论

“先做高精度负面筛选，再给幸存答案组打分”的方向已经得到实验支持。

- 最终评分、切换与恢复逻辑全部保留，但移除四个硬门控时：761/798。
- 加回四个硬门控后：768/798。
- 硬门控相对同一评分器的五套训练 OOF 增益为
  `[21, 20, 18, 20, 19]`，验证集增益为 +7。
- 最终 Model Z 相对 Model X 的五套 OOF 增益为
  `[10, 7, 8, 8, 5]`，验证集从 767 提升到 768。

因此，这套方法不是“只靠规则”或“只靠 learned scorer”，而是让两者承担不同责任：
硬门控只处理近乎确定的错误，较弱证据全部留在可逆的评分阶段。

## 2. 最终流程

### 阶段 0：唯一答案快速通道

若四个候选中只有一个可解析的最终答案组，直接返回该组的第一个可解析候选。

- 训练覆盖 2,641/4,755，验证覆盖 443/798。
- 两个开发 split 上均为 100% 正确。
- 这一步不判断推理质量，因为在官方指标下同组候选得到相同分数。

### 阶段 1：答案组级负面筛选

当前保留四类硬门控：

1. 同一答案组的多个推理共同包含未修复的广义算术错误；
2. 最终答案为分数；
3. 问题没有显式负数输入，但最终答案非正；
4. 显式 `Final Answer:` 与协议解析出的最终答案冲突。

门控作用在“答案组”，不是单条候选。只有组内所有支持证据都满足错误条件时才删除该组。
如果所有可解析组都被删除，则恢复全部可解析组，保证筛选不会让后续评分无答案可选。

### 阶段 2：幸存答案组评分

每条候选先得到 learned score，再叠加小而有界的结构信号：

- relative embedding ridge score；
- 按答案结构训练的 character TF-IDF/LinearSVC；
- 计算量覆盖率与重复出现覆盖率；
- 问题目标在推理末尾的保留程度；
- 稳定的隐式因子，例如 parents、round trip、月到年；
- remainder/difference 问题的小幅答案秩先验；
- `now/currently/everything` 问题的终端计算依赖覆盖。

答案组分数为组内最高候选分数加 `0.3 ×` 组内平均分数。这样既保留最强推理，也让多条独立支持提供有限加成。

### 阶段 3：窄门控切换与安全恢复

- omitted-factor switch 只在答案比值、问题因子、推理使用情况和分差同时满足条件时切换；
- boxed-answer recovery 只在目标一致、可见且对应协议可解析答案组时恢复；
- 缺失、互相冲突或协议不可见时保留原评分选择。

## 3. 为什么硬门控是合理的

| 门控 | 训练错误组 | 验证错误组 | 训练单侧 95% 精度下界 | 验证下界 | 验证净贡献 |
|---|---:|---:|---:|---:|---:|
| 分数答案 | 279/279 | 56/56 | 98.93% | 94.79% | +4 |
| 安全非正答案 | 122/122 | 25/25 | 97.57% | 88.71% | +2 |
| Final Answer 冲突 | 25/25 | 5/5 | 88.71% | 54.93% | 0 |
| 共同广义算术错误 | 6/6 | 3/3 | 60.70% | 36.84% | +1 |

“观测精度 100%”不等于置信度相同。算术门控样本很少，所以保留它的主要理由是：定义窄、五套 OOF 均 +1、验证 +1、且存在全删回退，而不是把 6/6 当成普适定律。

按答案结构拆解，硬门控在四答案相同的问题上贡献为 0；验证增益分别是
`3-1: +1`、`2-2: +2`、`2-1-1: +1`、全异: `+3`。这说明计算资源应集中在答案分歧行。

## 4. 为什么不改成软惩罚

固定扫描 `0.1–4.0` 的候选分数惩罚后：

- 分数答案与非正答案需要权重 4.0 才能匹配硬门控的正确数贡献；
- Final Answer 冲突需要 2.0；
- 广义算术在 OOF 上需要 1.0，验证集需要 2.0 才恢复硬门控增益。

没有较小软惩罚稳定超过硬门控。既然大惩罚已经接近“事实上的硬删除”，保留带安全回退的硬门控更简单、语义也更清楚。

## 5. 特征准入标准

新规则不要因为修好一个验证题就加入。采用以下顺序：

1. 先提出不依赖题号、reference 或 split 的机制解释；
2. 硬门控必须在答案组层面测精度，并设计全删回退；
3. 在主 OOF split 选择有限权重，再检查另外四套 balanced five-fold OOF；
4. 同时查看 fixes、harms、答案组变化和 churn，不能只看净增益；
5. 验证集不得回退，并用生产函数重新运行；
6. 最后检查任意长度输入、候选顺序、线程数、哈希、时间和资产大小。

普通 `total/altogether` 依赖特征是一个典型反例：`3-1` 权重 `-0.10` 的净 OOF 为
`[1, 0, 1, 0, 1]`，但两个 fold 分别发生 3 修复/3 伤害和 2 修复/2 伤害，验证答案组不变，且负方向缺少清晰机制，因此拒绝。

## 6. 当前错误与下一步研究方向

剩余 30 个验证错误的人工主因分类为：

- 11 个 multiplicity/time/participant accounting；
- 9 个 state transition/baseline ledger；
- 8 个 relational algebra/comparison；
- 1 个 discrete remainder；
- 1 个 geometry。

下一代 verifier 更值得研究的是：

1. **实体—状态账本**：明确初始量、增加、减少、最终量，避免只算变化量或漏加基线；
2. **参与者与时间展开**：把 parents、siblings、weekdays、round trips 等隐含计数转成角色化变量，而不是继续堆关键词；
3. **关系图或小型方程系统**：处理 twice、more than、difference、比例与未知数；
4. **目标条件的计算图匹配**：验证最终答案是否由回答目标所需的分支共同产生；
5. **不确定性路由**：低 margin 或高结构分歧时启用更强分析，高置信度行保持快速路径。

这些方向需要“结构化表示”，而不是更强的通用算术惩罚。已有实验显示，单位规则、普通 total 规则、严重算术规则和多数关键词先验都容易产生折间不稳定。

## 7. 当前冻结状态

- 入口：`submission.py`
- checkpoint：`runs/model_z_robust_semantic/`
- 验证：768/798 = 96.2406%
- 推理资产：1,575,004 bytes
- 四线程验证墙钟时间：约 1.56–1.57 秒
- 4,755 行训练输入压力预测加评分：约 6.49 秒墙钟时间
- 冻结后唯一一次测试：384/467 = 82.2270%
- 测试预测墙钟时间：约 1.94 秒；之后不再调参

验证集去掉 443 个唯一答案快速通道后，分歧题准确率为
325/355 = 91.5493%，仍比测试高 9.32 个百分点。因此测试下降不只是
“没有简单题”，也表明测试分歧题更难或存在分布变化；这只能作为结果解释，
不能用于测试后改模型。

测试前已运行并归档：

```bash
python experiments/source_manifest.py --verify
python experiments/pretest_freeze_audit.py
```

随后已按冻结记录执行一次正式 test prediction 与一次独立 scorer。
`pretest_freeze_audit.py` 现在会因 test 输出已经存在而按设计失败，因此不要覆盖测试前的 128 项审计记录；只需继续用 `source_manifest.py --verify` 确认推理源码和资产未变。

## 8. 证据文件索引

| 文件 | 用途 |
|---|---|
| `EXPERIMENTS.md` | 完整消融、拒绝路线、误差与资源分析 |
| `runs/model_z_robust_semantic/oof-audit.json` | Model Z 对 Model X 的五套 OOF fixes/harms |
| `runs/model_z_robust_semantic/gate-precision-audit.json` | 四个硬门控的组级精度与有限样本下界 |
| `runs/model_z_robust_semantic/hard-gate-marginal-audit.json` | 每个门控单独和 leave-one-out 贡献 |
| `runs/model_z_robust_semantic/hard-gate-structure-audit.json` | 门控贡献按答案结构拆解 |
| `runs/model_z_robust_semantic/gate-soft-vs-hard.json` | 固定权重网格下软惩罚与硬筛选比较 |
| `runs/model_z_robust_semantic/fallback-audit.json` | 全删回退及 boxed fallback 的自然支持 |
| `runs/model_z_robust_semantic/order-invariance-audit.json` | 候选重排后的答案组不变性 |
| `runs/model_z_robust_semantic/rebuild-audit.json` | 资产重建、选择 payload 与冻结文件一致性 |
| `runs/model_z_robust_semantic/source-manifest.json` | 27 个推理源码和 2 个资产的 SHA-256 |
| `runs/model_z_robust_semantic/runtime-dependency-audit.json` | 运行时实际加载模块与源码清单闭包核对 |
| `runs/model_z_robust_semantic/validation-error-catalog.json` | 30 个剩余错误的紧凑问题/答案/推理尾部目录 |
| `runs/model_z_robust_semantic/validation-error-taxonomy.json` | 人工错误类别到验证题号的映射 |
| `runs/model_z_robust_semantic/pretest-freeze-audit.json` | 128 项最终预测试检查结果 |
| `runs/model_z_robust_semantic/test-freeze.json` | 测试前冻结入口、资产和审计 hashes |
| `runs/model_z_robust_semantic/test-evaluation.json` | 唯一一次测试结果、时间与输出 hashes |
