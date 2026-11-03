\# 验收自动化说明



纯 PC 端，pytest 驱动。不需要板子、Vivado、外网。



\## 怎么跑



在当前目录（sim/acceptance）执行：



\- 跑测试：`python -m pytest . -q`

\- 出报告：`python run\_acceptance.py`



**运行后将**报告输出到 `sim/records/acceptance_report.json` 和 `.md`。



\## 文件说明



| 文件 | 内容 |

| --- | --- |

| conftest.py | 路径推导、mock\_serial、real\_serial 两个 fixture |

| test\_frame\_protocol.py | 帧协议边界 14 例 |

| test\_tool\_contract.py | CLI 契约 9 例 |

| test\_t1a\_pipeline.py | 全链路统计口径 8 例 |

| test\_repo\_consistency.py | 仓库一致性 8 例 |

| run\_acceptance.py | 一键跑并出 JSON/MD 报告 |



\## 用例覆盖



\- 协议边界：正常帧、逐字节、随机分片、前导垃圾、假帧头、双 A5、粘包、坏 CRC、截断、尾部多余、seq 回绕、下行长度异常、CRC 覆盖含帧头、帧文件尺寸异常

\- 工具契约：--help、--dry-run、--limit、--classes 校验、CSV 表头、报告矩阵、门限缺失、README 文档

\- 统计口径：准确率公式、混淆矩阵对角线和、行列合计、报告百分比、Wilson 区间

\- 仓库一致性：位宽算术、字数整除、规范声明、表格求和、CRC 文档、代码实现、v2.2 帧长、README 口径



\## SKIP 规则



SKIP 不计失败，只在这些情况发生：



1\. 没有 <COM_PORT> 串口（`test\_real\_serial\_optional`）

2\. 缺 `data/vectors_t1a/labels.csv`

3\. 缺 `sim/records/acceptance_matrix.md`

4\. 缺 `05\_权重布局与导出规范.md`、`18\_协议细节确认\_CRC覆盖范围.md` 或 v2.2 文档

5\. bit 流布局章节解析不出表格



\## 怎么加新用例



1\. 在对应的 `test\_\*.py` 里加一个 `def test\_xxx()`，函数名以 `test\_` 开头

2\. 需要假串口就加参数 `mock\_serial`，需要真板子就加 `real\_serial`

3\. 路径别写绝对路径，用 `HERE = Path(\_\_file\_\_).resolve().parent` 往上推

4\. 加完跑 `python -m pytest . -q` 确认



\## 证明测试有约束力（必做一次）



1\. 打开 `test\_repo\_consistency.py`

2\. 把 `assert sum(parts) == 4448` 改成 `4400`

3\. 跑 `python run\_acceptance.py`

4\. 报告里应出现 `FAIL 1`，并列出该用例

5\. 改回 `4448`，再跑，恢复全绿



\## 不写死路径



所有路径都由本目录位置推出仓库根，代码中不含绝对路径。



