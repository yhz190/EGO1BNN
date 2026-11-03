# golden 逐层向量格式说明

由 src/pc/golden_hw.py 生成，供 RTL testbench 与 T8 逐层比对。

## 目录
- frame_XXX_<源文件名>/ : 每帧一个目录
- summary.csv : 全部帧逐帧结果
- report.json : 自检结论

## 每帧文件
- L1_cnt.txt  : L1 匹配计数，8 通道 x 64 行，每行 64 个十进制数(0~9)
- L1_act.hex  : L1 激活位图，8x64 行，每行 16 个 hex(=64bit，MSB 在左)
- L2_act.hex  : 同上，8x64 行
- L3_act.hex  : 同上，4x64 行
- L4_feat.hex : L4 池化后 1024bit，16 行 x 16 hex
- L5_score.txt: 4 行得分 + 1 行 argmax

## 约定
- 行序 [channel][y]，x 从左到右，MSB 在左
- 激活 bit=1 表示 +1，bit=0 表示 -1
- 类别: 0=rock 1=paper 2=scissors 3=junk
- hw_eval 的 FC 偏置是浮点，RTL 是整数 thetaFC：
  L5 只比 argmax，L1~L4 位图必须 0 差异
