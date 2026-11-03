frame_words.hex     : 620 x 128 words (512 B/frame, word i = bytes 4i..4i+3 little endian)
scores.txt          : per line: f0 f1 f2 f3 pred  (Q6 integer scores from gen_golden_t8_v2.py)
tb_t8_expected.hex  : 620 lines, 80 bit/frame: [79:76]=pred [75:57]=f3 [56:38]=f2 [37:19]=f1 [18:0]=f0
                      (19-bit two's complement; one line per frame for $readmemh in tb_t8_620.v)
weights             : board/weights/onsite_cur16_1102/weights.hex (271 words, v5 beat-major, CRC16 0xB8C7)
frames              : data/vectors_t1a/*.bin, order = labels.csv order
generator           : src/pc/make_t8_rtl_data.py (self-checks + SHA256 below)
