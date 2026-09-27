# nano-char-tokenizer

nanoSeek 与 DTSeek 共用的中文字符级分词器模块。

- **词表大小**：8192（2^13，对齐硬件 GEMM 边界）
- **四区稀疏布局**：
  1. 基础文本字符区（ASCII + 全角标点）
  2. 机制符区间（`<pad>`, `<unk>`, `<bos>`, `<eos>`, `<sep>`, `<resp>`, `<topic>` 等）
  3. 常用汉字区（按词频降序，覆盖现代日常中文）
