# BPE Tokenizer

[Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)广泛用于LLM Tokenizer

1. 设置预期词表大小
2. 初始化词表，大小为256（byte-level）
3. 统计相邻字节对出现的频率，将频率最高的字节对合并加入词表
4. 重复步骤3，直至词表大小达到预期或所有相邻字节对的频率都为1

