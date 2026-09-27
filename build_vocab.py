"""构建标准纯字符级词表（三区全部严格对齐 2 的整数次幂，硬件 GEMM 与内存对齐最优）：

分区设计（总计 8192 = 2^13）：
1. ASCII 字符区 (0..127，共 128 = 2^7 项):
   - 0..127 槽位精确对应 ord(c)，即 chr(i) == i，如 '\n' 在 10, ' ' 在 32, 'A' 在 65, 'a' 在 97。
2. 系统机制词区 (128..255，共 128 = 2^7 项，对齐 2 的整数幂):
   - 包含特殊机制符：<pad>, <unk>, <bos>, <eos>, <sep>, <resp>, <topic>, <cont>, <slice>, <null>
   - 包含常用全角标点符号（，。！？；：“”‘’等）及预留系统槽位 <sys_i>。
3. 中文字符区 (256..8191，共 7936 项，起始点严格对齐 2^8 = 256 边界):
   - 覆盖 7936 个高频汉字，不足部分由 <res_han_i> 预留填充。
"""
import glob
import os
from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, Regex, AddedToken

# 1. 系统机制词
SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>", "<sep>",
    "<call>", "<result>", "<think>", "<answer>", "<tool>",
    "<search>", "<resp>", "<topic>", "<cont>", "<slice>", "<null>"
]

# 常见全角标点
FULLWIDTH_PUNCT = list("。！？，、；：“”‘’（）《》…—·～「」『』【】")


def build_pow2_aligned_tokenizer(out_path: str = "src/nano_char_tokenizer/tokenizer.json"):
    print("构建三区严格对齐 2 的整数次幂规范词表（8192 = 2^13）...")

    vocab = {}

    # ── 区 1: ASCII 直通区 (id 0..127, 共 128 = 2^7 项, 精确对齐 ord 值) ──
    for i in range(128):
        ch = chr(i)
        vocab[ch] = i

    assert len(vocab) == 128  # 2^7
    assert vocab["\n"] == 10
    assert vocab[" "] == 32
    assert vocab["0"] == 48
    assert vocab["A"] == 65
    assert vocab["a"] == 97

    # ── 区 2: 系统词与标点区 (id 128..255, 严格等于 128 = 2^7 项) ──
    sys_tokens = SPECIAL_TOKENS + [p for p in FULLWIDTH_PUNCT if p not in SPECIAL_TOKENS]
    sys_slots = 128  # 严格对齐 2^7 = 128 个槽位
    num_reserved = sys_slots - len(sys_tokens)
    for i in range(num_reserved):
        sys_tokens.append(f"<sys_{i}>")

    for tok in sys_tokens:
        vocab[tok] = len(vocab)

    assert len(vocab) == 256  # 2^8 边界

    # ── 区 3: 汉字区 (id 256..8191, 共 7936 项, 起始位置严格在 2^8) ──
    txt_files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))
    if not txt_files:
        raise FileNotFoundError("nanoSeek 对话语料未找到！")

    counter = Counter()
    for p in txt_files:
        with open(p, encoding="utf-8", errors="ignore") as f:
            counter.update(f.read())

    hanzi = [(c, cnt) for c, cnt in counter.items() if "\u4e00" <= c <= "\u9fff"]
    hanzi.sort(key=lambda x: -x[1])

    target_hanzi_count = 8192 - 256  # 7936
    selected_hanzi = [c for c, _ in hanzi[:target_hanzi_count]]

    for ch in selected_hanzi:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    # 填补预留汉字槽对齐 8192 (2^13)
    rem = 8192 - len(vocab)
    for i in range(rem):
        vocab[f"<res_han_{i}>"] = len(vocab)

    assert len(vocab) == 8192, f"Expected 8192, got {len(vocab)}"

    print(f"词表构建完成（总计 8192 = 2^13 项，各区对齐 2 的整数幂）：")
    print(f"  区 1 (ASCII 直通区 0..127): 128 项 (2^7)，满足 id == ord(c)")
    print(f"  区 2 (系统词与标点 128..255): 128 项 (2^7)，起始 128 (2^7)，结束 255")
    print(f"  区 3 (常用汉字区 256..8191): 7936 项，起始点严格对齐 256 (2^8)，总计对齐 8192 (2^13)")

    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")

    for tok_name in SPECIAL_TOKENS:
        tok.add_special_tokens([AddedToken(tok_name, special=True)])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tok.save(out_path)
    print(f"成功保存至 {out_path} ✅")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    build_pow2_aligned_tokenizer(os.path.join(out_dir, "src/nano_char_tokenizer/tokenizer.json"))
