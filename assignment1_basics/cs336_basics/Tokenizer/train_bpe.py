
from collections import Counter
from multiprocessing import Pool
import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Build final vocabulary
    vocab: dict[int, bytes] = {}
    token_id = 0

    # Add single-byte tokens first
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    
    # Add special tokens
    for special_token in special_tokens:
        vocab[token_id] = special_token.encode("utf-8")
        token_id += 1

    # Divide the whole file into small chunks
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        P = Pool(num_processes)
        async_results = []
        pre_tokens_before_count: list[bytes] = []
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            result = P.apply_async(
                pre_tokenize_chunk,
                args=(chunk, special_tokens),
            )
            async_results.append(result)
        P.close()
        P.join()
        for result in async_results:
            pre_tokens_before_count.extend(result.get())
        
    # Train BPE on the pre-tokenized chunks
    bp_counter: dict[bytes, int] = {}
    pre_tokens = Counter(pre_tokens_before_count)
    initial_bpe_tokens = {}
    for token_bytes, count in pre_tokens.items():
        # turn b' hello' into (b' ', b'h', b'e', b'l', b'l', b'o')
        # token_bytes[i:i+1] is the way to get a single bytes object slice
        initial_bpe_token_sequence = tuple(token_bytes[i:i+1] for i in range(len(token_bytes)))
        initial_bpe_tokens[initial_bpe_token_sequence] = count
    
    loop_count = vocab_size - len(special_tokens) - 256  # reserve 256 tokens for single-byte tokens
    bpe_tokens = dict(initial_bpe_tokens)  # copy initial tokens
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(loop_count):
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        for token_sequence, count in bpe_tokens.items():
            if len(token_sequence) < 2:
                continue
            for i in range(len(token_sequence) - 1):
                pair = (token_sequence[i], token_sequence[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        if not pair_counts:
            break
        
        # Find the most frequent pair
        most_frequent_pair = max(
            pair_counts.items(), 
            key=lambda item: (item[1], item[0])
        )[0]

        merges.append(most_frequent_pair)
        # Create new BPE tokens by merging the most frequent pair
        new_bpe_tokens: dict[tuple[bytes, ...], int] = {}
        merge_a, merge_b = most_frequent_pair
        merged_token = merge_a + merge_b
        # Add the new merged token to the vocabulary
        vocab[token_id] = merged_token
        token_id += 1
        for token_sequence, count in bpe_tokens.items():
            new_sequence = []
            i = 0
            while i < len(token_sequence):
                if i < len(token_sequence) - 1 and token_sequence[i] == merge_a and token_sequence[i + 1] == merge_b:
                    new_sequence.append(merged_token)
                    i += 2
                else:
                    new_sequence.append(token_sequence[i])
                    i += 1
            new_bpe_tokens[tuple(new_sequence)] = count
        bpe_tokens = new_bpe_tokens
    return vocab, merges

def remove_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
    """
    Remove special tokens from the chunk and separate them.
    """
    pattern = "|".join(map(re.escape, special_tokens))
    tokens = re.split(pattern, chunk)
    return [token for token in tokens if token]

def pre_tokenize_chunk(chunk: str, special_tokens: list[str]) -> list[bytes]:
    """
    Pre-tokenize the chunk by separating special tokens and normal text.
    """
    tokens = remove_special_tokens(chunk, special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = []
    
    for token in tokens:
        for match in re.finditer(PAT, token):
            pre_tokens.append(match.group(0).encode("utf-8"))
    return pre_tokens


def to_files(self, vocab_filepath: str, merges_filepath: str) -> None:
    """将训练结果保存到文件"""
    # 保存词汇表
    with open(vocab_filepath, 'wb') as f:
        # 写入词汇表大小
        f.write(len(self.token_vocab).to_bytes(4, byteorder='little'))
        
        # 写入每个token: <id(4字节)><长度(4字节)><token内容(bytes)>
        for token_id, token in self.token_vocab.items():
            f.write(token_id.to_bytes(4, byteorder='little'))
            f.write(len(token).to_bytes(4, byteorder='little'))
            f.write(token)
    
    # 保存合并规则
    with open(merges_filepath, 'wb') as f:
        # 写入合并规则数量
        f.write(len(self.merges).to_bytes(4, byteorder='little'))
        
        # 写入每个合并规则: <第一部分长度(4字节)><第一部分内容(bytes)><第二部分长度(4字节)><第二部分内容(bytes)>
        for first, second in self.merges:
            f.write(len(first).to_bytes(4, byteorder='little'))
            f.write(first)
            f.write(len(second).to_bytes(4, byteorder='little'))
            f.write(second)