import torch
from cs336_basics.Model.Softmax import Softmax
def decode(model, prompt, max_new_tokens = 50, temperature = 0.9, top_p = 0.9, eos_token_id = None):
    """
    Decode the prompt using the model.
    """
    out_put = prompt.clone()
    for i in range(max_new_tokens):
        logits = model(out_put)[..., -1, :] / temperature
        logits = Softmax(logits)
        logits = top_p_filter(logits, top_p)
        next_token = torch.multinomial(logits, num_samples = 1)
        out_put = torch.cat([out_put, next_token], dim = -1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return out_put

# def top_p_filter(probs, top_p = 0.9): 
#     """
#     Filter the logits using top-p filtering.
#     """
#     sorted_probs, sorted_indices = torch.sort(probs, descending=True)
#     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
#     cutoff = torch.searchsorted(cumulative_probs, top_p)
#     mask = torch.ones_like(probs, dtype=torch.bool)
#     mask[sorted_indices[:cutoff+1]] = False
#     probs = probs.clone()
#     probs[mask] = 0.0
#     probs /= probs.sum()
#     return probs

def top_p_filter(probs: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability > top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Create mask in original order
    indices_to_remove = torch.zeros_like(sorted_indices, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    
    # Apply mask and renormalize
    probs = probs.masked_fill(indices_to_remove, 0.0)
    return probs / probs.sum(dim=-1, keepdim=True)


from cs336_basics.Tokenizer.tokenizer import BPETokenizer
from cs336_basics.Model.TransformerLM import TransformerLM
from cs336_basics.Training.check_point import load_checkpoint
from einops import rearrange, repeat

if __name__ == "__main__":

    prompts = ["The quick brown fox jumps over the lazy dog",
               "Once upon a time,",
               "Tom and Lily are best friends.",]
    
    max_new_tokens = 256
    temperature = 1.2
    top_p = 0.9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    context_length = 256

    end_token = "<|endoftext|>"

    # Init tokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="./data/output/TinyStories_train_10000_token_vocab.bin",
        mergers_filepath="./data/output/TinyStories_train_10000_merges.bin",
        special_tokens=["<|endoftext|>"]
    )

    # Init model
    model = TransformerLM(
        vocab_size=10000,
        context_length=context_length,
        num_layers=4,
        num_heads=16,
        d_model=512,
        d_ff=1344,
        rope_theta=10000,
    )

    # 加载模型参数
    load_checkpoint(
        src="./data/model/final_model.pt",
        model=model,
        optimizer=None  
    )

    # 将模型移动到设备
    model.to(device)
    # 设置模型为评估模式
    model.eval()

    # 对输入进行分词
    inputs_ids = []
    len_inputs_ids = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.int32).to(device)
        inputs_ids.append(input_ids)
        len_inputs_ids.append(len(input_ids))
    
    len_inputs_ids = torch.tensor(len_inputs_ids, dtype=torch.int64, device=device)  # (batch_size,)
    
    # 如果batch中的序列长度不一致，需要进行padding填充
    # 否则无法作为一个矩阵输入到模型中
    # 这里先将所有输入左对齐，然后在右侧使用0填充
    pad_token_id = tokenizer.encode(end_token)[0]
    padded_inputs = torch.full(
        (len(inputs_ids), context_length), 
        fill_value=pad_token_id,
        dtype=torch.int32,
        device=device
    )
    for i, input_id in enumerate(inputs_ids):
        padded_inputs[i, :len(input_id)] = input_id


    end_token_id = tokenizer.encode(end_token)[0]
    is_end = torch.zeros(padded_inputs.shape[0], dtype=torch.bool, device=device)  # 记录每个序列是否已经结束

    # 生成阶段
    with torch.no_grad():
        for num in range(max_new_tokens):
            # (batch, max_seq_len) -> (batch_size, max_seq_len, vocab_size)
            logits = model(padded_inputs)
            index = len_inputs_ids - 1 + num
            index = repeat(index, 'b -> b 1 v', v=logits.shape[-1])  # (batch_size, 1, vocab_size)
            # 取出input_ids最后一个token的logits，这才是预测的token
            logits = torch.gather(logits, dim=1, index=index).squeeze(1) # (batch_size, vocab_size)

            # temperature 越大，logits更分布在数轴两端，输出越随机；
            # temperature 越小，logits分布都被压缩到0附近，输出越确定
            logits = logits / temperature
            # 计算softmax
            probs = Softmax(logits, dim=-1)

            # 使用top-p去除末尾概率
            probs = top_p_filter(probs, top_p)
            
            # 从概率分布中采样
            next_token_ids = torch.multinomial(probs, num_samples=1).to(dtype=torch.int32) # (batch_size, 1) 

            # 将采样的token添加到输出中
            next_token_index = (len_inputs_ids + num).unsqueeze(1) # (batch_size, 1)
            padded_inputs.scatter_(1, next_token_index, next_token_ids)

            # 更新is_end标志
            is_end = is_end | (next_token_ids == end_token_id)
            # 如果所有序列都已经结束，则提前退出
            if is_end.all():
                break
    
    # 解码输出序列
    outputs = []
    for i in range(padded_inputs.shape[0]):
        output_ids = padded_inputs[i, len_inputs_ids[i]:].cpu().numpy()
        output_text = tokenizer.decode(output_ids, end_token_id=end_token_id)
        outputs.append(output_text)
    
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: {output}")
            