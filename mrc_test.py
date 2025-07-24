from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import torch.nn.functional as F

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载并加载模型和tokenizer
# model_name = "bigwiz83/sapbert-from-pubmedbert-squad2"
output_dir = "mrc_runs/mrc20241031_10_"
# output_dir = "BioM-ELECTRA-Large-SQuAD2-BioASQ8B"
model = AutoModelForQuestionAnswering.from_pretrained(output_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# 示例数据a
q1 = "Where does IFN-γ promotes PD-L1 occur?"
c1 = "Originally identified in studies of cellular resistance to viral infection, interferon (IFN)-γ is now known to represent a distinct member of the IFN family and plays critical roles not only in orchestrating both innate and adaptive immune responses against viruses, bacteria, and tumors, but also in promoting pathologic inflammatory processes. IFN-γ production is largely restricted to T lymphocytes and natural killer (NK) cells and can ultimately lead to the generation of a polarized immune response composed of T helper (Th)1 CD4 T cells and CD8 cytolytic T cells. In contrast, the temporally distinct elaboration of IFN-γ in progressively growing tumors also promotes a state of adaptive resistance caused by the up-regulation of inhibitory molecules, such as programmed-death ligand 1 (PD-L1) on tumor cell targets, and additional host cells within the tumor microenvironment. This review focuses on the diverse positive and negative roles of IFN-γ in immune cell activation and differentiation leading to protective immune responses, as well as the paradoxical effects of IFN-γ within the tumor microenvironment that determine the ultimate fate of that tumor in a cancer-bearing individual."
question = q1
context = c1
# 对问题和上下文进行编码
# 对问题和上下文进行编码
# 对问题和上下文进行编码
inputs = tokenizer.encode_plus(
    question,
    context,
    return_tensors="pt",
    add_special_tokens=True,
)
input_ids = inputs["input_ids"].tolist()[0]

# 打印输入信息
# print(f"Input IDs: {input_ids}")
# print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")

# 获取模型的输出
outputs = model(**inputs.to(device))
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits

# # 打印模型输出分数
# print(f"Answer start scores: {answer_start_scores}")
# print(f"Answer end scores: {answer_end_scores}")

# 使用 softmax 函数计算概率
start_probs = F.softmax(answer_start_scores, dim=-1)
end_probs = F.softmax(answer_end_scores, dim=-1)

# 获取最高分的起始位置和结束位置
answer_start = torch.argmax(start_probs)
answer_end = torch.argmax(end_probs) + 1

# 打印起始和结束位置
# print(f"Answer start index: {answer_start}")
# print(f"Answer end index: {answer_end}")

# 计算置信度
confidence = (start_probs[0, answer_start] * end_probs[0, answer_end - 1]).item()

# 解码答案
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

# 获取答案在字符级别的开始和结束位置
inputs_with_offsets = tokenizer.encode_plus(question, context, return_offsets_mapping=True, add_special_tokens=True)
offsets = inputs_with_offsets['offset_mapping']

# 打印偏移映射
# print(f"Offsets: {offsets}")

# 查找答案的字符级别位置
sep_index = input_ids.index(tokenizer.sep_token_id)
context_start = sep_index + 1  # Context starts right after the first [SEP] token

# 打印上下文起始位置
# print(f"Context start index: {context_start}")

# 确保答案索引在上下文范围内
adjusted_answer_start = answer_start - context_start
adjusted_answer_end = answer_end - context_start

# 打印调整后的起始和结束位置
# print(f"Adjusted answer start index: {adjusted_answer_start}")
# print(f"Adjusted answer end index: {adjusted_answer_end}")

# 检查调整后的索引是否有效
# if adjusted_answer_start < 0 or adjusted_answer_end > len(offsets) or adjusted_answer_end <= adjusted_answer_start:
#     raise IndexError("Adjusted answer indices are out of the offset range")

answer_start_char = offsets[answer_start][0]
answer_end_char = offsets[answer_end - 1][1]

print(f"Answer: {answer}")
print(f"Confidence: {confidence:.4f}")
print(adjusted_answer_start,adjusted_answer_end)
# print(f"Answer Start Position (char level): {answer_start_char}")
# print(f"Answer End Position (char level): {answer_end_char}")
# print(f"Extracted Answer: '{context[answer_start_char:answer_end_char]}'")
