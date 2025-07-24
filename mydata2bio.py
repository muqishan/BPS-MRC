import json
from transformers import BertTokenizerFast

# 初始化分词器
tokenizer = BertTokenizerFast.from_pretrained("pubmedbert")

# 示例数据
with open('ner_utils/cell_ner/datasets/validation.json', 'r') as file:
    datas = json.load(file)

# 函数：将数据转换为BIO格式
def convert_to_bio(text, entities, tokenizer):
    tokens = tokenizer.tokenize(text)
    token_spans = tokenizer.encode_plus(text, return_offsets_mapping=True)["offset_mapping"][1:-1]
    bio_tags = ["O"] * len(tokens)

    for entity in entities:
        entity_start, entity_end = entity["char_span"]
        entity_type = entity["type"]
        # if entity_type == 'Signal pathway':
        #     entity_type = 'Signal'
        entity_started = False
        # if entity_start == 0:
        #     print()
        for idx, span in enumerate(token_spans):
            if span[0] < entity_start:
                continue
            if span[0] >= entity_end:
                break
            if span[0] >= entity_start and span[1] <= entity_end:
                if not entity_started:
                    bio_tags[idx] = f"B-{entity_type}"
                    # bio_tags[idx] = f"{entity_type}"
                    entity_started = True
                else:
                    bio_tags[idx] = f"I-{entity_type}"

    return list(zip(tokens, bio_tags))


# 转换数据
result = []
for data in datas:
    result.append(convert_to_bio(data["text"], data["entity_list"], tokenizer))

# 保存为文本文件
with open("ner_utils/cell_ner/datasets/validation.txt", "w", encoding="utf-8") as file:
    for bio_data in result:
        for token, tag in bio_data:
            file.write(f"{token}\t{tag}\n")
        file.write("\n")
