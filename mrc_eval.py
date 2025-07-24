# coding=utf-8
import json
import argparse
import logging
import os
import timeit
from datetime import datetime
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers import AutoModelForQuestionAnswering,AutoTokenizer,squad_convert_examples_to_features
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV2Processor

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            if args.model_type in ["bert", "xlnet", "albert", "electra"]:
                inputs["token_type_ids"] = batch[2]

            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs.to_tuple()]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # 计算预测结果
    os.makedirs(args.output_dir,exist_ok=True)
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix)) if args.version_2_with_negative else None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # 计算F1和精确度得分
    results = squad_evaluate(examples, predictions)
    return results

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    processor = SquadV2Processor()
    examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
    )
    if output_examples:
        return dataset, examples, features
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # 必要参数
    parser.add_argument(
        "--model_type",
        default='electra',
        type=str,
        help="模型类型。",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='BioM-ELECTRA-Large-SQuAD2-BioASQ8B',
        type=str,
        help="预训练模型的路径或名称。",
    )
    parser.add_argument(
        "--output_dir",
        default='mrc_runs',
        type=str,
        help="模型预测结果的输出目录。",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="输入的评估文件。",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="输入数据目录，应该包含任务的.json文件。",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        default=True,
        help="如果为真，则SQuAD示例包含一些没有答案的情况。",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="如果null_score - best_non_null大于该阈值，则预测为空。",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="WordPiece分词后的最大总输入序列长度。",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="在长文档切分成片段时，片段之间的跨越长度。",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="问题的最大token数量，超过该长度将被截断。",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="如果使用uncased模型，则需要设置此项。"
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="每个GPU/CPU的评估批次大小。"
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="如果为真，将打印所有与数据处理相关的警告。",
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument("--threads", type=int, default=1, help="用于将示例转换为特征的线程数")
    parser.add_argument("--no_cuda", action="store_true", help="在可用时不使用CUDA")
    parser.add_argument("--seed", type=int, default=42, help="初始化的随机种子。")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # 设置随机种子
    set_seed(args.seed)
    print(args.model_name_or_path)
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        use_fast=False,
    )
    print(args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    # 开始评估
    results = evaluate(args, model, tokenizer)

    current_date = datetime.now().strftime("%Y%m%d%H%M")
    print("Results: {}".format(results))
    results["model_path"] = args.model_name_or_path
    results["evalDataPaths"] = args.predict_file
    results["evalTimes"] = current_date

    file_name = f"eval_{current_date}.json"
    file_path = os.path.join(args.model_name_or_path, file_name)

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    return results

if __name__ == "__main__":
    main()
