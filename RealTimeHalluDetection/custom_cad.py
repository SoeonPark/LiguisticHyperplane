r"""
Legacy custom_cad.py is kept below as inert text because the previous file mixed
import-time evaluation setup, distributed decoding, and CLI execution in one
path. The runnable implementation starts after this raw string.

import argparse
import logging
import os

import datasets
import torch

import transformers
import accelerate
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
)

import numpy as np
from termcolor import colored
import json
from accelerate import InitProcessGroupKwargs
import datetime


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# CAD evaluate functions
import json
import argparse
from tqdm import tqdm
from pathlib import Path
# from datasets import load_dataset
# from evaluate import load
import statistics
import json
from collections import defaultdict
import os
import evaluate
from ipdb import set_trace as bp
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# evaluate fackKB: Put your huggingface access tokens
access_token = 
tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2, use_auth_token=access_token)

def evaluate_qa(index2ex, eval_file):
    print(eval_file)
    all_gold = []
    all_pred = []
    all_doc = []
    all_fact_score = []

    if os.path.exists(eval_file) == False:
        return 0
    with open(eval_file, "r") as f:
        output_data = [json.loads(line) for line in f]
    cov_em_all = []
    category2em = defaultdict(list)
    id2ex_output = {}
    for i, output in enumerate(output_data):
        index = output["input_index"]
        pred = output["string"][0]
        gold = index2ex[index]["gold_answers"] 
        if len(pred) < 3:
            print(pred)
            continue
        all_gold.append(gold)
        all_pred.append(pred)
        if len(pred) < 3:
            print(f"pred: {pred}")

        article = index2ex[index]["article"]
        summary = pred
        input = [[summary, article]]
        tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True)
        result = torch.softmax(factkb(**tokens).logits, dim = 1)
        # bp()
        fact_score = result[0][1].item()

        all_fact_score.append(fact_score)
        all_doc.append(article)
        output_dict = index2ex[index].copy()
        output_dict["pred"] = pred
        id2ex_output[i] = output_dict

    print("fact_score: ", statistics.mean(all_fact_score))
    # print(statistics.mean(cov_em_all))
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=all_pred, references=all_gold)
    print("rouge results: ", results)

    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=all_pred, references=all_doc, lang="en")
    # print("bertscore: ", results)
    print("bertscore: ")
    for k, v in results.items():
        if k in ["precision", "recall", "f1"]:
            print(f"{k}: {statistics.mean(v)}")
    return id2ex_output

# read data
def entity_data(dataset_path):
    raw_data = []
    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex["assigned_process"] == 0:
                raw_data.append(ex)
            # break
        # raw_data = json.loads(f.read())
    return raw_data


if __name__ == "__main__":
    # args parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/cnndm_example_input/cnndm_1_0.jsonl")
    parser.add_argument("--pred_path", type=str, default="./data/cnndm_example_input/cnndm_1.5_-0.5.jsonl.output_topp0.9_genlen100.jsonl")
    args = parser.parse_args()

    data_path = args.data_path
    pred_path = args.pred_path
    index2ex = entity_data(data_path)
    evaluate_qa(index2ex, pred_path)
    

# CAD functually related functions
def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, torch.finfo(logits.dtype).min)
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return (2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value)


def filter_logits_top_p(logits, top_p, negative_multiplier=False):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    if negative_multiplier:
        filtered_logits = logits.masked_fill(valid_indices == 0, 1000)
    else:
        filtered_logits = logits.masked_fill(valid_indices == 0, -1000)
    return filtered_logits


def decode(args, batch_input_ids, dec_depth, model, tokenizer):
    batch_size = args.per_device_eval_batch_size
    assert batch_input_ids.size(1) == args.context_size
    assert args.decode_truncate_len >= 0
    assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
    else:
        raise ValueError("context cannot be none")
    history_decode_ids = None

    past_key_values = None # necessary for causal models
    if args.model_category == 'seq2seq':
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            batch_input_ids[:, :args.context_size].clone(), dict(), None
        ) # this step includes encoding the context
        history_decode_ids = model._prepare_decoder_input_ids_for_generation(
            batch_input_ids.size(0),
            model_kwargs=model_kwargs,
            device=batch_input_ids.device,
        ) # create placeholder starter seq for decoding
    else:
        model_kwargs = None

    for _i in range(dec_depth):
        if args.model_category == 'causal':
            model_inputs = model.prepare_inputs_for_generation(unit_context_input_ids, past_key_values=past_key_values)
            outputs = model(**model_inputs, output_hidden_states=False)
        elif args.model_category == 'seq2seq':
            model_inputs = model.prepare_inputs_for_generation(history_decode_ids, **model_kwargs) # this incorporates past_key_values
            outputs = model(**model_inputs, output_hidden_states=False)
        else:
            raise ValueError("model category not supported")

        score = outputs.logits[:, -1:, :].clone().contiguous()

        if args.assigned_weight >= 0:
            score = filter_logits_top_p(score, top_p=args.filter_top_p)
        else:
            score = filter_logits_top_p(score, top_p=args.filter_top_p_prior, negative_multiplier=True)

        score = args.assigned_weight * score
        torch.distributed.all_reduce(score, group=args.gathering_group)

        projected_logits = logits_sampling_projection(score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)

        if not args.accelerator.is_main_process:
            projected_logits = torch.zeros_like(projected_logits)
        torch.distributed.all_reduce(projected_logits, group=args.gathering_group)

        simplex = torch.nn.functional.softmax(projected_logits, dim=-1)
        real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)

        if args.model_category == 'causal':
            unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1) # not really necessary but keeping

        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

        if args.model_category == 'causal':
            past_key_values = outputs.past_key_values
        elif args.model_category == 'seq2seq':
            model_kwargs["past_key_values"] = outputs.past_key_values

        # HACK: stop when seeing eos token, but asserting batch size is 1, unit_seq_len is 1, optimize later
        assert real_token_ids_list.size(0) == 1
        assert real_token_ids_list.size(1) == 1
        if real_token_ids_list[0][-1] == model.generation_config.eos_token_id:
            break

    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))#, skip_special_tokens=True)
    else:
        init_context_input_ids = None
        context_sequences = None
    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'), skip_special_tokens=True)
    logger.info(f"context: {context_sequences}")
    logger.info(f"sampled: {colored(str(sampled_sequences), 'red')}")

    return history_decode_ids, init_context_input_ids, None, sampled_sequences, context_sequences, None


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument("--init_blank_language_model", action="store_true", help="Whether or not to use a completely blank LM.")
    parser.add_argument(
        "--file_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--train_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--decode_truncate_len", type=int, default=50, help="",
    ) # how many to cut from right
    parser.add_argument(
        "--decode_depth", type=int, default=2, help="",
    )
    parser.add_argument(
        "--projection_top_p", type=float, default=0.2, help="",
    )
    parser.add_argument(
        "--filter_top_p", type=float, default=1.0, help="",
    )
    parser.add_argument(
        "--filter_top_p_prior", type=float, default=1.0, help="",
    )
    parser.add_argument("--big_model_inference", type=str, default="no")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=259200))])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        # set_seed(args.seed)
        accelerate.utils.set_seed(args.seed, device_specific=True) # differ slightly for each device

    if accelerator.is_main_process:
        pass
        # if args.output_dir is not None:
        #     os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.train_mode == "decode":
        if len(args.model_name_or_path.split('|')) > 0:
            main_model_name = args.model_name_or_path.split('|')[0]
            fallback_model_name = args.model_name_or_path.split('|')[1]
            args.model_name_or_path = main_model_name
            args.orig_model_name_or_path = fallback_model_name
        else:
            args.orig_model_name_or_path = args.model_name_or_path
    else:
        raise ValueError("training should be in a separate file (irrelevant in context-aware decoding)")

    # Han: assign ensemble models
    args.file_mode = args.file_mode.split('|')
    assert args.file_mode[0] == "fin"
    assert os.path.exists(args.file_mode[1])
    fin_path = args.file_mode[1]
    fin_data = []
    with open(fin_path, 'r', encoding='utf-8') as f:
        for line in f:
            proc_line = line.strip()
            if proc_line:
                fin_data.append(json.loads(proc_line))
    rank2model = dict()
    for _fd in fin_data:
        if _fd['assigned_process'] in rank2model: # sanity check
            assert ' '.join(rank2model[_fd['assigned_process']]) == ' '.join(_fd['assigned_model'].split('|'))
        else:
            rank2model[_fd['assigned_process']] = _fd['assigned_model'].split('|') 

    # Han: add gathering group
    default_backend = torch.distributed.get_backend(torch.distributed.distributed_c10d._get_default_group())
    args.gathering_group = torch.distributed.new_group(ranks=list(sorted(rank2model.keys())), backend=default_backend)

    if accelerator.process_index not in rank2model.keys(): # Han: exit if not in the ensemble
        return
    args.model_name_or_path = rank2model[accelerator.process_index][0]

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        if 'llama' in args.model_name_or_path.lower():
            from transformers import LlamaConfig
            config = LlamaConfig.from_pretrained(args.model_name_or_path)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if 'neox' in args.model_name_or_path.lower(): # Han: gpt-neox doesn't have a slow tokenizer, use GPTNeoXTokenizerFast
        from transformers import GPTNeoXTokenizerFast
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.model_name_or_path)
    elif 'llama' in args.model_name_or_path.lower():
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        assert args.use_slow_tokenizer == True 
        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

    if args.init_blank_language_model:
        raise ValueError("disabled")
        model = AutoModelForMaskedLM.from_config(config)
    elif args.model_name_or_path:
        if 't5' in args.model_name_or_path.lower() or 'tk' in args.model_name_or_path.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                ignore_mismatched_sizes=False,
                torch_dtype=torch.float16,
            )
            args.model_category = 'seq2seq'
            model = model.to(accelerator.device)
        else:
            if 'llama' in args.model_name_or_path.lower(): # llama special case
                from transformers import LlamaForCausalLM
                if args.big_model_inference == 'no':
                    model = LlamaForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        torch_dtype=torch.float16,
                    )
                    model = model.to(accelerator.device)
                else:
                    # Han: we assume 8 GPUs
                    if accelerator.process_index == 0:
                        local_devices = [0, 2, 4, 6]
                    elif accelerator.process_index == 1:
                        local_devices = [1, 3, 5, 7]
                    else:
                        raise ValueError("check accelerator.process_index")
                    # this is architecture specific
                    my_device_map = {'model.embed_tokens': local_devices[0],
                                    'lm_head': local_devices[0],
                                    'model.norm': local_devices[0]}
                    for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                        for layer_idx in layer_idx_list:
                            my_device_map[f'model.layers.{layer_idx}'] = local_devices[_device_i]
                    model = LlamaForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        device_map=my_device_map,
                        torch_dtype=torch.float16,
                    )
            elif args.big_model_inference == 'no':
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    torch_dtype=torch.float16, 
                )
                model = model.to(accelerator.device)
            elif args.big_model_inference == 'yes' and 'opt' in args.model_name_or_path.lower():
                # Han: we assume 8 GPUs
                if accelerator.process_index == 0:
                    local_devices = [0, 2, 4, 6]
                elif accelerator.process_index == 1:
                    local_devices = [1, 3, 5, 7]
                else:
                    raise ValueError("check accelerator.process_index")
                # this is architecture specific
                my_device_map = {'model.decoder.embed_tokens': local_devices[0],
                                'lm_head': local_devices[0],
                                'model.decoder.embed_positions': local_devices[0],
                                'model.decoder.final_layer_norm': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'model.decoder.layers.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            elif args.big_model_inference == 'yes' and 'neox' in args.model_name_or_path.lower():
                # Han: we assume 8 GPUs
                if accelerator.process_index == 0:
                    local_devices = [0, 2, 4, 6]
                elif accelerator.process_index == 1:
                    local_devices = [1, 3, 5, 7]
                else:
                    raise ValueError("check accelerator.process_index")
                # this is architecture specific
                my_device_map = {'gpt_neox.embed_in': local_devices[0],
                                'embed_out': local_devices[0],
                                'gpt_neox.final_layer_norm': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'gpt_neox.layers.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            elif args.big_model_inference == 'yes' and 'neo' in args.model_name_or_path.lower():
                # Han: we assume 8 GPUs
                if accelerator.process_index == 0:
                    local_devices = [0, 2, 4, 6]
                elif accelerator.process_index == 1:
                    local_devices = [1, 3, 5, 7]
                else:
                    raise ValueError("check accelerator.process_index")
                # this is architecture specific
                my_device_map = {'transformer.wte': local_devices[0],
                                'lm_head': local_devices[0],
                                'transformer.wpe': local_devices[0],
                                'transformer.drop': local_devices[0],
                                'transformer.ln_f': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'transformer.h.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            else:
                raise ValueError("check args.big_model_inference")

            args.model_category = 'causal'
        model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward) # referred to https://github.com/huggingface/accelerate/blob/38fd30e764ea87ef86e7d69fcba559c3605925b1/src/accelerate/accelerator.py#L1138
        model.forward = accelerate.utils.convert_outputs_to_fp32(model.forward)
    else:
        raise ValueError("specify --init_blank_language_model")

    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"model size: {sum(p.numel() for p in model.parameters())}")
    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)
    one_hot_value = 5.0 # unused

    ##########################################

    # change the output file name later
    out_json_fn = f"{fin_path}.output_topp{args.projection_top_p}_genlen{args.decode_depth}.jsonl"
    if accelerator.is_main_process:
        with open(out_json_fn, 'w') as f:
            f.write('placeholder, program not finished ...\n')

    args.tokenizer = tokenizer

    if args.train_mode == "decode":
        model.eval()

        args.one_hot_value = one_hot_value
        args.vocab_size = vocab_size
        args.hidden_size = hidden_size
        args.accelerator = accelerator

        export_list = []
        args.orig_decode_truncate_len = args.decode_truncate_len
        with torch.no_grad():
            for _fd in fin_data: # only support batch size 1 for now since the context size can be different across lines
                if _fd['assigned_process'] != args.accelerator.process_index: # remember to unblock barriers before this line
                    continue
                args.assigned_weight = _fd['assigned_weight']

                ctx_field_name = 'context_string'
                assert ctx_field_name in _fd
                assert args.per_device_eval_batch_size == 1

                input_ids = torch.LongTensor(tokenizer.encode(_fd[ctx_field_name], add_special_tokens=True)).unsqueeze(0).to(args.accelerator.device)
                args.context_size = input_ids.size(1)
                args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size # Han: this compensates for the unknown input context size

                if 'filter_p' in _fd: # token filtering
                    args.filter_top_p = _fd['filter_p']
                if 'filter_p_prior' in _fd:
                    args.filter_top_p_prior = _fd['filter_p_prior']

                if args.decode_truncate_len < 0:
                    continue # skipping very long examples
                logger.info(f"idx: {_fd['input_index']}")

                repeat_sample = 1 # change here manually if necessary
                for _r in range(repeat_sample):
                    history_decode_ids, _, _, sampled_sequences, _, _ = \
                        decode(args, input_ids, args.decode_depth, model, tokenizer)
                    if _r == 0: # first sample
                        # export to jsonl
                        for _i in range(args.per_device_eval_batch_size):
                            export_dict = dict()
                            export_dict['tokens'] = [history_decode_ids.tolist()[_i]]
                            export_dict['string'] = [sampled_sequences[_i]]
                            export_dict['assigned_process'] = _fd['assigned_process']
                            export_dict['assigned_model'] = args.model_name_or_path
                            export_dict['output_index'] = len(export_list)
                            export_dict['input_index'] = _fd['input_index']
                            export_list.append(export_dict)
                    else:
                        for _i in range(args.per_device_eval_batch_size):
                            export_list[-(args.per_device_eval_batch_size - _i)]['tokens'].append(history_decode_ids.tolist()[_i])
                            export_list[-(args.per_device_eval_batch_size - _i)]['string'].append(sampled_sequences[_i])

        if accelerator.is_main_process:
            if os.path.exists(out_json_fn):
                os.remove(out_json_fn)
                logger.info(f"Cleaning existing {out_json_fn}")
            with open(out_json_fn, mode="w") as f_out: # use mode 'a' if several processes are writing to the same file
                for export in export_list:
                    f_out.write(json.dumps(export))
                    f_out.write("\n")


if __name__ == "__main__":
    main()
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm


@dataclass
class CADGenerationResult:
    text: str
    token_ids: List[int]
    elapsed_sec: float
    per_token_ms: float
    num_prompt_branches: int


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved JSONL -> {path}")


def group_by_input_index(records: Sequence[Dict]) -> Dict[int, List[Dict]]:
    """
    Grouped the records by input index, and sorted the records in each group by assigned process (if any).
    1st Data: {"input_index": 0, "assigned_process": 0, "string": "pred1"}
        When the context is given, prompt to put for Expert model. (e.g., CNN Article + "Summarize the article in three sentences. Summary:")
    2nd Data: {"input_index": 0, "assigned_process": 1, "string": "pred2"}
        When the context is not given, prompt to put for Amateur model. (e.g., Only has "Summarize the article in three sentences. Summary:")
    """
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["input_index"])].append(record)
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item.get("assigned_process", 0)))
    return dict(sorted(grouped.items()))


def entity_data(dataset_path: str, assigned_process: int = 0) -> Dict[int, Dict]:
    examples = {}
    for ex in read_jsonl(dataset_path):
        if int(ex.get("assigned_process", assigned_process)) == assigned_process:
            examples[int(ex["input_index"])] = ex
    return examples


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        keep = logits.argmax(dim=-1, keepdim=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(-1, keep, False)
        return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove = torch.zeros_like(sorted_remove)
    remove.scatter_(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, torch.finfo(logits.dtype).min)


def filter_logits_top_p(
    logits: torch.Tensor,
    top_p: float,
    negative_multiplier: bool = False,
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError("logits must have shape (batch, seq, vocab)")
    if top_p is None or top_p >= 1.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat(
        [nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]],
        dim=-1,
    )
    valid_indices = nucleus.scatter(2, indices, nucleus)
    fill_value = 1000.0 if negative_multiplier else -1000.0
    return logits.masked_fill(valid_indices == 0, fill_value)


def logits_sampling_projection(
    logits: torch.Tensor,
    top_p: float,
    one_hot_value: float,
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError("logits must have shape (batch, seq, vocab)")
    filtered_logits = filter_logits_top_p(logits, top_p=top_p)
    selected = torch.distributions.Categorical(logits=filtered_logits).sample()
    return 2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value


def sample_next_token(
    logits: torch.Tensor,
    top_p: float = 0.9,
    top_k: int = 0,
    temperature: float = 1.0,
    do_sample: bool = True,
) -> torch.Tensor:
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if temperature is not None and temperature > 0:
        logits = logits / temperature
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)
    if do_sample:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return logits.argmax(dim=-1, keepdim=True)


class CAD:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        num_gpus: str = "1",
        max_gpu_memory: int = 27,
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.torch_dtype = torch_dtype
        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {}
        if self.device == "cuda":
            dtype = torch.float16 if self.torch_dtype in {"auto", "float16", "fp16"} else torch.float32
            kwargs["torch_dtype"] = dtype
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                n_gpus = int(self.num_gpus)
                if n_gpus != 1:
                    kwargs["device_map"] = "auto"
                    kwargs["max_memory"] = {i: f"{self.max_gpu_memory}GiB" for i in range(n_gpus)}
        elif self.device != "cpu":
            raise ValueError("device must be 'cuda' or 'cpu'")

        tokenizer_name = model_name if "vicuna" not in model_name else "huggyllama/llama-7b"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **kwargs,
        )
        if self.device == "cuda" and self.num_gpus == "1":
            model.cuda()
        elif self.device == "cpu":
            model.cpu()
        model.eval()
        return model, tokenizer

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        top_p: float = 0.9,
        top_k: int = 0,
        temperature: float = 1.0,
        do_sample: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> CADGenerationResult:
        # breakpoint()
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        start = time.perf_counter()
        with torch.no_grad():
            # breakpoint()
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                **kwargs,
            )
            # breakpoint()
        elapsed = time.perf_counter() - start
        gen_ids = outputs.sequences[0, input_ids.shape[-1]:].detach().cpu().tolist()
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # breakpoint()
        if verbose:
            print(f"MODEL OUTPUT:\n{text}")
        return CADGenerationResult(
            text=text,
            token_ids=[int(x) for x in gen_ids],
            elapsed_sec=float(elapsed),
            per_token_ms=float(1000.0 * elapsed / max(len(gen_ids), 1)),
            num_prompt_branches=1,
        )

    def generate_weighted(
        self,
        input_texts: Sequence[str],
        weights: Sequence[float],
        max_new_tokens: int = 100,
        filter_top_p: float = 1.0,
        filter_top_p_prior: float = 1.0,
        projection_top_p: float = 0.9,
        top_k: int = 0,
        temperature: float = 1.0,
        do_sample: bool = True,
        verbose: bool = True,
    ) -> CADGenerationResult:
        if len(input_texts) != len(weights):
            raise ValueError("input_texts and weights must have the same length")
        if not input_texts:
            raise ValueError("at least one input branch is required")

        device = next(self.model.parameters()).device
        branch_ids = [
            self.tokenizer(text, return_tensors="pt").input_ids.to(device)
            for text in input_texts
        ]
        generated: List[int] = []
        eos_id = self.tokenizer.eos_token_id

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                combined = None
                for ids, weight in zip(branch_ids, weights):
                    logits = self.model(input_ids=ids, use_cache=False).logits[:, -1:, :].contiguous() # (1, 1, vocab)
                    """
                    Logits for the last token in the sequence, shape (1, 1, vocab_size)
                    (Pdb) logits.shape
                    torch.Size([1, 1, 32000])
                    """
                    # breakpoint()
                    if weight >= 0:
                        logits = filter_logits_top_p(logits, top_p=filter_top_p)
                    else:
                        logits = filter_logits_top_p(
                            logits,
                            top_p=filter_top_p_prior,
                            negative_multiplier=True,
                        )
                    weighted = float(weight) * logits
                    combined = weighted if combined is None else combined + weighted
                # breakpoint()
                next_token = sample_next_token(
                    combined,
                    top_p=projection_top_p,
                    top_k=top_k,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                # breakpoint()
                """
                (Pdb) next_token
                tensor([[903]], device='cuda:0')
                (Pdb) next_token.shape
                torch.Size([1, 1])
                """
                token_id = int(next_token.item())
                generated.append(token_id)
                next_token = next_token.to(device)
                branch_ids = [torch.cat([ids, next_token], dim=1) for ids in branch_ids]
                if eos_id is not None and token_id == eos_id:
                    break
        # breakpoint()
        elapsed = time.perf_counter() - start
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        # breakpoint()
        if verbose:
            print(f"MODEL OUTPUT:\n{text}")
        return CADGenerationResult(
            text=text,
            token_ids=generated,
            elapsed_sec=float(elapsed),
            per_token_ms=float(1000.0 * elapsed / max(len(generated), 1)),
            num_prompt_branches=len(input_texts),
        )

    def generate_dataset(
        self,
        data_path: str,
        output_path: str,
        max_new_tokens: int = 100,
        projection_top_p: float = 0.9,
        filter_top_p: float = 1.0,
        filter_top_p_prior: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        temperature: float = 1.0,
        do_sample: bool = True,
        limit: Optional[int] = None,
        print_every: int = 20,
    ) -> str:
        records = read_jsonl(data_path)
        grouped = group_by_input_index(records) # dict type
        items = list(grouped.items()) # list type
        # breakpoint()
        """
        (Pdb) type(grouped)
        <class 'dict'>
        (Pdb) type(records)
        <class 'list'>
        (Pdb) type(items)
        <class 'list'>
        600
        (Pdb) len(grouped)
        300
        (Pdb) len(grouped[0][0])
        8
        (Pdb) grouped[0][1].keys()
        dict_keys(['input_index', 'assigned_model', 'assigned_process', 'context_string', 'assigned_weight', 'gold_answers', 'filter_p', 'article'])
        (Pdb) grouped[0][1].keys()
        dict_keys(['input_index', 'assigned_model', 'assigned_process', 'context_string', 'assigned_weight', 'gold_answers', 'article', 'filter_p'])
        """
        if limit is not None:
            items = items[:limit]

        outputs = []
        for output_index, (input_index, group) in enumerate(tqdm(items, desc="CAD generation")):
            # breakpoint()
            contexts = [row["context_string"] for row in group]
            weights = [float(row.get("assigned_weight", 1.0)) for row in group]

            if len(contexts) == 1 and abs(weights[0] - 1.0) < 1e-9:
                result = self.generate(
                    contexts[0],
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    do_sample=do_sample,
                    verbose=False,
                )
                # breakpoint()
            else:
                result = self.generate_weighted(
                    contexts,
                    weights,
                    max_new_tokens=max_new_tokens,
                    filter_top_p=filter_top_p,
                    filter_top_p_prior=filter_top_p_prior,
                    projection_top_p=projection_top_p,
                    top_k=top_k,
                    temperature=temperature,
                    do_sample=do_sample,
                    verbose=False,
                )
                # breakpoint()

            """(Pdb) result
            CADGenerationResult(text="Dzhokhar Tsarnaev has been convicted of killing three people in the Boston Marathon bombing two years ago. 
            What is the time period the story covers? From April 2013 to April 2015 What are some interesting details in the article? 
            Boston Marathon Victims Remembered on 'One Boston Day' . Read the article The victims of the Boston Marathon bombing. 
            I’ve attached a video about the Boston Marathon. Watch this", 
            token_ids=[360, 17599, 554, 8222, 19089, 11441, 5750, 756, 1063, 7602, 18186, 310, 23393, 2211, 2305, 297, 278, 12115, 
            1085, 25206, 13585, 292, 1023, 2440, 8020, 29889, 1724, 338, 278, 931, 3785, 278, 5828, 18469, 29973, 3645, 3786, 29871, 
            29906, 29900, 29896, 29941, 304, 3786, 29871, 29906, 29900, 29896, 29945, 1724, 526, 777, 8031, 4902, 297, 278, 4274, 29973, 
            12115, 1085, 25206, 7229, 9893, 22738, 287, 373, 525, 6716, 12115, 8373, 29915, 869, 7523, 278, 4274, 450, 6879, 9893, 310, 
            278, 12115, 1085, 25206, 13585, 292, 29889, 306, 30010, 345, 10959, 263, 4863, 1048, 278, 12115, 1085, 25206, 29889, 24274, 445], 
            elapsed_sec=22.01120653981343, per_token_ms=220.1120653981343, num_prompt_branches=2)
            """
            if print_every > 0 and output_index % print_every == 0:
                print(
                    f"[{output_index}] input_index={input_index} "
                    f"branches={len(contexts)} weights={weights} "
                    f"tok_ms={result.per_token_ms:.2f}"
                )
                print(result.text[:300].replace("\n", "\\n"))
            """
            [0] input_index=0 branches=2 weights=[1.0, 0.0] tok_ms=220.11
            Dzhokhar Tsarnaev has been convicted of killing three people in the Boston Marathon bombing two years ago. 
            What is the time period the story covers? From April 2013 to April 2015 What are some interesting details in the article? 
            Boston Marathon Victims Remembered on 'One Boston Day' . Read the artic
            """

            first = group[0]
            outputs.append({
                "tokens": [result.token_ids],
                "string": [result.text],
                "assigned_process": first.get("assigned_process", 0),
                "assigned_model": self.model_name,
                "assigned_weights": weights,
                "assigned_processes": [row.get("assigned_process", i) for i, row in enumerate(group)],
                "output_index": output_index,
                "input_index": int(input_index),
                "elapsed_sec": result.elapsed_sec,
                "per_token_ms": result.per_token_ms,
            })
            # breakpoint()
            """
            (Pdb) first.keys()
            dict_keys(['input_index', 'assigned_model', 'assigned_process', 'context_string', 'assigned_weight', 'gold_answers', 'filter_p', 'article'])
            (Pdb) outputs
            [{'tokens': [[450, 12115, 1085, 25206, 13585, 292, 338, 263, 1407, 14610, 322, 25305, 293, 1741, 393, 9559, 297, 12115, 29892, 16167, 373, 3786, 29871, 29896, 29945, 29892, 29871, 29906, 29900, 29896, 29941, 29889, 739, 471, 2309, 491, 1023, 1757, 29892, 323, 4183, 6468, 322, 360, 17599, 554, 8222, 19089, 11441, 5750, 29892, 1058, 892, 515, 6561, 305, 1460, 29874, 29889, 910, 766, 29887, 504, 292, 322, 5192, 2222, 1044, 15201, 17202, 310, 2305, 297, 12115, 322, 278, 18830, 4038, 29889, 2567, 393, 278, 14260, 338, 975, 29892, 372, 338, 701, 304, 278, 11099, 943, 304, 11097, 278, 6035, 18310, 363, 360]], 
            'string': ['The Boston Marathon bombing is a very sad and tragic event that happened in Boston, Massachusetts on April 15, 2013. It was done by two men, Tamerlan and Dzhokhar Tsarnaev, who were from Chechnya. This disgusting and heartless act affected thousands of people in Boston and the surrounding area. Now that the trial is over, it is up to the jurors to decide the punishment for D'], 'assigned_process': 0, 'assigned_model': 'huggyllama/llama-7b', 'assigned_weights': [1.0, 0.0], 'assigned_processes': [0, 1], 'output_index': 0, 'input_index': 0, 
            'elapsed_sec': 22.014898031949997, 'per_token_ms': 220.14898031949997}]
            """

        write_jsonl(outputs, output_path)
        return output_path


def load_factkb(device: str = "cpu", hf_token: Optional[str] = None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base",
        padding="max_length",
        truncation=True,
    )
    kwargs = {"num_labels": 2}
    if token:
        kwargs["token"] = token
    model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", use_safetensors=True, **kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def factkb_score(
    factkb_tokenizer,
    factkb_model,
    summary: str,
    article: str,
    device: str = "cpu",
) -> float:
    # breakpoint()
    tokens = factkb_tokenizer(
        [[summary, article]],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    # breakpoint()
    with torch.no_grad():
        result = torch.softmax(factkb_model(**tokens).logits, dim=1)
    return float(result[0][1].item())


def evaluate_qa(
    index2ex: Dict[int, Dict],
    eval_file: str,
    use_rouge: bool = True,
    use_bertscore: bool = True,
    use_factkb: bool = True,
    factkb_device: str = "cpu",
    hf_token: Optional[str] = None,
) -> Dict:
    # breakpoint()
    print(f"Evaluating: {eval_file}")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(eval_file)

    # breakpoint()
    output_data = read_jsonl(eval_file)
    all_gold: List[str] = []
    all_pred: List[str] = []
    all_doc: List[str] = []
    all_fact_score: List[float] = []
    id2ex_output = {}
    # breakpoint()
    """
    (Pdb) output_data[0]
    {'tokens': [[1551, 23168, 1862, 30010, 8373, 445, 1629, 10503, 440, 943, 310, 278, 12115, 1085, 25206, 13585, 292, 2996, 4208, 304, 6456, 278, 12080, 5714, 322, 1906, 1058, 892, 28606, 1023, 2440, 8020, 29889, 3115, 29892, 360, 17599, 554, 8222, 19089, 11441, 5750, 674, 2317, 14260, 2446, 4723, 304, 1074, 565, 540, 20586, 278, 4892, 27368, 363, 278, 29871, 29941, 29900, 21090, 4475, 304, 278, 13585, 886, 29889, 1724, 6297, 1258, 278, 1634, 272, 2153, 1708, 297, 445, 4274, 29973, 2688, 15593, 287, 10503, 440, 943, 310, 278, 13585, 292, 322, 1906, 1058, 8496, 30010, 29873, 14111, 396, 29933, 11253, 12742]], 'string': ['On Patriots’ Day this year survivors of the Boston Marathon bombing came together to remember the lives lost and those who were injured two years ago. Also, Dzhokhar Tsarnaev will stand trial next week to see if he receives the death penalty for the 30 charges related to the bombings. What role did the reporters play in this article? They interviewed survivors of the bombing and those who couldn’t observe #BostonDay'], 'assigned_process': 0, 'assigned_model': 'huggyllama/llama-7b', 'assigned_weights': [1.5, -0.5], 'assigned_processes': [0, 1], 'output_index': 0, 'input_index': 0, 'elapsed_sec': 21.795718614012003, 'per_token_ms': 217.95718614012003}
    (Pdb) len(output_data)
    20
    """

    factkb_tokenizer = None
    factkb_model = None
    if use_factkb:
        factkb_tokenizer, factkb_model = load_factkb(factkb_device, hf_token=hf_token)

    for i, output in enumerate(output_data):
        index = int(output["input_index"])
        if index not in index2ex:
            print(f"[WARN] missing input_index={index} in references")
            continue
        pred = output.get("string", [""])[0]
        if len(pred.strip()) < 3:
            print(f"[WARN] short prediction for input_index={index}: {pred!r}")
            continue

        # breakpoint()
        ex = index2ex[index]
        gold = ex.get("gold_answers", "")
        article = ex.get("article", ex.get("context_string", ""))

        # breakpoint()
        """
        (Pdb) ex.keys()
        dict_keys(['input_index', 'assigned_model', 'assigned_process', 'context_string', 'assigned_weight', 'gold_answers', 'filter_p', 'article'])
        """
        all_gold.append(gold)
        all_pred.append(pred)
        all_doc.append(article)
        # breakpoint()
        output_dict = dict(ex)
        output_dict["pred"] = pred
        output_dict["output_index"] = output.get("output_index", i)
        id2ex_output[index] = output_dict

        if use_factkb and factkb_tokenizer is not None and factkb_model is not None:
            all_fact_score.append(
                factkb_score(factkb_tokenizer, factkb_model, pred, article, device=factkb_device)
            )

    # breakpoint()
    metrics = {
        "num_predictions": len(all_pred),
        "factkb": statistics.mean(all_fact_score) if all_fact_score else None,
        "rouge": None,
        "bertscore": None,
    }

    if not all_pred:
        print("[WARN] no valid predictions to evaluate")
        return {"metrics": metrics, "examples": id2ex_output}

    if use_factkb and all_fact_score:
        print(f"fact_score: {metrics['factkb']:.6f}")

    if use_rouge:
        import evaluate

        rouge = evaluate.load("rouge")
        metrics["rouge"] = rouge.compute(predictions=all_pred, references=all_gold)
        print("rouge results:", metrics["rouge"])

    if use_bertscore:
        import evaluate

        bertscore = evaluate.load("bertscore")
        result = bertscore.compute(predictions=all_pred, references=all_doc, lang="en")
        metrics["bertscore"] = {
            key: statistics.mean(value)
            for key, value in result.items()
            if key in {"precision", "recall", "f1"}
        }
        print("bertscore:")
        for key, value in metrics["bertscore"].items():
            print(f"{key}: {value:.6f}")

    return {"metrics": metrics, "examples": id2ex_output}


def default_output_path(data_path: str, projection_top_p: float, max_new_tokens: int) -> str:
    return f"{data_path}.output_topp{projection_top_p}_genlen{max_new_tokens}.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Run custom CAD generation/evaluation.")
    parser.add_argument("--mode", choices=["generate", "evaluate", "generate_and_evaluate"], default="generate")
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max-gpu-memory", type=int, default=27)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./data/cnndm_example_input/cnndm_1_0.jsonl")
    parser.add_argument("--pred-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--projection_top_p", type=float, default=0.9)
    parser.add_argument("--filter_top_p", type=float, default=1.0)
    parser.add_argument("--filter_top_p_prior", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--eval-assigned-process", type=int, default=0)
    parser.add_argument("--skip-rouge", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--factkb", action="store_true")
    parser.add_argument("--factkb-device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--hf-token", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_path = args.pred_path

    if args.mode in {"generate", "generate_and_evaluate"}:
        output_path = args.output_path or default_output_path(
            args.data_path,
            args.projection_top_p,
            args.max_new_tokens,
        )
        cad = CAD(
            args.model_name,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
        )
        pred_path = cad.generate_dataset(
            data_path=args.data_path,
            output_path=output_path,
            max_new_tokens=args.max_new_tokens,
            projection_top_p=args.projection_top_p,
            filter_top_p=args.filter_top_p,
            filter_top_p_prior=args.filter_top_p_prior,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            do_sample=not args.greedy,
            limit=args.limit,
            print_every=args.print_every,
        )

    if args.mode in {"evaluate", "generate_and_evaluate"}:
        if pred_path is None:
            raise ValueError("--pred-path is required for evaluate mode")
        refs = entity_data(args.data_path, assigned_process=args.eval_assigned_process)
        payload = evaluate_qa(
            refs,
            pred_path,
            use_rouge=not args.skip_rouge,
            use_bertscore=not args.skip_bertscore,
            use_factkb=args.factkb,
            factkb_device=args.factkb_device,
            hf_token=args.hf_token,
        )
        metric_path = str(Path(pred_path).with_suffix(Path(pred_path).suffix + ".metrics.json"))
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(payload["metrics"], f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Saved metrics -> {metric_path}")


if __name__ == "__main__":
    main()
