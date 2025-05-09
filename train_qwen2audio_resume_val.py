"""Code for finetune_qwen2audio"""
import os
os.environ['WANDB_DISABLE_CODE'] = 'true'
os.environ["WANDB_API_KEY"]='c90fecd56f771e43bd9fb69a0909cc7187b40f50'#guorui账号log
import json
import torch
import logging
import argparse
import re
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import wandb
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup, GenerationConfig
import shutil
import json
from qwen2audio_basic.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
from qwen2audio_basic.processing_qwen2_audio import Qwen2AudioProcessor
from collections import defaultdict
import librosa
from io import BytesIO
from urllib.request import urlopen
from functools import partial
import requests
# from transformers.pipelines.audio_utils import ffmpeg_read # apt install ffmpeg   firstly
# import torchaudio

os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs


class SFT_dataset(Dataset):
    def __init__(self, args, processor):
        self.args = args
        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor

        self.ignore_index = -100
        self.sep = '\n'
        self.roles = ['User:', 'Assistant:']
        self.ignore_len = len(self.tokenizer.encode(self.sep + self.roles[1], add_special_tokens=False))
        self.debug = True

        with open(args.train_data_path, 'r', encoding='utf-8') as f:
            self.data_dict = json.load(f)
        # with open(args.train_audio_dirs,'r') as f:
        #     self.audio_dirs_dict=json.load(f)

        # choose data source
        sources = list(
            self.data_dict.keys())  # [1:] if len(list(self.data_dict.keys()))>1 else list(self.data_dict.keys())
        # data source
        self.data = [self.preprocess([item['prompt'], item['asr_label']]) for source in sources for item in
                     self.data_dict[source]]

        # audios_path=[os.path.join(self.audio_dirs_dict[source],item['audio_name']) for source in sources for item in self.data_dict[source]]
        audios_path = [item['path'] for source in sources for item in self.data_dict[source]]
        # random.shuffle(audios_path)
        # audios_path=[os.path.join(args.train_audio_dir,audio) for audio in audio_names if audio.endswith('.wav')]

        self.sr = self.feature_extractor.sampling_rate
        self.audios = [librosa.load(audio_path, sr=self.sr)[0] for audio_path in audios_path if
                       audio_path.endswith('.wav')]

        # rank=int(os.getenv('RANK', -1))
        # if rank==0 or rank== -1:
        #     print(audio_inputs.shape)

    def __getitem__(self, index):
        self.data[index].update({'audios': self.audios[index]})
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def preprocess(self, data):
        input_ids = []
        labels = []
        attention_mask = []
        if not isinstance(data, list):
            raise ValueError('The data must be a list.')
        for ind, d in enumerate(data):
            if ind % 2 == 1:
                inputs = self.tokenizer(self.sep + self.roles[1] + d, add_special_tokens=False,
                                        max_length=self.args.max_seq_len, truncation=True)
                value_ids = inputs['input_ids'] if isinstance(inputs['input_ids'], list) else inputs[
                    'input_ids'].tolist()
                input_ids += value_ids
                attention_mask += inputs['attention_mask'] if isinstance(inputs['attention_mask'], list) else inputs[
                    'attention_mask'].tolist()
                labels += [self.ignore_index] * self.ignore_len + value_ids[self.ignore_len:]
                if len(labels) >= self.args.max_seq_len:
                    break
            else:
                pre_str = self.sep if len(input_ids) > 0 else ''
                inputs = self.tokenizer(pre_str + self.roles[0] + d, add_special_tokens=False,
                                        max_length=self.args.max_seq_len, truncation=True)
                value_ids = inputs['input_ids'] if isinstance(inputs['input_ids'], list) else inputs[
                    'input_ids'].tolist()
                attention_mask += inputs['attention_mask'] if isinstance(inputs['attention_mask'], list) else inputs[
                    'attention_mask'].tolist()
                input_ids += value_ids
                if len(labels) > 0:
                    labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * (len(value_ids) - 1)
                else:
                    labels += [self.ignore_index] * len(value_ids)
        input_ids.append(self.tokenizer.eos_token_id)
        attention_mask.append(1)
        labels.append(self.tokenizer.eos_token_id)

        return {'input_ids': input_ids[:self.args.max_seq_len],
                'attention_mask': attention_mask[:self.args.max_seq_len], 'labels': labels[:self.args.max_seq_len]}

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        audios = [item['audios'] for item in batch]

        audio_inputs = self.feature_extractor(audios, sampling_rate=self.sr, return_attention_mask=True)
        input_features = audio_inputs['input_features'] if isinstance(audio_inputs['input_features'], list) else \
        audio_inputs['input_features'].tolist()
        feature_attention_mask = audio_inputs['attention_mask'] if isinstance(audio_inputs['attention_mask'], list) else \
        audio_inputs['attention_mask'].tolist()

        if self.debug and dist.get_rank() == 0:
            sample_list = self.tokenizer.decode(batch[0]['input_ids']).split(self.tokenizer.eos_token)
            for sample in sample_list:
                print('\n*****************************************************')
                print(sample + self.tokenizer.eos_token)
            self.debug = False

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            'attention_mask': torch.LongTensor(attention_mask),
            'input_features': torch.FloatTensor(input_features),
            'feature_attention_mask': torch.Tensor(feature_attention_mask)
        }

# class TestDataset(Dataset):
#     def __init__(self, data_path, tokenizer):
#         self.data = []
#         with open(data_path) as f:
#             self.data = json.load(f)
#         dist_flag_0 = True if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0) else False
#         if dist_flag_0:
#             print(f'load {len(self.data)} data from {data_path}')
#         self.tokenizer = tokenizer
#         self.debug = True

#     def __getitem__(self, index):
#         item = self.data[index]
#         return {
#             'data': item,
#             'input': item['question']
#         }

#     def __len__(self):
#         return len(self.data)
    
#     def collate_fn(self, batch):
#         batch_query = [x['input'] for x in batch]
#         batch_data = [x['data'] for x in batch]
#         out_batch = {}
#         out_batch['data'] = batch_data
#         out_batch['input_ids'] = self.tokenizer(batch_query, return_tensors='pt', padding=True)['input_ids']
#         out_batch['attention_mask']=self.tokenizer(batch_query, return_tensors='pt', padding=True)['attention_mask']
#         dist_flag_0 = True if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0) else False
#         if self.debug and dist_flag_0:
#             decoded_texts = self.tokenizer.batch_decode(out_batch['input_ids'], skip_special_tokens=False)
#             print(f'*******************batch_texts[0]**********************************')
#             print(decoded_texts[0])
#             self.debug = False
#         return out_batch
    

# def get_response(batch_input_ids, batch_output_ids, tokenizer, num_return):
#     responses_list=[]
#     batch_return=[]
#     input_len = len(batch_input_ids[0])
#     for idx, output_ids in enumerate(batch_output_ids):
#         generated_ids = output_ids[input_len:]
#         batch_return.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
#         if idx % num_return == num_return-1:
#             responses_list.append(batch_return)
#             batch_return=[]
#     return responses_list


# def extract_and_choose_answer(pattern, model_answer):
#     matches = re.findall(pattern, model_answer)
#     option_count = {}
#     for match in matches:
#         option_count[match.upper()] = option_count.get(match.upper(), 0) + 1

#     if not option_count:
#         loose_pattern = r'[A-F]'
#         if pattern == loose_pattern:
#             return None
#         else:
#             return extract_and_choose_answer(loose_pattern, model_answer) 
        
#     max_count = max(option_count.values())
#     max_options = [option for option, count in option_count.items() if count == max_count]
#     return max_options[0]
    
    
    
# def generate_score(generate_datas):
#     all = defaultdict(int)
#     right = defaultdict(int)
#     accuracy_dict = defaultdict(int)
    
#     print(f'**** Eval Total:{len(generate_datas)}****')
#     for item in generate_datas:
#         source = item["source"]
#         for answer in item["model_answer"]:
#             all[source] += 1  
#             pattern = r'[（\(]([A-Fa-f])[）\)]'
#             extract_answer = extract_and_choose_answer(pattern, answer)
#             if item['answer'] == extract_answer:
#                 right[source] += 1
                
#     print(f'****eval_all:{all}****')
#     print(f'****eval_right:{right}****')        
                
#     for key in right:
#         accuracy_dict[key] = right[key] / all[key]
            
#     return accuracy_dict
    

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss
    

def train(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps) 

    if accelerator.is_main_process:
        wandb.init(project = args.experiment_name, config=args, dir=args.log_dir)
    
    accelerator.print(f'args:\n{args}')
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu*dist.get_world_size()*accelerator.gradient_accumulation_steps
    processor = Qwen2AudioProcessor.from_pretrained(args.model_path ,trust_remote_code=True)
    left_tokenizer = processor.tokenizer#,pad_token='<|endoftext|>', eos_token='<|endoftext|>'
    # feature_extractor=processor.feature_extractor

    if left_tokenizer.pad_token is None:
        left_tokenizer.pad_token=left_tokenizer.eos_token
    
    if args.checkpoint_path:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(os.path.join(args.checkpoint_path, "tfmr"), trust_remote_code=True)
    else:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path, trust_remote_code=True)

    config=model.config
    accelerator.print(f'config:\n{config}')
    accelerator.print(config.audio_config.encoder_layers)

    #freeze AudioEncoder, Projector
    if args.freeze:
        for name,params in model.named_parameters():
            if 'multi_modal_projector' in name or 'audio_tower' in name:
                params.requires_grad =False
            # if 'audio_tower' in name:
            #     params.requires_grad =False


    if args.gradient_checkpointing and not args.freeze:
        model.gradient_checkpointing_enable()


    def trainable_parameters(model):
        """
        Checks the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return trainable_params, all_param
    
    trainable_params, all_param=trainable_parameters(model)
    accelerator.print(f"trainable params: {trainable_params}({trainable_params/1e9:.4f}B) || all params: {all_param}({all_param/1e9:.4f}B) || trainable%: {100 * trainable_params / all_param}")
    

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = SFT_dataset(args, processor)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    # dev_dataset = TestDataset(args.dev_data_dir, left_tokenizer)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_bsz_per_gpu, shuffle=False, drop_last=False, collate_fn=dev_dataset.collate_fn)
    generation_config = GenerationConfig.from_pretrained(args.model_path, pad_token_id=left_tokenizer.pad_token_id, num_return_sequences=1, max_new_tokens=256, min_new_tokens=2, do_sample=False, temperature=1.0, top_k=50, top_p=1.0)
            
    num_training_steps = int(len(train_dataloader) * (args.n_epochs + 0.35) ) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} train_data_dir:{args.train_data_path} lr:{args.learning_rate} num_training_steps:{num_training_steps}')
    
    # model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(model, optimizer, train_dataloader, dev_dataloader)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    if args.checkpoint_path:
        # accelerator.load_state(args.checkpoint_path)
        if os.path.isfile(os.path.join(args.checkpoint_path, "training_state.pt")):
            training_state = torch.load(os.path.join(args.checkpoint_path, "training_state.pt"))
            start_epoch = training_state["epoch"]
            start_step = training_state["step"]+1
            global_step = training_state["global_step"]
            accelerator.print(f"Loaded at {start_epoch} epoch, {start_step} step and {global_step} global step")
        else:
            raise ValueError(f"training_state.pt not found at: {args.checkpoint_path}")
    else:
        start_epoch = 0
        start_step = 0
        global_step = 0

    if args.save_step <= 0:
        args.save_step=max(len(train_dataloader)//1,1)
        accelerator.print(f'Save step setted to {args.save_step}')
    # if args.eval_step <= 0:
    #     args.eval_step=len(train_dataloader) // 10
    #     accelerator.print(f'Eval step setted to {args.eval_step}')

    metric = SFTMetric(device=torch.cuda.current_device())

    #Code for saving checkpoints
    def save_checkpoint(epoch, step, global_step):
        #check ckpt nums and delete the oldest
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        output_dir = os.path.join(save_dir, 'tfmr')
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
            if args.max_ckpts > 1:
                if num_checkpoints >= args.max_ckpts:
                    checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                    oldest_checkpoint = checkpoint_files[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint))        
            os.makedirs(save_dir, exist_ok=True)
        # save 16-bit model
        if accelerator.state.deepspeed_plugin.zero_stage==3:
            unwrap_model = accelerator.unwrap_model(model)         
            unwrap_model.save_pretrained(output_dir,is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(model))
        else:
            if accelerator.is_main_process:
                model.save_pretrained(output_dir,state_dict=accelerator.get_state_dict(model))
        if accelerator.is_main_process:
            left_tokenizer.save_pretrained(output_dir)
            copy_files = []
            for item in os.listdir(args.model_path):
                if os.path.exists(os.path.join(output_dir,item)):
                    continue
                if item.startswith("pytorch_model") and item.endswith(".bin"):
                    continue
                if item.endswith(".index.json") or item.endswith(".safetensors"):
                    continue
                s = os.path.join(args.model_path, item)
                if os.path.isfile(s):
                    shutil.copy(s, os.path.join(output_dir,item))
                copy_files.append(item)
            print(f'huggingface model save in {output_dir}, copy file:{copy_files}')

        accelerator.wait_for_everyone()
        # accelerator.save_state(save_dir)
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')

    # accelerator.print(accelerator.deepspeed_config)
    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:

            if (epoch==start_epoch and batch_cnt<start_step) or epoch<start_epoch:
                if (batch_cnt+1) % accelerator.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids=batch['input_ids']
            labels=batch['labels']
            attention_mask=batch['attention_mask']
            input_features=batch['input_features']
            feature_attention_mask=batch['feature_attention_mask']

            # rank=int(os.getenv('RANK', -1))
            # if rank==0 or rank== -1:
            #     print(input_ids.shape,input_ids.dtype)
            #     print(labels.shape,labels.dtype)
            #     print(attention_mask.shape,attention_mask.dtype)
            #     print(input_features.shape,input_features.dtype)
            #     print(feature_attention_mask.shape,feature_attention_mask.dtype)
            
            output = model(input_ids=input_ids,labels=labels,attention_mask=attention_mask,
                           input_features=input_features,feature_attention_mask=feature_attention_mask,
                           return_dict=True,use_cache=False)
            loss = output.loss

            metric(output.logits, output.labels, loss)
            acc, train_loss = metric.get_metric()
            accelerator.backward(loss)
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

            # if global_step % args.eval_step == 0:
            #     torch.cuda.empty_cache()
            #     model.eval() 
            #     generate_datas = []
            #     dataloader_iterator = tqdm(dev_dataloader, total=len(dev_dataloader)) if accelerator.is_main_process else dev_dataloader

            #     for batch in dataloader_iterator:
            #         batch_input_ids = batch["input_ids"]
            #         batch_data = batch["data"]
            #         batch_attention_mask=batch['attention_mask']
            #         batch_output_ids = accelerator.unwrap_model(model).generate(batch_input_ids, attention_mask=batch_attention_mask,generation_config=generation_config)
            #         batch_responses = get_response(batch_input_ids, batch_output_ids, left_tokenizer, args.num_return)
                    
            #         for data, responses in zip(batch_data, batch_responses):
            #             answer_list = []
            #             for response in responses:
            #                 answer_list.append(response)
            #             data['model_answer'] = answer_list
            #         generate_datas.extend(batch_data)
                    
            #     all_gpu_data =  [None] * dist.get_world_size()
            #     dist.all_gather_object(all_gpu_data, generate_datas)
            #     all_data = [item for sublist in all_gpu_data for item in sublist]
            #     if accelerator.is_main_process:
            #         score_dict = generate_score(all_data)
            #         wandb.log(score_dict, step=global_step)

            #     model.train()           
            
            if global_step % args.save_step == 0:
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, batch_cnt, global_step)
                
        accelerator.wait_for_everyone()
        save_checkpoint(epoch, batch_cnt, global_step)
        start_step = 0
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Experiment Args
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--checkpoint_path',default=None, type=str)

    # Model Args
    parser.add_argument('--model_path', default='', type=str)

    # Data Args
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--train_audio_dirs', default='', type=str)
    # parser.add_argument('--dev_data_dir', default='', type=str)
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=1, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    
    # Training Args
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=1, type=int)
    parser.add_argument('--eval_bsz_per_gpu', default=4, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=2, type=int)
    parser.add_argument('--quantization_bit', default=None, type=float)

    # Eval Args
    parser.add_argument('--num_return', default=1, type=int)
    
    # Other Args
    parser.add_argument('--save_step', default=-1, type=int)
    # parser.add_argument('--eval_step', default=-1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_seq_len', default=4096, type=int)
    parser.add_argument('--freeze', action='store_true')

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)


    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # print('torch.cuda.is_available():',torch.cuda.is_available())

    set_seed(args.seed)
    train(args)