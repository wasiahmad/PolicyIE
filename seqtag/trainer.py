import os
import io
import json
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from seqtag.utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, tokenizer=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        emb_index = {}
        kwargs = {}
        if args.model_type in ['rnn-emb', 'feature']:
            kwargs['vocab_size'] = tokenizer.vocab_size
            if self.args.embed_file:
                fin = io.open(self.args.embed_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
                n, d = map(int, fin.readline().split())
                for line in fin:
                    tokens = line.rstrip().split(' ')
                    emb_index[tokens[0]] = list(map(float, tokens[1:]))
                kwargs['embed_size'] = d

        self.config = self.config_class.from_pretrained(
            args.model_name_or_path,
            **kwargs
        )
        self.model = self.model_class.from_pretrained(
            args.model_name_or_path,
            config=self.config,
            args=args,
            intent_label_lst=self.intent_label_lst,
            slot_label_lst=self.slot_label_lst
        )

        if args.model_type in ['rnn-emb', 'feature'] and emb_index:
            self.model.load_pretrained_vectors(emb_index, tokenizer.get_vocab())

        params = list(self.model.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        logger.info("Total model #parameters: {}".format(total_params))

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)
        self.args.n_gpu = torch.cuda.device_count()

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        # multi-gpu training
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Train!
        logger.info("***** Running training *****")
        logger.info(" Num examples = %d", len(self.train_dataset))
        logger.info(" Num Epochs = %d", self.args.num_train_epochs)
        logger.info(" Total train batch size = %d", self.args.train_batch_size)
        logger.info(" Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info(" Total optimization steps = %d", t_total)
        logger.info(" Logging steps = %d", self.args.logging_steps)
        logger.info(" Save steps = %d", self.args.save_steps)

        global_step = 0
        best_score = 0.0
        patience = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        accum_loss, loss_count = 0, 0

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'intent_label_ids': batch[3],
                    'slot_labels_ids': batch[4]
                }

                if self.args.model_type in ['bert', 'albert']:
                    inputs['token_type_ids'] = batch[2]

                if self.args.model_type in ['feature']:
                    inputs['postag_ids'] = batch[5]

                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                accum_loss += loss.item() * batch[0].size(0)
                loss_count += batch[0].size(0)
                log_info = 'Iteration [%.3f]' % (accum_loss / loss_count)
                epoch_iterator.set_description("%s" % log_info)

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        logger.info(" lr = %f", scheduler.get_lr()[0])
                        logger.info(" loss = %.3f", (tr_loss - logging_loss) / self.args.logging_steps)
                        logging_loss = tr_loss

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        result = self.evaluate("valid")
                        valid_metric = "slot_f1"
                        if result[valid_metric] > best_score:
                            logger.info("result['{}']={} > best_score={}".format(
                                valid_metric, result[valid_metric], best_score))
                            best_score = result[valid_metric]
                            self.save_model()
                            logger.info("Reset patience to 0")
                            patience = 0
                        else:
                            patience += 1
                            logger.info("Hit patience={}".format(patience))
                            if self.args.eval_patience > 0 and patience > self.args.eval_patience:
                                logger.info("early stop! patience={}".format(patience))
                                epoch_iterator.close()
                                train_iterator.close()
                                return global_step, tr_loss / global_step

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'valid':
            dataset = self.dev_dataset
        else:
            raise Exception("Only valid and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # multi-gpu eval
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'intent_label_ids': batch[3],
                    'slot_labels_ids': batch[4]
                }

                if self.args.model_type in ['bert', 'albert']:
                    inputs['token_type_ids'] = batch[2]

                if self.args.model_type in ['feature']:
                    inputs['postag_ids'] = batch[5]

                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                if self.args.n_gpu > 1:
                    # mean() to average on multi-gpu parallel evaluating
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                out_slot_labels_ids = np.append(
                    out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        # we ignore 'Other' and 'UNK'labeled intent examples during exact match computation
        # ignore_intent_label = None
        ignore_intent_label = [
            self.intent_label_lst.index('UNK'),
            self.intent_label_lst.index('Other')
        ]

        total_result = compute_metrics(
            intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list,
            ignore_intent_label=ignore_intent_label
        )
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        if mode == 'test':
            # Write to output file
            pred_file = os.path.join(self.args.model_dir, 'predictions.txt')
            ref_file = os.path.join(self.args.model_dir, 'references.txt')
            with open(pred_file, "w", encoding="utf-8") as f1, open(ref_file, "w", encoding="utf-8") as f2:
                for slot_preds, intent_pred, slot_gold, intent_gold in \
                        zip(slot_preds_list, intent_preds, out_slot_label_list, out_intent_label_ids):
                    pred_line = ""
                    for pred in slot_preds:
                        pred_line += pred + " "
                    f1.write("{}\t{}\n".format(self.intent_label_lst[intent_pred], pred_line.strip()))
                    ref_line = ""
                    for gold in slot_gold:
                        ref_line += gold + " "
                    f2.write("{}\t{}\n".format(self.intent_label_lst[intent_gold], ref_line.strip()))

            eval_outfile = os.path.join(self.args.model_dir, 'eval_results.txt')
            with open(eval_outfile, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, sort_keys=True)

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.config = self.config_class.from_pretrained(
                self.args.model_dir
            )
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          config=self.config,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
