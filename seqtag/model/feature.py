import os
import io
import json
import copy
import torch
import logging
import torch.nn as nn
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier

logger = logging.getLogger(__name__)


class FeatConfig:

    def __init__(
            self,
            vocab_size=30522,  # we use Bert Vocab
            embed_size=256,
            hidden_size=256,
            num_hidden_layers=1,
            hidden_dropout_prob=0.1,
            initializer_range=0.02,
            pad_token_id=0,
            postag_embed_size=32,
            postag_vocab_size=50,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.postag_embed_size = postag_embed_size
        self.postag_vocab_size = postag_vocab_size

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def __repr__(self):
        json_string = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        return "{} {}".format(self.__class__.__name__, json_string)

    def save_pretrained(self, save_directory):
        """
        Save a configuration object to the directory `save_directory`, so that it
        can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.
        Args:
            save_directory (:obj:`string`):
                Directory where the configuration JSON file will be saved.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, "config.json")

        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        json_config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.isfile(json_config_file):
            with open(json_config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            config_dict = json.loads(text)
        else:
            config_dict = {}
        config = cls(**config_dict)
        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        return config


class JointFeatModel(nn.Module):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointFeatModel, self).__init__()
        self.config = config
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.embed_size,
            padding_idx=config.pad_token_id
        )

        if args.use_postag:
            self.postag_embeddings = nn.Embedding(
            config.postag_vocab_size,
            config.postag_embed_size,
            padding_idx=config.pad_token_id
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if args.use_postag:
            self.intent_classifier = IntentClassifier(
                config.embed_size + config.postag_embed_size, self.num_intent_labels, args.dropout_rate
            )
            self.slot_classifier = SlotClassifier(
                config.embed_size + config.postag_embed_size, self.num_slot_labels, args.dropout_rate
            )
        else:
            self.intent_classifier = IntentClassifier(
                config.embed_size, self.num_intent_labels, args.dropout_rate
            )
            self.slot_classifier = SlotClassifier(
                config.embed_size, self.num_slot_labels, args.dropout_rate
            )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.initializer_range = config.initializer_range
        self.apply(self._init_weights)

    def load_pretrained_vectors(self, emb_index, vocab):
        emb_size = self.word_embeddings.weight.size(1)
        pretrained = torch.empty(len(vocab), emb_size, dtype=torch.float).normal_(
            mean=0.0, std=self.initializer_range
        )
        for w, idx in vocab.items():
            if w in emb_index:
                pretrained[idx] = torch.tensor(emb_index[w], dtype=torch.float)
        self.word_embeddings.weight.data.copy_(pretrained)

    def freeze_word_embeddings(self):
        self.word_embeddings.weight.requires_grad = False

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, intent_label_ids, slot_labels_ids, postag_ids):
        embeddings = self.word_embeddings(input_ids)
        if self.args.use_postag:
            postag_embeddings = self.postag_embeddings(postag_ids)
            pooled_output = torch.cat((embeddings[:, 1:], postag_embeddings[:, 1:]), dim=-1).mean(dim=1) # ignoring [CLS]
            joint_embeddings = torch.cat((embeddings, postag_embeddings), dim=-1)
        else:
            # pooled_output = embeddings[:, 0]  # [CLS]
            pooled_output = embeddings[:, 1:].mean(dim=1)  # ignoring [CLS]
            joint_embeddings = embeddings

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(joint_embeddings)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),)
        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits # Logits is a tuple of intent and slot logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = cls(**kwargs)
        model_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.isfile(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
            model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
            Arguments:
                save_directory: directory to which to save.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")

        model_to_save.config.save_pretrained(save_directory)
        torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))
