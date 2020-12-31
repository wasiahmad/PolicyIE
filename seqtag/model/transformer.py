import os
import json
import copy
import torch
import logging
import torch.nn as nn
from torchcrf import CRF
from transformers.tokenization_bert import BertTokenizer
from .module import IntentClassifier, SlotClassifier

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "transformer": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "transformer": 512,
}
PRETRAINED_INIT_CONFIGURATION = {
    "transformer": {"do_lower_case": True},
}


class TransformerTokenizer(BertTokenizer):
    r"""
    Constructs a  TransformerTokenizer.
    :class:`~TransformerTokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.
    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION


class TransformerConfig:

    def __init__(
            self,
            vocab_size=30522,  # we use Bert Vocab
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        return output

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

    def __repr__(self):
        json_string = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        return "{} {}".format(self.__class__.__name__, json_string)

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


class JointTransformer(nn.Module):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointTransformer, self).__init__()
        self.config = config
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act
        )
        encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.model = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=config.num_hidden_layers,
            norm=encoder_norm
        )

        self.intent_classifier = IntentClassifier(
            config.hidden_size, self.num_intent_labels, args.dropout_rate
        )
        self.slot_classifier = SlotClassifier(
            config.hidden_size, self.num_slot_labels, args.dropout_rate
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.initializer_range = config.initializer_range
        self.apply(self._init_weights)

    def freeze_word_embeddings(self):
        self.word_embeddings.weight.requires_grad = False

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, intent_label_ids, slot_labels_ids):
        input_shape = input_ids.size()
        device = input_ids.device
        position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        sequence_output = self.model(
            embeddings.transpose(0, 1),  # n, bsz, emb_dim
            src_key_padding_mask=~attention_mask.bool()
        ).transpose(0, 1)
        pooled_output = sequence_output[:, 0]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

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
