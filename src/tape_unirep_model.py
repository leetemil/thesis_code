import torch
import torch.nn as nn
from tape import ProteinModel, ProteinConfig
from tape.models.modeling_utils import SequenceClassificationHead, SequenceToSequenceClassificationHead, ValuePredictionHead, PairwiseContactPredictionHead
from tape.registry import registry

from scipy import stats

from models import UniRep

class UniRepReimpConfig(ProteinConfig):
    def __init__(self, num_tokens = 30, padding_idx = 0, embed_size: int = 10, hidden_size: int = 512, num_layers: int = 1, representation = "mean", train_inner = True, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.padding_idx = padding_idx
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.initializer_range = 0.02
        self.representation = representation
        self.train_inner = train_inner

class UniRepReimpAbstractModel(ProteinModel):
    config_class = UniRepReimpConfig
    base_model_prefix = 'unirep_reimp'

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@registry.register_task_model('embed', 'unirep_reimp')
class UniRepReimpModel(UniRepReimpAbstractModel):
    # init expects only a single argument - the config
    def __init__(self, config: UniRepReimpConfig):
        super().__init__(config)
        self.inner_model = UniRep(config.num_tokens, config.padding_idx, config.embed_size, config.hidden_size, config.num_layers)
        self.representaton = config.representation
        self.train_inner = config.train_inner

        self.init_weights()

    def forward(self, input_ids, input_mask = None):
        if self.train_inner:
            out, lengths, state = self.inner_model.run_rnn(input_ids)
        else:
            with torch.no_grad():
                self.inner_model.eval()
                out, lengths, state = self.inner_model.run_rnn(input_ids)

        if self.representaton == "mean":
            representations = out.sum(1) / lengths.unsqueeze(1)
        elif self.representaton == "last":
            representations = state[0].squeeze()
        else:
            raise ValueError(f"Unrecognized representation type {self.representation}.")

        return (out, representations)

@registry.register_task_model('fluorescence', 'unirep_reimp')
@registry.register_task_model('stability', 'unirep_reimp')
class UniRepReimpForValuePrediction(UniRepReimpAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep_reimp = UniRepReimpModel(config)

        predict_size = config.hidden_size
        self.predict = ValuePredictionHead(predict_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):
        outputs = self.unirep_reimp(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # outputs = self.predict(pooled_output, targets) + outputs[2:]
        # # (loss), prediction_scores, (hidden_states)
        # return outputs

        prediction, *_ = self.predict(pooled_output)

        outputs = (prediction,)

        if targets is not None:
            loss = nn.MSELoss()(prediction, targets)
            metrics = {'spearman_rho': stats.spearmanr(prediction.cpu().detach(), targets.cpu().detach()).correlation}

            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction

@registry.register_task_model('remote_homology', 'unirep_reimp')
class UniRepReimpForSequenceClassification(UniRepReimpAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep_reimp = UniRepReimpModel(config)

        predict_size = config.hidden_size

        self.classify = SequenceClassificationHead(predict_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep_reimp(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # outputs = self.classify(pooled_output, targets) + outputs[2:]
        # # (loss), prediction_scores, (hidden_states)
        # return outputs

        prediction, *_ = self.classify(pooled_output)

        outputs = (prediction,)

        if targets is not None:
            loss = nn.CrossEntropyLoss()(prediction, targets)
            is_correct = prediction.float().argmax(-1) == targets
            is_valid_position = targets != -1

            # cast to float b/c otherwise torch does integer division
            num_correct = torch.sum(is_correct * is_valid_position).float()
            accuracy = num_correct / torch.sum(is_valid_position).float()
            metrics = {'acc': accuracy}

            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction

@registry.register_task_model('secondary_structure', 'unirep_reimp')
class UniRepReimpForSequenceToSequenceClassification(UniRepReimpAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep_reimp = UniRepReimpModel(config)
        self.classify = SequenceToSequenceClassificationHead(config.hidden_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep_reimp(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # outputs = self.classify(sequence_output, targets) + outputs[2:]
        # # (loss), prediction_scores, (hidden_states)
        # return outputs

        prediction, *_ = self.classify(sequence_output)

        outputs = (prediction,)

        if targets is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-1)(prediction.view(-1, prediction.size(2)), targets.view(-1))
            # cast to float b/c float16 does not have argmax support
            is_correct = prediction.float().argmax(-1) == targets
            is_valid_position = targets != -1

            # cast to float b/c otherwise torch does integer division
            num_correct = torch.sum(is_correct * is_valid_position).float()
            accuracy = num_correct / torch.sum(is_valid_position).float()
            metrics = {'acc': accuracy}

            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction

@registry.register_task_model('contact_prediction', 'unirep_reimp')
class UniRepReimpForContactPrediction(UniRepReimpAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep_reimp = UniRepReimpModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.unirep_reimp(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
