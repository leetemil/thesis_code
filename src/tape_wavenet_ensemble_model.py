import torch
import torch.nn as nn

from tape import ProteinModel, ProteinConfig
from tape.models.modeling_utils import SequenceClassificationHead, SequenceToSequenceClassificationHead, ValuePredictionHead, PairwiseContactPredictionHead
from tape.registry import registry

from scipy import stats
from models import WaveNet

class WaveNetConfig(ProteinConfig):
    def __init__(self,
        input_channels: int = 30,
        residual_channels: int = 48,
        out_channels: int = 30,
        stacks: int = 6,
        layers_per_stack: int = 9,
        total_samples: int = 0, # I think we can make do by just setting this to 0
        l2_lambda : int = 0,
        bias : bool = True,
        dropout : float = 0.5,
        bayesian : bool = False,
        **kwargs):

        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.out_channels = out_channels
        self.stacks = stacks
        self.layers_per_stack = layers_per_stack
        self.total_samples = total_samples
        self.l2_lambda = l2_lambda
        self.bias = bias
        self.dropout = dropout
        self.bayesian = bayesian
        self.initializer_range = 0.02 # stolen from unirep

class WaveNetAbstractModel(ProteinModel):
    config_class = WaveNetConfig
    base_model_prefix = 'wavenet_ensemble'

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@registry.register_task_model('embed', 'wavenet_ensemble')
class WaveNetModel(WaveNetAbstractModel):
    # init expects only a single argument - the config
    def __init__(self, config: WaveNetConfig):
        super().__init__(config)
        self.models = nn.ModuleList()

        for i in range(3):
            model = WaveNet(
                config.input_channels,
                config.residual_channels,
                config.out_channels,
                config.stacks,
                config.layers_per_stack,
                config.total_samples,
                config.l2_lambda,
                config.bias,
                config.dropout,
                config.bayesian,
                backwards = False
            )
            self.models.append(model)

        for i in range(3):
            model = WaveNet(
                config.input_channels,
                config.residual_channels,
                config.out_channels,
                config.stacks,
                config.layers_per_stack,
                config.total_samples,
                config.l2_lambda,
                config.bias,
                config.dropout,
                config.bayesian,
                backwards = True
            )
            self.models.append(model)

        self.init_weights()

    def forward(self, input_ids, input_mask = None):
        output, pooled = zip(*[m.get_representation(input_ids) for m in self.models])
        output = torch.cat(output, dim = 2)
        pooled = torch.cat(pooled, dim = 1)
        return output, pooled

@registry.register_task_model('fluorescence', 'wavenet_ensemble')
@registry.register_task_model('stability', 'wavenet_ensemble')
class WaveNetForValuePrediction(WaveNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.wavenet = WaveNetModel(config)
        predict_size = 6 * config.residual_channels
        self.predict = ValuePredictionHead(predict_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):
        sequence_output, pooled_output = self.wavenet(input_ids, input_mask=input_mask)
        prediction, *_ = self.predict(pooled_output)
        outputs = (prediction,)

        if targets is not None:
            loss = nn.MSELoss()(prediction, targets)
            metrics = {'spearman_rho': stats.spearmanr(prediction.cpu().detach(), targets.cpu().detach()).correlation}
            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction

@registry.register_task_model('remote_homology', 'wavenet_ensemble')
class WaveNetForSequenceClassification(WaveNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.wavenet = WaveNetModel(config)
        predict_size = 6 * config.residual_channels
        self.classify = SequenceClassificationHead(predict_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        sequence_output, pooled_output = self.wavenet(input_ids, input_mask=input_mask)
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

@registry.register_task_model('secondary_structure', 'wavenet_ensemble')
class WaveNetForSequenceToSequenceClassification(WaveNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.wavenet = WaveNetModel(config)
        predict_size = 6 * config.residual_channels
        self.classify = SequenceToSequenceClassificationHead(predict_size, config.num_labels, ignore_index=-1)
        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        sequence_output, pooled_output = self.wavenet(input_ids, input_mask=input_mask)
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

@registry.register_task_model('contact_prediction', 'wavenet_ensemble')
class WaveNetForContactPrediction(WaveNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.wavenet = WaveNetModel(config)
        predict_size = 6 * config.residual_channels
        self.predict = PairwiseContactPredictionHead(predict_size, ignore_index=-1)
        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):
        outputs = self.wavenet(input_ids, input_mask=input_mask)
        sequence_output = outputs[0]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
