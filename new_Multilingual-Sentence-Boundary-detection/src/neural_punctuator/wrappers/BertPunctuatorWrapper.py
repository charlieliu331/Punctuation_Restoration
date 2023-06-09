from neural_punctuator.base.BaseWrapper import BaseWrapper
from neural_punctuator.models.BertPunctuator import BertPunctuator
from neural_punctuator.preprocessors.BertPreprocessor import BertPreprocessor
from neural_punctuator.trainers.BertPunctuatorTrainer import BertPunctuatorTrainer


class BertPunctuatorWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)

        self._config = config
        self._preprocessor = BertPreprocessor(config)
        self._classifier = BertPunctuator(config)
        self._trainer = BertPunctuatorTrainer(self._classifier, self._preprocessor, self._config)

    def train(self):
        self._trainer.train()

    def validate(self):
        self._trainer.val()
        
    def predict(self):
        raise NotImplementedError
