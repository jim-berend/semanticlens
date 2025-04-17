import abc

from torch.nn import Module


class VisionLanguageFoundationModel(Module, abc.ABC):
    @abc.abstractmethod
    def encode_vision(self, *args, **kwargs): ...

    @abc.abstractmethod
    def encode_text(self, *args, **kwargs): ...


class VisionLanguageProcessor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *, images, text, **kwargs): ...
