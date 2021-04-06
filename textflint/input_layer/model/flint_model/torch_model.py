"""
PyTorch Model Wrapper
--------------------------
"""


import torch
from torch.nn import CrossEntropyLoss


from .flint_model import FlintModel
from ....common.utils import device


class TorchModel(FlintModel):
    r"""
    Loads a PyTorch model (`nn.Module`) and tokenizer.

    """
    def __init__(
        self,
        model,
        tokenizer,
        task='SA',
        batch_size=32
    ):
        """

        :param torch.nn.Module model: target PyTorch model
        :param tokenizer: tokenizer whose output can be packed as a tensor and
            passed to the model. No type requirement, but most have `tokenizer`
            method that accepts list of strings.
        :param str task: task name
        :param int batch_size:  batch size to apply evaluation
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        super().__init__(model, tokenizer, task=task, batch_size=batch_size)
        self.model = model.to(device)

    def __call__(self, *inputs):
        raise NotImplementedError()

    def get_model_grad(self, text_inputs, loss_fn=CrossEntropyLoss()):
        r"""
        Get gradient of loss with respect to input tokens.

        :param str|[str] text_inputs: input string or input string list
        :param torch.nn.Module loss_fn: loss function.
            Default is `torch.nn.CrossEntropyLoss`
        :return: Dict of ids, tokens, and gradient as numpy array.

        """
        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` "
                f"that returns `torch.nn.Embedding` object that represents "
                f"input embedding layer"
            )

        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be of type `torch.nn.Module`.")

        self.model.train()

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)
        self.model.zero_grad()
        model_device = next(self.model.parameters()).device

        inputs_ids = self.encode(text_inputs)
        ids = [torch.tensor(ids).to(model_device) for ids in inputs_ids]

        predictions = self.model(text_inputs)

        output = predictions.argmax(dim=1)
        loss = loss_fn(predictions, output)
        loss.backward()

        # grad w.r.t to word embeddings
        grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad}

        return output

    def encode(self, inputs):
        r"""
        Tokenize inputs and convert it to ids.

        :param inputs: model original input
        :return: list of inputs ids

        """

        raise NotImplementedError()
