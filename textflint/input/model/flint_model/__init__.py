"""
Model Wrappers
--------------------------
textflint can attack any model that takes a list of strings as input and outputs a list of predictions.
This is the idea behind *model flint_model*: to help your model conform to this API, we've provided the
``textflint.models.flint_model.ModelWrapper`` abstract class.


We've also provided implementations of model flint_model for common patterns in some popular
machine learning frameworks:

"""

from .flint_model import FlintModel

