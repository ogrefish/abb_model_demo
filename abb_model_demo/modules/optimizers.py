"""
In this land are found modules that provide the optimizing
steps. These are currently only used for the neural net
gradient descent routines.

This design implementation is currently inheritance oriented.
The optimization modules are intended to be components of a
pipeline (see pipelines.py). Could/should this be refactored to
be more composition-oriented? It seems simple to maintain given
the current scope.
"""

import torch


class AbbNnOptimizerBase():
    """
    Base class for all neural net optimizer modules. Provides
    some default functionality for batch_start, batch_step and
    epoch_step.
    """

    def __init__(self, hyperparam_dict):
        """Initializes an base-optimizer module.

        Calls the `verify_hyperparams` function that must be implemented by
        daughter classes.

        Args:
            hyperparam_dict (dict): A dictionary containing hyperparameter values.

        Attributes:
            hyperparam_dict (dict): The original hyperparameter dictionary.
            optimizer (obj): The optimizer used by the model. (default: None)
            scheduler (obj): The scheduler used by the model. (optional, default: None)
        """
        self.hyperparam_dict = hyperparam_dict
        self.optimizer = None  # set in derived class
        self.scheduler = None  # (optionally) set in derived class

        self.verify_hyperparams()


    def verify_hyperparams(self):
        raise NotImplementedError("AbbNnOptimizerBase.verify_hyperparams")


    def batch_start(self):
        """
        Default start-of-batch behavior.
        Calls optimizer.zero_grad()
        """
        self.optimizer.zero_grad()


    def batch_step(self):
        """
        Default batch step behavior.
        Calls optimizer.step()
        """
        self.optimizer.step()


    def epoch_step(self):
        """
        Default epoch step behavior.
        Calls scheduler.step() if scheduler exists.
        """
        if self.scheduler is not None:
            self.scheduler.step()


class AbbNnAdamOptimizer(AbbNnOptimizerBase):
    """
    Adam optimizer specific implementation of an AbbNnOptimizerBase.

    Properties:
       expected_hyperparam_list (List[str]): list of keys that must be
         present in the hyperparam_dict passed to __init__
    """

    def __init__(self, hyperparam_dict):
        """Initializes an AbbNnAdamOptimizer

        The `verify_hyperparams` method specified in this class is called
        by `super().__init__`

        Args:
            hyperparam_dict (dict): A dictionary containing hyperparameter values.

        Attributes:
            hyperparam_dict (dict): The original hyperparameter dictionary.
            optimizer (obj): The optimizer used by the model. (will be Adam)
            scheduler (obj): The scheduler used by the model. (will be None)
        """
        super().__init__(hyperparam_dict)

        self.optimizer = None
        self.scheduler = None

        # hyper params checked in super init


    @property
    def expected_hyperparam_list(self):
        return [
            "learn_rate",
            "weight_decay"
            ]


    def prepare_optimizer(self, model_parameters):
        """Sets up the optimizer for the model using the `model_parameters`
        dictionary.

        optimizer will be of type torch.optim.Adam
        schedule will be None (not used for Adam)

        Args:
            model_parameters (dict): A dictionary containing parameters for the
              model & optimizer

        Raises:
            AttributeError: If an optimizer is already built.
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                model_parameters,
                lr=self.hyperparam_dict["learn_rate"],
                weight_decay=self.hyperparam_dict["weight_decay"]
                )
            self.scheduler = None
        else:
            raise AttributeError(f"optimizer already built: {self.optimizer}")


    def verify_hyperparams(self):
        """Check that all expected hyperparameters are actually in the
        hyperparam_dict. See `expected_hyperparam_list` property.

        Raises ValueError if required hyperparameters are not in the dict
        """
        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            raise ValueError(f"missing hyper parameters: {missing_param_list}")


class AbbNnAdadeltaOptimizer(AbbNnOptimizerBase):
    """
    Ada-delta optimizer specific implementation of an AbbNnOptimizerBase.

    Properties:
       expected_hyperparam_list (List[str]): list of keys that must be
         present in the hyperparam_dict passed to __init__
    """

    def __init__(self, hyperparam_dict):
        """Initializes an AbbNnAdadeltaOptimizer

        The `verify_hyperparams` method specified in this class is called
        by `super().__init__`

        Args:
            hyperparam_dict (dict): A dictionary containing hyperparameter values.

        Attributes:
            hyperparam_dict (dict): The original hyperparameter dictionary.
            optimizer (obj): The optimizer used by the model. (will be Adadelta)
            scheduler (obj): The scheduler used by the model. (will be StepLR)
        """
        super().__init__(hyperparam_dict)

        self.optimizer = None
        self.scheduler = None

        # hyper params checked in super init


    @property
    def expected_hyperparam_list(self):
        return [
            "learn_rate",
            "scheduler_step_size",
            "scheduler_gamma"
            ]


    def prepare_optimizer(self, model_parameters):
        """Sets up the optimizer for the model using the `model_parameters`
        dictionary.

        optimizer will be of type torch.optim.Adadelta
        schedule will be of type torch.optim.lr_scheduler.StepLR

        Args:
            model_parameters (dict): A dictionary containing parameters for the
              model & optimizer

        Raises:
            AttributeError: If an optimizer is already built.
        """
        if (self.optimizer is None) and (self.scheduler is None):
            self.optimizer = torch.optim.Adadelta(
                model_parameters,
                lr=self.hyperparam_dict["learn_rate"]
                )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.hyperparam_dict["scheduler_step_size"],
                gamma=self.hyperparam_dict["scheduler_gamma"]
                )
        else:
            raise AttributeError(f"optimizer already built: {self.optimizer}")


    def verify_hyperparams(self):
        """Check that all expected hyperparameters are actually in the
        hyperparam_dict. See `expected_hyperparam_list` property.

        Raises ValueError if required hyperparameters are not in the dict
        """
        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            raise ValueError(f"missing hyper parameters: {missing_param_list}")
