"""
"""

import torch


class AbbNnOptimizerBase():
    """
    """

    def __init__(self, hyperparam_dict):
        self.hyperparam_dict = hyperparam_dict
        self.optimizer = None  # set in base class
        self.scheduler = None  # (optionally) set in base class

        self.verify_hyperparams()


    def verify_hyperparams(self):
        raise NotImplementedError("AbbNnOptimizer.verify_hyperparams")


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
    """

    def __init__(self, model_parameters, hyperparam_dict):
        super().__init__(hyperparam_dict)

        self.optimizer = torch.optim.Adam(
            model_parameters,
            lr=self.hyperparam_dict["learn_rate"],
            weight_decay=self.hyperparam_dict["weight_decay"]
            )
        self.scheduler = None


    @property
    def expected_hyperparam_list(self):
        return [
            "learn_rate",
            "weight_decay"
            ]


    def verify_hyperparams(self):
        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            emsg = f"missing hyper parameters: {missing_param_list}"
            raise ValueError(f"AbbNnAdamOptimizer.verify_hyperparams {emsg}")


class AbbNnAdadeltaOptimizer(AbbNnOptimizerBase):
    """
    """

    def __init__(self, model_parameters, hyperparam_dict):
        super().__init__(hyperparam_dict)

        self.optimizer = torch.optim.Adadelta(
            model_parameters,
            lr=self.hyperparam_dict["learn_rate"]
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hyperparam_dict["scheduler_step_size"],
            gamma=self.hyperparam_dict["scheduler_gamma"]
            )


    @property
    def expected_hyperparam_list(self):
        return [
            "learn_rate",
            "scheduler_step_size",
            "scheduler_gamma"
            ]


    def verify_hyperparams(self):
        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            emsg = f"missing hyper parameters: {missing_param_list}"
            raise ValueError(f"AbbNnAdadeltaOptimizer.verify_hyperparams {emsg}")
