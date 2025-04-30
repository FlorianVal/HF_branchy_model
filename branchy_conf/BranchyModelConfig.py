from typing import List, Optional
from transformers import PretrainedConfig
import logging

logger = logging.getLogger(__name__)

class BranchyModelConfig(PretrainedConfig):
    """
    Configuration class for BranchyModel. This class extends the PretrainedConfig class from the Transformers
    library, providing configuration specific to models with branch functionality.

    Attributes:
        branch_locations (List[int]): Specifies the indices of layers after which branches are added. These indices
            start from 0, and each index represents a layer in the underlying transformer model.
        penalty_weight (Optional[float]): The weight of the penalty term used in the "penalized_cross_entropy" loss.
            This parameter is required and must be greater than 0
        window_size (int): Determines the number of tokens each branch considers from the input sequence. This allows
            for reducing the computational load by limiting the context size each branch processes.

    Example:
        config = BranchyModelConfig(
            branch_locations=[2, 4, 6],
            window_size=512
        )

    Note:
        This configuration class is specifically designed for use with the BranchyModel class, enabling flexible
        and customizable branching within transformer models.
    """
    model_type = "branchy"  # Optional, but useful for identifying the model type in the Transformers library

    def __init__(
        self,
        model_str: str = None,
        head_thresholds: Optional[List[float]] = None,
        confidence_metric: Optional[str] = "breaking_ties",
        branch_locations: Optional[List[int]] = None,
        branch_number: Optional[int] = 3,
        penalty_weight: Optional[float] = 0,
        head_window_size: int = 512,
        copy_lm_head: Optional[bool] = False,
        **kwargs
    ):
        """
        Initializes the BranchyModelConfig.

        Args:
            model_str (str): The model string to be used for the model. From Huggingface's model hub.
            branch_locations (List[int], optional): Locations of the branches. Defaults to None, indicating no branches.
            branch_number (Optional[int], optional): Number of branches if branch_locations is not provided. Defaults to 3.
            penalty_weight (Optional[float], optional): Weight for the penalty in loss calculation.
                . Defaults to None.
            head_window_size (int, optional): Number of tokens each branch can see. Defaults to 512.
        """
        self.model_str = model_str
        self.head_thresholds = head_thresholds
        self.confidence_metric = confidence_metric
        assert self.confidence_metric in ["breaking_ties", "max"], "confidence_metric must be 'breaking_ties' or 'max'. It should depend on how you found the thresholds."
        self.branch_locations = branch_locations 
        self.penalty_weight = penalty_weight
        self.head_window_size = head_window_size
        if branch_locations is not None and branch_number is not None:
            logger.warning("Both branch_locations and branch_number are provided. Using branch_locations.")
        self.branch_number = branch_number if branch_locations is None else len(branch_locations)
        self.copy_lm_head = copy_lm_head
        #assert self.model_str is not None, "model_str must be provided."
        assert self.branch_number > 0, "branch_number must be a positive integer."
        assert isinstance(self.penalty_weight, float) or isinstance(self.penalty_weight, int), "penalty_weight must be a float or an integer."
        assert self.penalty_weight >= 0 and self.penalty_weight <= 1, "penalty_weight must be in the range [0, 1]."
        if branch_locations is not None:
            assert all([isinstance(loc, int) for loc in self.branch_locations]), "Branch locations must be integers."
            assert all([loc >= 0 for loc in self.branch_locations]), "Branch locations must be non-negative."
        if self.head_window_size is not None:
            assert self.head_window_size > 0 , "head_window_size must be a positive integer or None."
        if type(self.head_thresholds) == list:
            assert len(self.head_thresholds) == self.branch_number, "Number of thresholds must match number of branches."
            assert all([isinstance(threshold, float) for threshold in self.head_thresholds]), "Thresholds must be floats."
        super().__init__(**kwargs)  # Initialize with base class parameters