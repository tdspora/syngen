from typing import Dict, Any, Optional


class FeatureLossLayer:
    """
    TensorFlow-free placeholder for the old Keras FeatureLossLayer.
    In the PyTorch port, losses are computed explicitly in the training loop,
    so this class is a no-op kept only to preserve construction sites that
    previously instantiated the Keras layer.
    """

    def __init__(self, feature: Any, **kwargs: Any) -> None:
        self.feature = feature
        self.name: Optional[str] = kwargs.get("name")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Keras code used to call the layer to add a loss; here we simply
        # passthrough the last positional argument when present to avoid
        # breaking call sites that might expect a return value.
        return args[-1] if args else None


class SampleLayer:
    """
    TensorFlow-free placeholder for the old Keras SampleLayer.
    The reparameterization trick and capacity-style loss should be implemented
    directly in the PyTorch model/criterion. This class remains only to avoid
    import errors until the model is fully ported.
    """

    def __init__(self, gamma: float, capacity: float, **kwargs: Any) -> None:
        self.gamma = gamma
        self.max_capacity = capacity

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "SampleLayer is deprecated in the PyTorch port. Implement reparameterization "
            "and capacity loss directly in the model forward/loss computation."
        )

    @property
    def get_config(self) -> Dict[str, Any]:
        return {"gamma": self.gamma, "capacity": self.max_capacity}
