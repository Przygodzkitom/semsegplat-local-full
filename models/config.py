class ModelConfig:
    def __init__(self, num_classes=2, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names if class_names else ["Background", "Object"]
        self.model_name = "unet"
        self.encoder_name = "resnet101"
        self.encoder_weights = "imagenet"
        self.learning_rate = 1e-4  # Default learning rate, won't be used for inference
        
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config
