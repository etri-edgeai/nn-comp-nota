from netspresso.compressor import ModelCompressor, Task, Framework


def model_compression():
    compressor = ModelCompressor(email="email", password="password")

    # Upload Model
    UPLOAD_MODEL_NAME = "vit"
    TASK = Task.IMAGE_CLASSIFICATION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = "model_vit.pt"
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]

def main():
    model_compression = "not implemented yet"
    compressed_model = model_compression()


if __name__ == "__main__":
    main()
