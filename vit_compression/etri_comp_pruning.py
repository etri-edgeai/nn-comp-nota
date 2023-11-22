from netspresso.compressor import ModelCompressor, Task, Framework


def model_compression(args):
    compressor = ModelCompressor(email=args.email, password=args.password)

    # Upload Model
    UPLOAD_MODEL_NAME = "vit"
    TASK = Task.IMAGE_CLASSIFICATION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = "model_vit.pt"
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]

def main(args):
    model_compression = "not implemented yet"
    compressed_model = model_compression(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('-mail', '--email', type=str, default="email", help='')
    parser.add_argument('-pw', '--password', type=str, default="password", help='')
    args = parser.parse_args()

    main(args)
