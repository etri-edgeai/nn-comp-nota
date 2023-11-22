from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod, Options


def model_compression(num, compression_type="l2norm"):
    compressor = ModelCompressor(email=args.email, password=args.password)

    # Upload Model
    UPLOAD_MODEL_NAME = "vit"
    TASK = Task.IMAGE_CLASSIFICATION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = "model_vit.pt"
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]

    model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )

    if compression_type == "l2norm":
        COMPRESSED_MODEL_NAME = "l2norm_vit"
        COMPRESSION_METHOD = CompressionMethod.PR_L2
        RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
        RECOMMENDATION_RATIO = 0.2

        OUTPUT_PATH = f"compressed_vit_l2norm_{num}.pt"
        opt=Options(group_policy='none')
        compressed_model = compressor.recommendation_compression(
            model_id=model.model_id,
            model_name=COMPRESSED_MODEL_NAME,
            compression_method=COMPRESSION_METHOD,
            recommendation_method=RECOMMENDATION_METHOD,
            recommendation_ratio=RECOMMENDATION_RATIO,
            output_path=OUTPUT_PATH,
            options=opt,
        )

def main(args):
    model_compression = "not implemented yet"
    compressed_model = model_compression(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('-mail', '--email', type=str, default="email", help='')
    parser.add_argument('-pw', '--password', type=str, default="password", help='')
    args = parser.parse_args()

    main(args)
