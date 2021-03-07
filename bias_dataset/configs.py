
biased_config = {
    'background_split': 2,
    'dataset_seed': 0,
    'tex_res': 400,
    'tile_size': 64,
    'train_samples': 50000,
    'test_samples': 1000,
    'exclude_bias_textures': True,
    'fix_test_set': True,
    'batch_size': 100,
    'bias': {
        1: {
            "source_1_id": "'1bbf4548.png",
            "source_2_id": "'2fbd466c.png",
            "source_1_bias": 0.0,
            "source_2_bias": 1.0
        },
        2: {
            "source_1_id": "'feeccd96.png",
            "source_2_id": "'f135d029.png",
            "source_1_bias": 0.0,
            "source_2_bias": 1.0
        },
    },
    'textures_path': 'bias_dataset/textures/',
}

unbiased_config = {
    'background_split': 2,
    'dataset_seed': 0,
    'tex_res': 400,
    'tile_size': 64,
    'train_samples': 50000,
    'test_samples': 1000,
    'exclude_bias_textures': False,
    'fix_test_set': True,
    'batch_size': 100,
    'bias': None,
    'textures_path': 'bias_dataset/textures/',
}
