# Getting Started


## Overview

- 입력된 yaml 형태의 config 경로를 읽고, config, data, model을 aws에 업로드 한다.
- Core쪽에 API request(compressions/create)를 보낸다.
- Core쪽에서 완료 progress(3)이 넘어올 때 까지 주기적으로 status를 print한다.
- Core쪽 process가 완료된 후에는 aws로부터 결과와 모델 파일등을 다운로드 받는다.


## Code Flow(main.py)

### 1. Upload config, data, model

```python
 # Upload config, data, model
comp_sess.upload_config(config_path=args.config, storage_config=configs["STORAGE"])
comp_sess.upload_data(data_path=configs["DATASET"]["path"], dataset_type=configs["DATASET"]["type"], storage_config=configs["STORAGE"])
comp_sess.upload_model(model_path=configs["INPUT"]["path"], model_type=configs["INPUT"]["type"], storage_config=configs["STORAGE"])
```

### 2. Run compression session

```python
compression_id = comp_sess.compress()
```

### 3. Monitor compression

```python
    while True:
        current_status = get_compression_status(compression_id, return_type=ReturnDataType.JSON)
        current_compression_id = current_status["compression_id"]
        print("==========current status==========")
        print(current_status)
        print("==================================")
        if current_compression_id == None: # 요청한 compression_id가 없는 경우, 프로그램 종료
            print(f"compression_id({current_compression_id}) does not exists")
            exit(0)
        elif current_status["status"] != 0: # 도중에 error가 발생한 경우(status!=0), 프로그램 종료
            print("error in progressing")
            exit(0)
        elif current_status["progress"] == 1: # compression task가 queue에서 대기하고 있음.
            print("[*] compression task in queue!")
            print("[*] currently stacked compression queue size : ", get_task_queue_size())
        elif current_status["progress"] == 2:
            print("[*] compression is in progress.")
        elif current_status["progress"] == 3: # 정상적으로 프로세스가 완료된 경우
            print("compression completed!")
            break
        print(f"[*] running time in current step: ", calculate_duration(time_from=current_status["updated_time"]))
        print(f"[*] total time: ", calculate_duration(time_from=current_status["created_time"]))
        print("\n\n")
        time.sleep(20)
```

### 4. Download result

```python
aws_auth_info = connection.get_auth()
    result = get_result(compression_id, return_type=ReturnDataType.JSON)
    print(result)
    log_file = connection.download_result_from_s3_with_url(aws_auth_info, compression_id, s3_url=result["url_log"])
    print(f"[*] log file(filename: {log_file}) saved")
    input_type_compressed_model = connection.download_result_from_s3_with_url(aws_auth_info, compression_id, s3_url=result["url_input_type_compressed_model"])
    print(f"[*] input type compressed model(filename: {input_type_compressed_model}) saved")
    converted_type_compressed_model = connection.download_result_from_s3_with_url(aws_auth_info, compression_id, s3_url=result["url_converted_type_compressed_model"])
    print(f"[*] converted type compressed model(filename: {converted_type_compressed_model}) saved")
```



## Modules

### 1. Compress Session(modules/compression_sessions.py)

#### Class purpose

1. config, data, model을 cloud(s3)로 업로드
2. compression session을 수행(core를 구동)

#### Class attribute

#### Class methods

- upload_config
  ```python
  """Upload config to s3.
  
  Args:
      config_path (str): config yaml file path 
      storage_config (Dict[str, Any]): storage config
  
  Return:
      bool: Whether its valid config
  """
  ```

- upload_data
  ```python
  """Upload data to s3.
  
  Args:
      data_path (str):  
      storage_config (Dict[str, Any]): storage config
      dataset_type (str): type of dataset: ["imagefolder", "npy"]
  """
  ```

- upload_model
  ```python
  """Upload model.
  
  Args:
      model_path (str): path to model
      storage_config (Dict[str, Any]): storage config from user_config
      model_type (str): type of model [pb, h5]
  """
  ```

- compress
  ```python
  """Requests create to core.
  
  Return:
      str: compression id
  """
  ```
 

### 2. Monitoring(modules/monitoring_apis.py)

netspresso_server 쪽으로 monitoring 요청을 보내는 함수들의 모음

각 API requests들의 return은 [netspresso-server](https://github.com/nota-github/netspresso-server/blob/develop/netspresso_server/urls.py) 를 확인

## Run
requirements.txt가 설치된 환경에서 구동

AWS 로그인 정보가 필요
```python
export PYTHONPATH=. && python netspresso_cli/main.py login
```

구동 example
```python
export PYTHONPATH=. && python netspresso_cli/main.py run --config config_files/tutorial_tf22.yml
```

### Terminal 예시
```bash
compression id: 86877af4-bd86-404d-86d8-7756cee9aaab
==========current status==========
{'compression_id': '86877af4-bd86-404d-86d8-7756cee9aaab', 'compression_number': 125, 'configs': {'COMPRESSION_ALIAS': 'tutorial_0_yaml', 'COMPRESSION_CONSTRAINTS': {'acceptable_drop_percent_point': 75, 'objective': 'accuracy'}, 'DATASET': {'dataloader_config': {'default_batch_size': 16, 'preprocessing': {'mean': [0, 0, 0], 'rescale_value': 255, 'std': [1, 1, 1]}}, 'path': {'zip_dir': 'https://netspresso-test-bucket2.s3.us-east-2.amazonaws.com/datasets/CIFAR10-images.zip'}, 'type': 'imagefolder'}, 'INPUT': {'image_height_width': [32, 32], 'path': 'https://netspresso-test-bucket2.s3.us-east-2.amazonaws.com/models/vgg_model.zip', 'test_accuracy_percent': 82, 'type': 'pb'}, 'OUTPUT': {'dtype': 'float16', 'model_type': 'tflite', 'test_device': 'pc'}, 'STORAGE': {'destination_path': '/example1_yaml', 'region_name': 'us-east-2', 's3_bucket_name': 'nota-netspresso-bucket', 'type': 's3'}, 'TASK': 'classification'}, 'created_time': '2021-05-17 09:56:45', 'progress': 1, 'sequence_name': 'tutorial_0_yaml', 'status': 0, 'updated_time': '2021-05-17 09:56:45', 'worker_assigned': None}
==================================
[*] compression task in queue!
[*] currently stacked compression queue size :  1
[*] running time in current step:  0:00:01.916832
[*] total time:  0:00:01.918583

==========current status==========
{'compression_id': '86877af4-bd86-404d-86d8-7756cee9aaab', 'compression_number': 125, 'configs': {'COMPRESSION_ALIAS': 'tutorial_0_yaml', 'COMPRESSION_CONSTRAINTS': {'acceptable_drop_percent_point': 75, 'objective': 'accuracy'}, 'DATASET': {'dataloader_config': {'default_batch_size': 16, 'preprocessing': {'mean': [0, 0, 0], 'rescale_value': 255, 'std': [1, 1, 1]}}, 'path': {'zip_dir': 'https://netspresso-test-bucket2.s3.us-east-2.amazonaws.com/datasets/CIFAR10-images.zip'}, 'type': 'imagefolder'}, 'INPUT': {'image_height_width': [32, 32], 'path': 'https://netspresso-test-bucket2.s3.us-east-2.amazonaws.com/models/vgg_model.zip', 'test_accuracy_percent': 82, 'type': 'pb'}, 'OUTPUT': {'dtype': 'float16', 'model_type': 'tflite', 'test_device': 'pc'}, 'STORAGE': {'destination_path': '/example1_yaml', 'region_name': 'us-east-2', 's3_bucket_name': 'nota-netspresso-bucket', 'type': 's3'}, 'TASK': 'classification'}, 'created_time': '2021-05-17 09:56:45', 'progress': 2, 'sequence_name': 'tutorial_0_yaml', 'status': 0, 'updated_time': '2021-05-17 09:56:48', 'worker_assigned': '172.31.33.155:8000'}
==================================
[*] compression is in progress.
[*] running time in current step:  0:00:19.325615
[*] total time:  0:00:22.326615

...

==========current status==========
{'compression_id': '7cdf0e17-934c-494f-a5fd-644a109c9285', 'compression_number': 467, 'configs': {'COMPRESSION_ALIAS': 'tutorial_tf22_yaml', 'COMPRESSION_CONSTRAINTS': {'acceptable_drop_percent_point': 75, 'objective': 'accuracy'}, 'DATASET': {'dataloader_config': {'default_batch_size': 16, 'preprocessing': {'mean': [0.4914, 0.4822, 0.4465], 'rescale_value': 255, 'std': [0.2023, 0.1994, 0.201]}}, 'path': {'zip_dir': 'https://netspresso-test-bucket2.s3.us-east-2.amazonaws.com/datasets/CIFAR10-images.zip'}, 'type': 'imagefolder'}, 'INPUT': {'image_height_width': [32, 32], 'path': 'https://netspresso-test-bucket2.s3.us-east-2.amazonaws.com/models/vgg_model.zip', 'test_accuracy_percent': 79.66, 'type': 'pb'}, 'MOCK_TEST': True, 'OUTPUT': {'dtype': 'float16', 'model_type': 'tflite', 'test_device': 'pc'}, 'STORAGE': {'destination_path': '/example_tf22_yaml', 'region_name': 'us-east-2', 's3_bucket_name': 'nota-netspresso-bucket', 'type': 's3'}, 'TASK': 'classification'}, 'created_time': '2021-05-06 06:23:08', 'progress': 3, 'sequence_name': 'tutorial_tf22_yaml', 'status': 0, 'updated_time': '2021-05-06 06:31:30', 'worker_assigned': '172.31.16.109:8000'}
==================================
compression completed!
{'finish_code': 0, 'finished_time': '2021-05-06 06:31:30', 'result': "{'original': {'performance': {'Acc(%)': 78.43, 'Runtime_mean(ms)': 20.3126, 'Runtime_std(ms)': 1.2692}, 'size': {'Flops(M)': 626.8053, 'Trainable(M)': 14.7198, 'Non_trainable': 0, 'Layers': 21, 'Model_size(MB)': 58.9455}}, 'compressed': {'size': {'Flops(M)': 102.7329, 'Trainable(M)': 2.3674, 'Non_trainable': 0, 'Layers': 21, 'Model_size(MB)': 9.5355}, 'performance': {'Acc(%)': 64.52, 'Runtime_mean(ms)': 17.4322, 'Runtime_std(ms)': 0.3623}}, 'quantized': {'size': {'Model_size(MB)': 4.7484}, 'performance': {'Accuracy(%)': 64.5132, 'Runtime_mean(ms)': 2.8873, 'Runtime_std(ms)': 0.0722}}, 'device': {}}", 'url_converted_type_compressed_model': 'https://nota-netspresso-bucket.s3.us-east-2.amazonaws.com/7cdf0e17-934c-494f-a5fd-644a109c9285/data/7cdf0e17-934c-494f-a5fd-644a109c9285/0/converted/tflite_float16.zip', 'url_input_type_compressed_model': 'https://nota-netspresso-bucket.s3.us-east-2.amazonaws.com/7cdf0e17-934c-494f-a5fd-644a109c9285/data/7cdf0e17-934c-494f-a5fd-644a109c9285/0/compressed_model.zip', 'url_log': 'https://nota-netspresso-bucket.s3.us-east-2.amazonaws.com/7cdf0e17-934c-494f-a5fd-644a109c9285/data/7cdf0e17-934c-494f-a5fd-644a109c9285/0/user_report.log'}
[*] log file(filename: result/7cdf0e17-934c-494f-a5fd-644a109c9285/user_report.log) saved
[*] input type compressed model(filename: result/7cdf0e17-934c-494f-a5fd-644a109c9285/compressed_model.zip) saved
[*] converted type compressed model(filename: result/7cdf0e17-934c-494f-a5fd-644a109c9285/tflite_float16.zip) saved
[*] worker status as YAML format
- status: 0
  updated_time: Thu, 06 May 2021 06:31:34 GMT
  worker_id: 172.31.16.109:8000
- status: 0
  updated_time: Tue, 04 May 2021 07:53:42 GMT
  worker_id: 172.31.28.134:8000

[*] compression status list as DATA_FRAME format
                           compression_id  compression_number config_type  ... status         updated_time     worker_assigned
0    efb4e10e-ec4e-4ea4-b82b-ce645ad30045                 124  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
1    67aabfff-a703-4bf7-887f-1720d2a6ebc5                 125  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
2    d3da0a91-766c-4477-ad62-eb92b137d2ed                 126  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
3    93154683-e3e1-41ca-ba75-5703ccba285b                 127  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
4    71a64815-6ec3-4450-a20c-5e363e1adb8a                 128  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
..                                    ...                 ...         ...  ...    ...                  ...                 ...
195  78cfe3e4-0e85-4726-b164-88599ee07454                 319  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
196  7ce4caca-c3d6-43e1-8b13-e836e580aa40                 320  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
197  5cb9f92f-6d87-4060-aac4-aa3211280562                 321  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
198  7a786050-d54d-4b27-a5f8-03bf2655d7d2                 322  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
199  734cc5a9-3395-40c4-8900-d4c3b4687925                 323  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000

[200 rows x 10 columns]

```

