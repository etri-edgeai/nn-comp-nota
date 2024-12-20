import requests
import os
import json
from urllib.request import urlretrieve
from netspresso_cli import settings
from netspresso_cli.clouds.types import ReturnDataType, DataSetFormat
from netspresso_cli.clouds.codec import encoder


def get_compression_status_list(
    return_type: ReturnDataType = ReturnDataType.JSON,
) -> ReturnDataType:
    """Get compression status list.
    
    compression status list consists as follows,

    [*] compression status list as DATA_FRAME format
                               compression_id  compression_number config_type  ... status         updated_time     worker_assigned
    0    efb4e10e-ec4e-4ea4-b82b-ce645ad30045                 124  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
    1    67aabfff-a703-4bf7-887f-1720d2a6ebc5                 125  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
    2    d3da0a91-766c-4477-ad62-eb92b137d2ed                 126  constraint  ...      0  2021-05-04 06:48:09  172.31.16.109:8000
    ..                                    ...                 ...         ...  ...    ...                  ...                 ...
    """
    r = requests.get(
        f"http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/api/v1/compressions"
    )
    return encoder(json_data=r.json(), output_format=return_type)


def get_worker_status_list(
    return_type: ReturnDataType = ReturnDataType.JSON,
) -> ReturnDataType:
    r = requests.get(
        f"http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/api/v1/worker_status_list"
    )
    return encoder(json_data=r.json(), output_format=return_type)


def get_compression_status(compression_id: str, return_type: ReturnDataType = ReturnDataType.JSON):
    r = requests.get(
        f"http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/api/v1/compressions/{compression_id}"
    )
    return encoder(json_data=r.json(), output_format=return_type)

def get_task_queue_size():
    r = requests.get(
        f"http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/api/v1/task_queue_status"
    )
    return r.json()

def get_result(
    compression_id: str,
    return_type: ReturnDataType = ReturnDataType.JSON,
) -> ReturnDataType:
    r = requests.get(
        f"http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/api/v1/compressions/{compression_id}/result"
    )
    return encoder(json_data=r.json(), output_format=return_type)


def download_log_file(compression_id: str, dst_folder_path:str)->None:
    target_url = f'http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/compression_status/{compression_id}/log/download/'
    return download_file(compression_id, dst_folder_path, target_url)

def download_original_type_compressed_model_file(compression_id: str, dst_folder_path:str)->None:
    target_url = f'http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/compression_status/{compression_id}/original_model/download/'
    return download_file(compression_id, dst_folder_path, target_url)

def download_converted_type_compressed_model_file(compression_id: str, dst_folder_path:str)->None:
    target_url = f'http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/compression_status/{compression_id}/compressed_model/download/'
    return download_file(compression_id, dst_folder_path, target_url)


def download_file(compression_id: str, dst_folder_path:str, target_url: str)->None:
    # get file name
    target_filename = None
    pre_response = requests.head(target_url)
    if pre_response.status_code >= 400: # there is an error
        print("download_log_file failed!")
    else:
        try:
            target_filename = pre_response.headers["Content-Disposition"].split("=")[1].strip("\"")
            target_path = os.path.join(dst_folder_path, target_filename)
            urlretrieve(target_url, target_path)
        except Exception as e:
            print(f"error occured! {e}")
            exit(0)
    return target_filename


def delete_compression_id_in_task_queue(compression_id: str)->None:
    """delete compression_id in task queue
    (example)
    print(delete_compression_id_in_task_queue("ff2d0848-5daf-48de-9e61-0e1eb7761766"))
    """
    data = {
        "compression_id": compression_id,  # yaml config
    }
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    r = requests.post(
        f"http://{settings.API_SERVER.HOST}:{settings.API_SERVER.PORT}/api/v1/task_queue/delete",
        data=json.dumps(data), headers=headers
    )
    return r.json() # return compression_id
