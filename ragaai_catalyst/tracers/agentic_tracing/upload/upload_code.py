from aiohttp import payload
import requests
import json
import os
import time
import logging
from ragaai_catalyst.ragaai_catalyst import RagaAICatalyst
logger = logging.getLogger(__name__)

def upload_code(hash_id, zip_path, project_name, dataset_name, base_url=None):
    code_hashes_list = _fetch_dataset_code_hashes(project_name, dataset_name, base_url)

    if hash_id not in code_hashes_list:
        presigned_url = _fetch_presigned_url(project_name, dataset_name, base_url)
        _put_zip_presigned_url(project_name, presigned_url, zip_path)

        response = _insert_code(dataset_name, hash_id, presigned_url, project_name, base_url)
        return response
    else:
        return "Code already exists"

def _fetch_dataset_code_hashes(project_name, dataset_name, base_url=None):
    payload = {}
    headers = {
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "X-Project-Name": project_name,
    }

    try:
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        start_time = time.time()
        endpoint = f"{url_base}/v2/llm/dataset/code?datasetName={dataset_name}"
        response = requests.request("GET", 
                                    endpoint, 
                                    headers=headers, 
                                    data=payload,
                                    timeout=99999)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

        if response.status_code == 200:
            return response.json()["data"]["codeHashes"]
        else:
            raise Exception(f"Failed to fetch code hashes: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list datasets: {e}")
        raise 

def _fetch_presigned_url(project_name, dataset_name, base_url=None):
    payload = json.dumps({
            "datasetName": dataset_name,
            "numFiles": 1,
            "contentType": "application/zip"
            })

    headers = {
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "Content-Type": "application/json",
        "X-Project-Name": project_name,
    }

    try:
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        start_time = time.time()
        endpoint = f"{url_base}/v1/llm/presigned-url"
        response = requests.request("GET", 
                                    endpoint, 
                                    headers=headers, 
                                    data=payload,
                                    timeout=99999)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

        if response.status_code == 200:
            return response.json()["data"]["presignedUrls"][0]
        else:
            raise Exception(f"Failed to fetch code hashes: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list datasets: {e}")
        raise

def _put_zip_presigned_url(project_name, presignedUrl, filename):
    headers = {
            "X-Project-Name": project_name,
            "Content-Type": "application/zip",
        }

    if "blob.core.windows.net" in presignedUrl:  # Azure
        headers["x-ms-blob-type"] = "BlockBlob"
    print(f"Uploading code...")
    with open(filename, 'rb') as f:
        payload = f.read()

    start_time = time.time()
    response = requests.request("PUT", 
                                presignedUrl, 
                                headers=headers, 
                                data=payload,
                                timeout=99999)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.debug(
        f"API Call: [PUT] {presignedUrl} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
    if response.status_code != 200 or response.status_code != 201:
        return response, response.status_code

def _insert_code(dataset_name, hash_id, presigned_url, project_name, base_url=None):
    payload = json.dumps({
        "datasetName": dataset_name,
        "codeHash": hash_id,
        "presignedUrl": presigned_url
        })
    
    headers = {
        'X-Project-Name': project_name,
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
        }
    
    try:
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        start_time = time.time()
        endpoint = f"{url_base}/v2/llm/dataset/code"
        response = requests.request("POST", 
                                    endpoint, 
                                    headers=headers, 
                                    data=payload,
                                    timeout=99999)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
        if response.status_code == 200:
            return response.json()["message"]
        else:
            raise Exception(f"Failed to insert code: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to insert code: {e}")
        raise