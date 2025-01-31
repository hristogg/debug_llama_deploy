import requests
import json
import time
from datetime import datetime

test_local = 'http://localhost:8080'

cloudrun_url = 'https://llamadebugcont-475381037639.europe-west3.run.app'

service_name = 'testwork'
#Change below to test_local if tesitng local deplyoment
endpoint = f'{cloudrun_url}/deployments/{service_name}/tasks/run'
print(f'Trying to endpoint: {endpoint}')
headers = {'Content-type': 'application/json'}
now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
payload = {'query': 'Who is Paul Graham', 'user': 'test'}
data = {"input": json.dumps(payload)}
print(f'Starting request : {now}')
print(f'Used data is {data} and endpoint is {endpoint}')
response = requests.post(endpoint,json=data,headers=headers)
now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
print(f'Finished request {now}')
print(response)
print(response.json())

