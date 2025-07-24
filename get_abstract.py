# import pymysql
# import json
# import requests
# import pandas as pd
# from tqdm import tqdm
# # from googletrans import Translator
# import torch# import time

# from transformers import MarianMTModel, MarianTokenizer
# import random
# import random
# import uuid

# def generate_unique_id():
#     # 使用 uuid4 生成一个唯一的 ID，然后取前 16 个字符
#     unique_id = uuid.uuid4().hex
#     return unique_id

# MYSQLDB = {
#     'host': '47.254.123.228',
#     'port': 3306,
#     'user': 'wrl',
#     'password': 'wrl12177',
#     'db': 'pubmed',
#     'charset': 'utf8',
# }

# def get_mysql_data(cursor, sql, limit_num):
#     i = 0
#     all_data = []
#     while True:
#         cursor.execute(f"{sql} LIMIT %s, %s", (i * limit_num, limit_num))
#         each_data = cursor.fetchall()
#         if each_data:
#             # 获取列名
#             columns = [col[0] for col in cursor.description]
#             for row in each_data:
#                 row_dict = dict(zip(columns, row))
#                 all_data.append(row_dict)
#             i += 1
#         else:
#             break
#     return all_data

# def get_mysql_connection():
#     return pymysql.Connect(
#         host=MYSQLDB['host'],
#         port=MYSQLDB['port'],
#         user=MYSQLDB['user'],
#         passwd=MYSQLDB['password'],
#         db=MYSQLDB['db'],
#         charset=MYSQLDB['charset']
#     )

# def get_self_data():
#     connection = get_mysql_connection()
#     cursor = connection.cursor()

#     try:
#         # sql_query = 'SELECT abstract,title FROM pubmed_all_0611a WHERE state = 1'
#         sql_query = 'SELECT abstract,title,url_id FROM pubmed_daixie WHERE state = 1'
#         result = get_mysql_data(cursor, sql_query, 60000)
#     except Exception as e:
#         result = []
#         print(f"初始化失败 error={e}")
#     finally:
#         cursor.close()
#         connection.close()
    
#     return result

# def request(text):
#     url = 'http://192.168.2.201:5000/predict_'
#     res = requests.post(url=url, json={'text':text})
#     return json.loads(res.text)

# # 调用函数并获取数据
# data = get_self_data()
# result = []
# for infos in tqdm(data):
#     abss = infos['title'] + '.' + infos['abstract'] + '.' + 'PMID:' +  infos['url_id']
#     result.append(abss)
#     # print()       
#     #          
# with open('datasetsutiles/metabolism20241017/metabolism1.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)

import pymysql
import json
import requests
import pandas as pd
from tqdm import tqdm
# from googletrans import Translator
import torch# import time


from transformers import MarianMTModel, MarianTokenizer
import random
import random
import uuid
import time
from nanoid import generate

def generate_unique_id(length=8):
    # 使用默认的字符集（a-zA-Z0-9）
    return generate('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', length)

def generate_unique_id_int(length=8):
    # 使用默认的字符集（a-zA-Z0-9）
    return generate('0123456789', length)



def request(text):
    url = 'http://192.168.2.201:5000/predict_'
    res = requests.post(url=url, json={'text':text})
    return json.loads(res.text)

# # 调用函数并获取数据
# data = get_self_data()
result = []
with open('datasetsutiles/metabolism20241017/metabolism.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for idx,infos in enumerate(tqdm(data)):
    response = request(infos)
    text = response['text']
    if len(response['relation_list']) == 0:
        continue
    enReList = []
    temp = {}
    for relation in response['relation_list']:
        entityAsite = relation['entity_a_pos']
        entityBsite = relation['entity_b_pos']
        entityAid = generate_unique_id()
        entityBid = generate_unique_id()
        if str(entityAsite) not in temp.keys():
            temp[str(entityAsite)] = entityAid
        if str(entityBsite) not in temp.keys():
            temp[str(entityBsite)] = entityBid
        enA = {
                "value": {
                    "start": entityAsite[0],
                    "end": entityAsite[1],
                    "text": relation['entity_a_name'],
                    "labels": [
                        relation['entity_a_label']
                    ]
                },
                "id": temp[str(entityAsite)],
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual"
            }
        if enA not in enReList:
            enReList.append(enA)
        enB = {
                "value": {
                    "start": entityBsite[0],
                    "end": entityBsite[1],
                    "text": relation['entity_b_name'],
                    "labels": [
                        relation['entity_b_label']
                    ]
                },
                "id": temp[str(entityBsite)],
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual"
            }
        if enB not in enReList:
            enReList.append(enB)
        relationText = relation['relation']
        if 'inhibits function' == relation['relation']:
            relationText = 'inhibits'
        if 'promotes function' == relation['relation']:
            relationText = 'promotes'
        if 'shortening' == relation['relation']:
            relationText = 'abbreviation'
        if 'Signal' == relation['relation']:
            relationText = 'Signal pathway'
        enReList.append({
                        "from_id": temp[str(entityAsite)],
                        "to_id": temp[str(entityBsite)],
                        "type": "relation",
                        "direction": "right",
                        "labels": [
                            relationText
                        ]
                    })
  
    
    result.append({
        "id": int(generate_unique_id_int()),
        "annotations": [
          {
            "id": int(generate_unique_id_int()),
            "result": enReList,
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": "2023-05-05T09:11:47.308616Z",
            "updated_at": "2023-05-05T09:11:47.308616Z",
            "lead_time": 44622.122,
            "prediction": {},
            "result_count": 0,
            "task": "",
            "parent_prediction": "",
            "parent_annotation": ""
          }
        ],
        "file_upload": "",
        "drafts": [],
        "predictions": [],
        "data": {
          "text": text
        },
        "meta": {},
        "created_at": "2023-05-05T09:11:47.308616Z",
        "updated_at": "2023-05-05T09:11:47.308616Z",
        "project": idx # 自增ID
      })
    
random.shuffle(result)
with open('datasetsutiles/metabolism20241017/metabolismPredict.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)