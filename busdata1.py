from pykafka import KafkaClient
import json
from datetime import datetime
import uuid
import time
import sys
#READ COORDINATES FROM GEOJSON
input_file = open('./data/bus1.json')
json_array = json.load(input_file)
coordinates = json_array['features'][0]['geometry']['coordinates']

#GENERATE UUID
def generate_uuid():
    return uuid.uuid4()

#KAFKA PRODUCER
client = KafkaClient(hosts="localhost:9092")
topic = client.topics['gcs']
producer = topic.get_sync_producer()

#CONSTRUCT MESSAGE AND SEND IT TO KAFKA
data = {}
data['channel'] = '00001'

def generate_checkpoint(coordinates):
    i = 0
    while i < len(coordinates):
        data['key'] = data['channel'] + '_' + str(generate_uuid())
        data['timestamp'] = str(datetime.utcnow())
        data['latitude'] = coordinates[i][1]
        data['longitude'] = coordinates[i][0]
        message = json.dumps(data)
        print(message)
        producer.produce(message.encode('ascii'))
        time.sleep(1)

        #if bus reaches last coordinate, start from beginning
        if i == len(coordinates)-1:
            i = 0
        else:
            i += 1
if __name__ == '__main__':
    try:
        generate_checkpoint(coordinates)
    except:
        sys.exit(0)

"""
#------------  open two servers -------------
Step1:
    cd D:\kafka\bin\windows 
    zookeeper-server-start.bat ../../config/zookeeper.properties
Step2:
    cd D:\kafka\bin\windows
    kafka-server-start.bat ../../config/server.properties
Step3:(if exsist topic, go to step4)
    cd D:\kafka\bin\windows
    kafka-topics.bat --zookeeper localhost:2181 --topic testbusdata --create --partitions 1 --replication-factor 1

#-----------  option (or use pykafkafclient)  -----------
producer:
    cd D:\kafka\bin\windows
    kafka-console-producer.bat --broker-list localhost:9092 --topic testbusdata
consumer:
    cd D:\kafka\bin\windows
    kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic testbusdata 
pykafka producer:
    cd /your data path/
    python busdata1.py
    python busdata2.py
    python busdata3.py

#------------ shotdown server ---------------
Step1:
    kafka-server-stop.bat
Step2:
    kafka-zookeeper-stop.bat
"""