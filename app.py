from flask import Flask, render_template, Response
from pykafka import KafkaClient
from pykafka.common import OffsetType

def get_kafka_client():
    return KafkaClient(hosts='localhost:9092')

app = Flask(__name__)

@app.route('/')
def index():
    return(render_template('index.html'))

#Consumer API
@app.route('/topic/<topicname>')
def get_messages(topicname):
    client = get_kafka_client()
    def events():
        for i in client.topics[topicname].get_simple_consumer(
            auto_offset_reset=OffsetType.LATEST,
            reset_offset_on_start=True):
            yield 'data:{0}\n\n'.format(i.value.decode())
    return Response(events(), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True, port=5002)
    
"""
#------------  open two servers -------------
Step1:
    cd D:\kafka\bin\windows 
    zookeeper-server-start.bat ../../config/zookeeper.properties
Step2:
    cd D:\kafka\bin\windows
    kafka-server-start.bat ../../config/server.properties
Step3:(if exsist topic, it's ok)
    cd D:\kafka\bin\windows
    kafka-topics.bat --zookeeper localhost:2181 --topic gcs --create --partitions 1 --replication-factor 1

#-----------  option (or use pykafkafclient)  -----------
producer:
    cd D:\kafka\bin\windows
    kafka-console-producer.bat --broker-list localhost:9092 --topic gcs
consumer:
    cd D:\kafka\bin\windows
    kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic gcs 
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