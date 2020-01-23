from flask import Flask, render_template, Response
import socket, json, time

app = Flask(__name__)

@app.route('/')
def index():
    return(render_template('index.html'))

@app.route('/topic/gcs')
def get_messages():
    def events():        
        (csock, adr) = sock.accept()
        print ("Client Info: ", csock, adr) 
        client = csock 
        try:                   
            while 1:
                msg = client.recv(1024).decode('utf-8') # type(msg):str
                if not msg:
                    print("----------------------------------------------")
                    print(f'client {adr} closed')                       
                else:
                    print ("Client send: " + msg)                                             
                yield "data:{0}\n\n".format(msg) 
        except Exception as e:
            print(e)
            pass
    return Response(events(), mimetype="text/event-stream")

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', 9999))
    sock.listen(5)
    print('Server start ...')
    app.run(host='0.0.0.0',port=5002,threaded=False)
    

