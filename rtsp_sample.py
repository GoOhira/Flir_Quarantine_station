import rtsp
import cv2
import numpy as np
import requests
import time
import random
import struct
from ethernetip import EtherNetIP
from PIL import Image
from io import BytesIO

def testENIP(flug):
    hostname = "172.21.48.11"
    broadcast = "172.21.48.255"
    inputsize = 1
    outputsize = 1
    EIP = EtherNetIP(hostname)
    C1 = EIP.explicit_conn(hostname)

    """
    listOfNodes = C1.scanNetwork(broadcast,5)
    print("Found ", len(listOfNodes), " nodes")
    for node in listOfNodes:
        name = node.product_name.decode()
        sockinfo = SocketAddressInfo(node.socket_addr)
        ip = socket.inet_ntoa(struct.pack("!I",sockinfo.sin_addr))
        print(ip, " - ", name)
    """

    pkt = C1.listID()
    if pkt != None:
        print("Product name: ", pkt.product_name.decode())

    pkt = C1.listServices()
    print("ListServices:", str(pkt))

    # Connected
    C1.registerSession()

    r = C1.getAttrSingle(0x67, 1, 1)   #clas, inst, attr   #Flir get pallette class=0x67, instance=1,attribute=1
    if None == r:
        print("Could read 0x67 :",r)
    else:
        print("Failed to read 0x67:", r)

    # Flir MSX
    path = C1.mkReqPath(0x70, 1, None)#mkReqPath(self, clas, inst, attr):
    #data = bytes.fromhex("2A 2E 69 6D 61 67 65 2E 73 79 73 69 6D 67 2E 66 75 73 69 6F 6E 2E 66 75 73 69 6F 6E 44 61 74 61 2E 66 75 73 69 6F 6E 4D 6F 64 65 03 00 00 00")
    if flug == 1:
        string = '.image.sysimg.fusion.fusionData.fusionMode'.encode('ascii')
        data = len(string).to_bytes(1,'big') + string + bytes.fromhex("03 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
        # Visual mode
        data = len(string).to_bytes(1,'big') + string + bytes.fromhex("01 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
        string2 = '.image.sysimg.fusion.fusionData.useLevelSpan'.encode('ascii')
        data = len(string2).to_bytes(1,'big') + string2 + bytes.fromhex("00 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
    elif flug == 2:
        # MSX mode
        string = '.image.sysimg.fusion.fusionData.fusionMode'.encode('ascii')
        data = len(string).to_bytes(1,'big') + string + bytes.fromhex("03 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
    else:
        # MSX mode
        string = '.image.sysimg.fusion.fusionData.fusionMode'.encode('ascii')
        data = len(string).to_bytes(1,'big') + string + bytes.fromhex("03 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
        # Visual mode
        data = len(string).to_bytes(1,'big') + string + bytes.fromhex("01 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
        string2 = '.image.sysimg.fusion.fusionData.useLevelSpan'.encode('ascii')
        data = len(string2).to_bytes(1,'big') + string2 + bytes.fromhex("00 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
        # Thermal mode
        data = len(string).to_bytes(1,'big') + string + bytes.fromhex("01 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
        string2 = '.image.sysimg.fusion.fusionData.useLevelSpan'.encode('ascii')
        data = len(string2).to_bytes(1,'big') + string2 + bytes.fromhex("01 00 00 00")
        r = C1.unconnSend(0x35, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,
    #LED ON/OFF
    str_torch = '.system.vcam.torch'.encode('ascii')
    data = len(str_torch).to_bytes(1,'big') + str_torch + bytes.fromhex("00")   #01:ON 00:OFF
    print(data)
    r = C1.unconnSend(0x33, path+data, random.randint(1,4026531839))  #unconnSend(self, service, data,

    #pallette
    str_pal = 'bw.pal'.encode('ascii')
    #str_pal = 'iron.pal'.encode('ascii')
    data = len(str_pal).to_bytes(1,'big') + str_pal
    r = C1.setAttrSingle(0x67, 1, 1, data)    #setAttrSingle(self, clas, inst, attr, data):

    # quality
    data = bytes.fromhex("00")  #00:high, 01:normal, 02:low
    r = C1.setAttrSingle(0x67, 1, 3, data)

    #adjust
    mintemp = 20
    maxtemp = 30
    str_adjust = 'Manual'.encode('ascii')



    #str_adjust = 'Auto'.encode('ascii')
    data = len(str_adjust).to_bytes(1,'big') + str_adjust
    r = C1.setAttrSingle(0x67, 1, 4, data)    #setAttrSingle(self, clas, inst, attr, data):
    """
    data = bytes.fromhex("01")  #00:do nothing, 01:execute
    r = C1.setAttrSingle(0x67, 1, 9, data)
    str_adjust = 'Linear'.encode('ascii')
    #str_adjust = 'Histogram'.encode('ascii')
    data = len(str_adjust).to_bytes(1,'big') + str_adjust
    r = C1.setAttrSingle(0x67, 1, 10, data)    #setAttrSingle(self, clas, inst, attr, data):
    """
    #min
    tem = 273.15+mintemp        #IEEE 32-bit Single Precision Floating Point
    temp = np.ones(1, dtype=np.float32).astype('float32') * tem
    data = struct.pack('<f',temp)
    r = C1.setAttrSingle(0x67, 1, 5, data)
    #max
    tem = 273.15+maxtemp        #IEEE 32-bit Single Precision Floating Point
    temp = np.ones(1, dtype=np.float32).astype('float32') * tem
    data = struct.pack('<f',temp)
    r = C1.setAttrSingle(0x67, 1, 6, data)

    r = C1.getAttrSingle(0x67, 1, 5)   #clas, inst, attr   #Flir get pallette class=0x67, instance=1,attribute=1
    if None == r:
        print("Could read 0x67 :",r)
    else:
        print("Failed to read 0x67:", struct.unpack('<f',r[1]))


# https://github.com/opencv/data/haarcascades/
# git cloneして使用
#cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')


testENIP(1)
print("acaca")
baseurl = "172.21.48.11"

#baseurl = "222.229.69.231"
#rtsp_server_uri = 'rtsp://222.229.69.231/avc'
#rtsp_server_uri = 'rtsp://222.229.69.231/avc?overlay=off'
rtsp_server_uri = 'rtsp://admin:admin@'+baseurl+'/avc?overlay=off'

http_server_uri = 'http://222.229.69.231/livelonly/adumin:admin'
http_server_uri = 'http://222.229.69.231/snapshot.jpg?user=user&pwd=user'
vis_uri = 'http://222.229.69.231/home/setviewmode/VISUAL'
vis_uri2 = 'http://222.229.69.231/home/getviewmode'
ir_uri = 'http://222.229.69.231/home/setviewmode/IR'
ir_uri2 = 'http://222.229.69.231/home/getviewmode'
status = 'http://222.229.69.231/camera/status'
login = 'http://222.229.69.231/'


"""
# sincle fetch
client = rtsp.Client(rtsp_server_uri, verbose=True)
client.read().show()
client.close()
"""

preview_flug = 0;
get_flug = 1;
# stream preview
if preview_flug == 1:
    with rtsp.Client(rtsp_server_uri) as client:
        _image = client.read()
        client.preview()
        print("ababa")
        #print(_image.shape)
        im = np.asarray(_image)[:, :, ::-1]

if get_flug == 1:
    flug = 1
    count = 0
    print("flug=0まではOK")
    with rtsp.Client(rtsp_server_uri) as client:
        client.preview()
        while True:
            #ループ回数を表示
            count = count +1
            print(count)
    #        process_image(_image)
            _image = client.read()
            #_image.show()
            im = np.asarray(_image)[:, :, ::-1]
    #        image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('flir', im)
            key=cv2.waitKey(1)
            #       key=input(block=False)
            #フラグが1の時にはカラー画像(顔検出)
            if flug==1:
                testENIP(1)
                _image = client.read()
                #_image.show()
                im = np.asarray(_image)[:, :, ::-1]
                cv2.imshow('flir', im)
                #r = requests.get(vis_uri)
                flug=2
                #r = requests.get(vis_uri2)
                print("testENIP == 1\n")
                print("**************************\n")
                print("im.shape=\n")
                print(im.shape)
                # 処理速度を高めるために画像をグレースケールに変換したものを用意
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                print("グレイスケール画像を表示....")
                cv2.imshow('flir', gray)
                #認識結果の保存
                cv2.imwrite('./result/test1_gray.png', gray)
                cv2.imwrite('./result/test1_RGB.png', im)
                time.sleep(2)

                # 顔検出
                print("顔検出開始")
                facerect = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=3,
                    minSize=(10, 10)
                )
                color = (255, 255, 255) #白
                # 検出した場合
                print("検出結果=")
                print(facerect)
                if len(facerect) > 0:
                    print("顔面検出成功")

                    #検出した顔を囲む矩形の作成
                    for rect in facerect:
                        cv2.rectangle(im, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)

                    #認識結果の表示
                    cv2.imshow('flir', im)
                    cv2.imwrite('./result/test1_detect.png', im)
                time.sleep(2)


            else:
                testENIP(2)
                cv2.imshow('flir', im)
                time.sleep(2)
                #r = requests.get(ir_uri)
                flug=1
                print("testENIP == 2 \n")
            if key ==ord('q'):
                break
"""
# both
#r = requests.get(login,auth=('admin', 'admin'))
#print(r.status_code)
while True:
    #r = requests.post(vis_uri)
    #print(r.status_code)
    #print(r.content)
    #r2 = requests.post(vis_uri2)
    #print(r2.text)
    #r = requests.get(http_server_uri)
    #im = Image.open(BytesIO(r.content))
    #img= np.asarray(im)[:, :, ::-1]
    #cv2.imshow('flir vis', img)
    #r = requests.post(ir_uri)
    #print(r.content)
    #r = requests.post(ir_uri2)
    #print(r.text)
    r = requests.get(rtsp_server_uri)
    im = Image.open(BytesIO(r.content))
    img= np.asarray(im)[:, :, ::-1]
    cv2.imshow('flir ir', img)

    key = cv2.waitKey(100)
    if key==ord('q'):
        break
"""
