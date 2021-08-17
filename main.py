from flask import Flask,render_template,redirect,request,url_for
import numpy as np
import face_recognition as fr
import cv2
import os
app= Flask(__name__)


@app.route("/")
def home():
    return render_template("frontend3.html")
@app.route("/proceed",methods=["GET","POST"])
def index2():
    if(request.method=="POST"):
        return render_template("frontend2.html")
@app.route("/submit",methods=["GET","POST"])
def resultpage():
    if(request.method=="POST"):
        return render_template("submit.html")
@app.route("/backend",methods=["GET","POST"])
def index3():
    if(request.method=="POST"):
        
        video_capture = cv2.VideoCapture(0)
        video_capture.set(3,640)
        video_capture.set(4,480)

        #classnames=["Mobile"]
        config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weights_path='frozen_inference_graph.pb'
        model=cv2.dnn_DetectionModel(weights_path,config_file)
        model.setInputSize(320,320)
        model.setInputScale(1.0/ 127.5)
        model.setInputMean((127.5,127.5,127.5))
        model.setInputSwapRB(True)

        first_image = fr.load_image_file("Face/face1.jpg")
        first_face_encoding = fr.face_encodings(first_image)[0]

        known_face_encondings = [first_face_encoding]
        known_face_names = ["Name  of candidate"]
        cnt=0
        while True: 
            ret, frame = video_capture.read()
            #frame=cv2.resize(frame1,dsize=(600,600),interpolation=cv2.INTER_AREA)
            # small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            rgb_frame =frame[:, :, ::-1]

            face_locations = fr.face_locations(rgb_frame)
            face_encodings = fr.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                matches = fr.compare_faces(known_face_encondings, face_encoding)

                name = "Unknown"

                face_distances = fr.face_distance(known_face_encondings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                if name=="Unknown":
                    cnt+=1
                if cnt>=5:
                    return render_template("finalpage.html")
                
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0),3)

                cv2.rectangle(frame, (left, bottom-40), (right, bottom), (0,255,0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                print(name)

            

            #..........................................................................................................................#
            classIds,confs,bbox=model.detect(frame,confThreshold=0.5)
            print(classIds,bbox)

            if len(classIds)!=0:
                for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    
                    
                    if classId==77:
                        print("You have been highlighted")
                        cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
                        cv2.putText(frame,"Mobile",(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cnt+=1
                        if cnt>=5:
                            return render_template("finalpage.html")
                    elif classId==73:
                        cv2.putText(frame,"Laptop",(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cnt+=1
                    '''elif classId==1:
                        cv2.putText(frame,"Person",(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)'''


            cv2.imshow('Webcam_facerecognition', frame)
            #cv2.imshow("Object Recognition",frame)
        #.................................................................................................................................#
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        return render_template("frontend3.html")

 
    
# @app.route("/result",methods=['GET','POST'])
# def result():
#     if request.method=="POST":



if __name__=="__main__":
    app.run(debug=True)


