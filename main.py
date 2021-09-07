import cv2 
import torch 
import numpy as np 
import torchvision as vision 
import imutils
import recognitionModel as nn 
from torchvision import transforms
import os
import speech_text as speech
import threading 
import tkinter as tk
import pickle

WEIGHTS__ = os.path.join(os.getcwd(), 'model weights/weights.path')
labels = ["fist","L","nothing","okay","palm","Peace"]

#loading the grammar
with open('grammar.pickle','rb') as grammar:
    commands = pickle.load(grammar)

# commands = {"fist":"Hey Google! Suggest me any book to read",
#             "L":"Hey Google , suggest me any movie",
#             "palm":"Hey Google! Show me my agenda for today",
#             "Peace":"Hey Google! Weather in Rome today ",
#             "okay":"Hey Google ! Check my mail"}

# reply ={"fist":"Hey Google! Make me laugh",
#             "L":"Hey Google! What is new on netflix",
#             "palm":"Hey google, set the alarm for tomorrow at 7 35 am",
#             "Peace":"Hey Google! Show me how to pass exams",
#             "okay":"Hey google! Show me flight prices from Rome to Paris for this week"
#             }


def getBackgroundSubstraction(background, frame):
    
    """"
    input: background image and the actual image
    output: segmented image with substracted background
    """
    
    g_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    g_foreground = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    dif = cv2.absdiff(g_background, g_foreground)
    _,result = cv2.threshold(dif, 25, 255, cv2.THRESH_BINARY)
    
    return result


if os.path.exists(os.path.join(os.getcwd(),'background.jpg')):
    os.remove(os.path.join(os.getcwd(),'background.jpg'))


model = nn.Net()
model.load_state_dict(torch.load(WEIGHTS__))
img_path = 'asl-alphabet/asl_alphabet_train/asl_alphabet_train/B/B1.jpg'
#declaring the transformer for normalizing the data
transform = transforms.Compose([transforms.ToTensor(), #data normalization
                                transforms.Normalize(
                                    (0.5,), 
                                    (0.5,))])

#Frame and camera setup
camera = cv2.VideoCapture(0)
top,right,bottom,left = 10,110,210,310


#Message window setup
messageWindow = tk.Tk()
window = tk.Text(messageWindow, height=30, width=50)
window.pack()




while(True):
    #get the current frame
    # frame = imutils.resize(frame, width=700)
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame,1)   
    clone = frame.copy()
    roi = frame[top:bottom, right:left]
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    cv2.imshow("Video Feed", clone)

    #estabilish the background at startup
    if not os.path.exists(os.path.join(os.getcwd(),'background.jpg')):
        cv2.imwrite('./background.jpg', img=roi)
        background = cv2.imread('./background.jpg')
        cv2.waitKey(1650)
        cv2.destroyAllWindows()
        


    keypress = cv2.waitKey(1) & 0xFF

    rgb = getBackgroundSubstraction(background, roi)

    # rgb = cv2.fastNlMeansDenoising(rgb,None,60,7,21)
    # rgb = cv2.cvtColor(roi, cv2.cv2.COLOR_BGR2GRAY)
    # rgb = cv2.flip(rgb, 1) #uncomment this line if you are a right-handed person
    # rgb = cv2.resize(rgb,(200,200))

    
    tensor_image = transform(rgb)
    tensor_image = tensor_image[None]
    tensor_image = tensor_image.type(torch.FloatTensor)
    
    with torch.no_grad():
        prediction = model(tensor_image)
        pred = prediction.max(1, keepdim=True)[1]
        pred = np.array(pred)
        prediction = np.array(prediction)
        
        
        cv2.putText(rgb, labels[int(pred[0][0])], (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        if (labels[int(pred[0][0])] != 'nothing'):

            #If L and higher thaan threshold
            if(labels[int(pred[0][0])] == 'L'):
                if prediction[0][int(pred[0][0])] >17.0:
                    message ="Command:" +commands[labels[int(pred[0][0])]]
                    speech.text_to_spech(commands[labels[int(pred[0][0])]])
                    print("prediction %s"%labels[int(pred[0][0])],prediction[0][int(pred[0][0])])
                    window.insert(tk.END, message + '\n')
                    text = speech.speec_to_text()
                    window.insert( tk.END, text + '\n')
                    window.update()      

            #if peace and higher than threshold
            elif(labels[int(pred[0][0])] == 'Peace'):
                if prediction[0][int(pred[0][0])] >18.0:
                    message ="Command:" +commands[labels[int(pred[0][0])]]
                    speech.text_to_spech(commands[labels[int(pred[0][0])]])
                    print("prediction %s"%labels[int(pred[0][0])],prediction[0][int(pred[0][0])])
                    window.insert(tk.END, message + '\n')
                    text = speech.speec_to_text()
                    window.insert(tk.END, text + '\n')
                    window.update()      

            #if any other command or label and higher than threshold        
            elif prediction[0][int(pred[0][0])] >16.0:
                print("prediction %s"%labels[int(pred[0][0])],prediction[0][int(pred[0][0])])
                message = "Command:" + commands[labels[int(pred[0][0])]]
                speech.text_to_spech(commands[labels[int(pred[0][0])]])
                window.insert(tk.END,message + '\n')
                text = speech.speec_to_text()
                window.insert(tk.END,text + '\n')
                window.update()      

    # cv2.imshow("Roi",rgb)
    # capture_pic(530)


    if keypress == 32: #press space, in case where there is a lot of noise and the threshold is not achieved
        message = "Command:" + commands[labels[int(pred[0][0])]]
        speech.text_to_spech(commands[labels[int(pred[0][0])]])
        window.insert(tk.END,message + '\n')
        text = speech.speec_to_text()
        window.insert(tk.END,text + '\n')
        window.update() 
        print(text)

    #This section was for testing purposes to add more commands and see if the system can work for a long time
    # if keypress == ord("a"):
    #     message = "Command:" + reply[labels[int(pred[0][0])]]
    #     speech.text_to_spech(reply[labels[int(pred[0][0])]])
    #     window.insert(tk.END,message + '\n')
    #     text = speech.speec_to_text()
    #     window.insert(tk.END,text + '\n')
    #     window.update() 
    #     print(text)

    #Catch the background manually by pressing s
    # if keypress == ord("s"):
    #     cv2.imwrite('./background.jpg', img=roi)
    #     cv2.waitKey(1650)
    #     cv2.destroyAllWindows()

    #close the application
    if keypress == ord("q"):
        break
    

# free up memory
camera.release()
cv2.destroyAllWindows()
# tk.mainloop()
