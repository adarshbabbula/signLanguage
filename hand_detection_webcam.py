import cv2
import mediapipe as mp 
import joblib
import numpy as np 
from gtts import gTTS
import os
from s import *
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
s=''
i=0
k=0
current=0
vowels={"ఆ":"ా", "ఇ":"ి","ఈ":"ీ","ఉ":"ు","ఊ":"ూ", "ఋ":"ృ", "ౠ":"ౄ", "ఎ":"ె", "ఏ":"ే", "ఐ": "ై", "ఒ": "ొ", "ఓ": "ో", "ఔ": "ౌ", "అం": "ం", "అః":"ః" }
dictionary={
  "a":'అ',
  "aa":"ఆ",
  "i":'ఇ',
  "u": "ఉ",
  "e":"ఎ",
  "ai":"ఐ",
  "o": "ఒ",
  "au": "ఔ",
  "am": "అం",
  "ka": "క",
  "ga": "గ",
  "ja": "జ",
  "tta": "ట",
  "da":"డ",
  "ta": "త",
  "dha": "ద",
  "na": "న",
  "pa":"ప",
  "fa":"ఫ",
  "ba": "బ",
  "ma": "మ",
  "ya": "య",
  "ra": "ర",
  "la": "ల",
  "va": "వ",
  "sa": "స",
  "ha":"హ",
  "cha": "చ",
}
# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def data_clean(landmark):
  
  data = landmark[0]
  
  try:
    data = str(data)

    data = data.strip().split('\n')

    garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

    without_garbage = []

    for i in data:
        if i not in garbage:
            without_garbage.append(i)

    clean = []

    for i in without_garbage:
        i = i.strip()
        clean.append(i[2:])

    for i in range(0, len(clean)):
        clean[i] = float(clean[i])

    
    return([clean])

  except:
    return(np.zeros([1,63], dtype=int)[0])

while cap.isOpened():
  success, image = cap.read()
  
  image = cv2.flip(image, 1)
  
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cleaned_landmark = data_clean(results.multi_hand_landmarks)
    #print(cleaned_landmark)

    if cleaned_landmark:
      clf = joblib.load('model.pkl')
      y_pred = clf.predict(cleaned_landmark)
      if i==0:
        current=y_pred[0]
      if i==30:
        # if(k==0):
        #   clf = joblib.load('model.pkl')
        #   y_pred = clf.predict(cleaned_landmark)
        #   s+=str(dictionary[y_pred[0]])
        # if(k==1):
        #   clf = joblib.load('model.pkl')
        #   y_pred = clf.predict(cleaned_landmark)
        #   last=s[-1]
        #   c=last
        #   print("b: "+s[:-1])
        #   s=str(s[:-1]+str(fun1(last, dictionary[str(y_pred[0])])))
        #   print("a: "+s[:-1])
        #   print("s: "+fun1(last, dictionary[str(y_pred[0])]))
        # if(k==2):
        #   clf = joblib.load('model1.pkl')
        #   y_pred = clf.predict(cleaned_landmark)
        #   last=s[-1]
        #   print(last,dictionary[str(y_pred[0])],c)
        #   # s=s[:-1]+fun2(last,dictionary[str(y_pred[0])],c)
        # for i in s:
        #   print(i,end=' ')
        # print(" ")
        # k=(k+1)%3
        # print(k)
        clf = joblib.load('model.pkl')
        y_pred = clf.predict(cleaned_landmark)
        print(dictionary[y_pred[0]])
        i=0
      if current==y_pred[0]:
        i+=1
      else:
        current=y_pred[0]
        i=0
    image = cv2.putText(image, str(y_pred[0]), (50,150), cv2.FONT_HERSHEY_SIMPLEX,  3, (0,0,255), 2, cv2.LINE_AA) 
  cv2.imshow('MediaPipe Hands', image)
  
  if cv2.waitKey(5) & 0xFF == 27:
    mytext = s
    language = 'te'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("sound.mp3")
    break

hands.close()
cap.release()
cv2.destroyAllWindows()