import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7)

startDis = None
scale = 0
cx, cy = 200, 200

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]

        hand1_fingers = detector.fingersUp(hand1)
        hand2_fingers = detector.fingersUp(hand2)

        if hand1_fingers == [1, 1, 0, 0, 0] and hand2_fingers == [1, 1, 0, 0, 0]:
            lmList1 = hand1["lmList"]
            lmList2 = hand2["lmList"]

            length, info, img = detector.findDistance(hand1["center"], hand2["center"], img)
            midX = (hand1["center"][0] + hand2["center"][0]) // 2
            midY = (hand1["center"][1] + hand2["center"][1]) // 2

            cv2.putText(img, f'Distance: {int(length)}', (midX - 50, midY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            startDis = length

        else:
            startDis = None

    else:
        startDis = None

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
