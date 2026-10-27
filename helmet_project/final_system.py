from ultralytics import YOLO
import cv2
import time
import winsound
import threading

model_ppe = YOLO("runs/detect/train/weights/best.pt")   # helmet + vest
model_person = YOLO("yolov8n.pt")                      # person detection

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

last_save_time = 0

def play_alert():
    winsound.Beep(1000, 500)  # frequency (Hz), duration (ms)

cv2.namedWindow("Mining Safety System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mining Safety System", 1100, 750)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    results_ppe = model_ppe(frame, conf=0.4)
    results_person = model_person(frame, conf=0.4)

    boxes_ppe = results_ppe[0].boxes
    boxes_person = results_person[0].boxes

    helmet_detected = False
    vest_detected = False
    violation = False   # ✅ ADD THIS

    person_detected = False
    unconscious_detected = False

    if boxes_person is not None and len(boxes_person) > 0:
        for box in boxes_person:
            cls = int(box.cls[0])

            if cls == 0:
                person_detected = True

                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width = x2 - x1
                height = y2 - y1

                # 🧠 Unconscious logic
                if width > height:
                    unconscious_detected = True

    if boxes_ppe is not None and len(boxes_ppe) > 0:
        for box in boxes_ppe:
            cls = int(box.cls[0])

            if cls == 0:
                helmet_detected = True
            elif cls == 1:
                vest_detected = True

    # Draw YOLO boxes first
    display_frame = results_ppe[0].plot()

    # Draw person boxes also
    for box in boxes_person or []:
        cls = int(box.cls[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, "Person", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        

    # Add ALERT text on SAME frame
    if person_detected:
        if unconscious_detected:
            cv2.putText(display_frame, "PERSON DOWN!", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            violation = True

        if not helmet_detected:
            cv2.putText(display_frame, "NO HELMET!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            violation = True

        if not vest_detected:
            cv2.putText(display_frame, "NO VEST!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            violation = True
    else:
        cv2.putText(display_frame, "NO PERSON", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    if violation:
        current_time = time.time()
        if current_time - last_save_time > 5:
            #filename = f"violation_{int(current_time)}.jpg"
            #cv2.imwrite(filename, frame)
            #print(f"[ALERT] Violation saved: {filename}")

            # 🔊 PLAY SOUND
            threading.Thread(target=play_alert).start()

            last_save_time = current_time
            
    
    cv2.putText(display_frame, f"Helmet: {helmet_detected}", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(display_frame, f"Vest: {vest_detected}", (50, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(display_frame, f"Person: {person_detected}", (50, 210),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(display_frame, f"Unconscious: {unconscious_detected}", (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    


    # Resize frame for better presentation
    display_frame = cv2.resize(display_frame, (1100, 750))

    # Title bar
    cv2.rectangle(display_frame, (0, 0), (1100, 45), (40, 40, 40), -1)
    cv2.putText(display_frame, "AI-Based Real-Time Mining Safety Monitoring",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # SHOW WINDOW
    cv2.imshow("Mining Safety System", display_frame)

    # EXIT KEY
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()