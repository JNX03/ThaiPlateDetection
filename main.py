import cv2
import easyocr
import numpy as np
import re
from fuzzywuzzy import process

known_provinces = ["กรุงเทพมหานคร", "เชียงใหม่", "ภูเก็ต", "นนทบุรี", "ชลบุรี", "ปทุมธานี", 
                   "นครราชสีมา", "สงขลา", "ระยอง", "ขอนแก่น", "นครปฐม"]

reader = easyocr.Reader(['th', 'en'])
detected_plates = {}

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    processed_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    return processed_image

def postprocess_text(text):
    text = re.sub(r'[^0-9ก-ฮ]', '', text)
    match = re.match(r"([ก-ฮ]+)(\d+)", text)
    if match:
        thai_part, number_part = match.groups()
        corrected_text = f"{thai_part} {number_part}"
        return corrected_text
    return text

def match_province(detected_text):
    best_match = process.extractOne(detected_text, known_provinces)
    if best_match and best_match[1] > 50:
        return best_match[0]
    return None

def merge_text_fragments(detected_texts):
    merged_text = ' '.join(detected_texts)
    possible_province = None
    plate_text = merged_text
    for detected_text in detected_texts:
        possible_province = match_province(detected_text)
        if possible_province:
            plate_text = merged_text.replace(possible_province, '').strip()
            break
    plate_text = postprocess_text(plate_text)
    plate_pattern = r"^[ก-ฮ]{1,2}\d{1,4}$|^\d{1}[ก-ฮ]{1,2}\d{1,4}$"
    if not re.match(plate_pattern, plate_text):
        plate_text = "Invalid Plate"
    return plate_text, possible_province

def find_license_plate_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    possible_plates = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 1.5 < aspect_ratio < 5.5 and w > 50 and h > 15:
                possible_plates.append((x, y, w, h))
    return possible_plates

def save_plate_image(plate_image, plate_id):
    filename = f"plate_{plate_id}.jpg"
    cv2.imwrite(filename, plate_image)
    print(f"Plate image saved: {filename}")

def detect_license_plate_realtime():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (640, 480))
        preprocessed_frame = preprocess_image(frame_resized)
        plate_contours = find_license_plate_contours(preprocessed_frame)
        for (x, y, w, h) in plate_contours:
            plate_image = frame_resized[y:y+h, x:x+w]
            plate_id = f"{x}-{y}-{w}-{h}"
            if plate_id in detected_plates:
                continue
            result = reader.readtext(plate_image)
            if result:
                detected_texts = []
                for (bbox, text, prob) in result:
                    if prob > 0.5:
                        detected_texts.append(text)
                        print(f"Detected Text: {text} (Confidence: {prob:.2f})")
                if detected_texts:
                    combined_text, corrected_province = merge_text_fragments(detected_texts)
                    if not corrected_province:
                        corrected_province = ''
                    if combined_text != "Invalid Plate":
                        save_plate_image(plate_image, plate_id)
                        detected_plates[plate_id] = {'text': combined_text, 'province': corrected_province}
                        display_text = combined_text
                        if corrected_province:
                            display_text += f", {corrected_province}"
                        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame_resized, display_text, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("License Plate Detection", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

detect_license_plate_realtime()
