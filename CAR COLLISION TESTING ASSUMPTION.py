import cv2
import numpy as np
import torch
import torchvision
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configurations
CONFIDENCE_THRESHOLD = 0.5
FRAMERATE = 10
PPM = 8.8
VIDEO_PATH = r"C:\Users\Aditya Singh\OneDrive\Desktop\SPEED DECTETION\2165-155327596_tiny.mp4"
VIDEO_FRAME_RATE = 30  # Example frame rate
DISTANCE_UNIT_CONVERSION = 0.1  # Conversion factor from pixels to meters
TIME_UNIT_CONVERSION = 1 / VIDEO_FRAME_RATE  # Conversion factor from frames to seconds
AIRBAG_TIMINGS = [0.5, 1.0, 1.5]  # Different timings for airbag deployment in seconds

def load_model():
    logger.info("Loading pre-trained model...")
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    model.eval()
    return model

def estimate_speed(location1, location2, ppm, time_unit_conversion, distance_unit_conversion):
    d_pixels = math.sqrt((location2[0] - location1[0]) ** 2 + (location2[1] - location1[1]) ** 2)
    d_meter = d_pixels * distance_unit_conversion / ppm
    speed = d_meter / time_unit_conversion * 3.6  # Convert m/s to km/h
    return speed

def detect_cars(frame, frame_rate, ppm, time_unit_conversion, distance_unit_conversion, model):
    # Detect cars in the frame
    img_tensor = torchvision.transforms.ToTensor()(frame)
    with torch.no_grad():
        predictions = model([img_tensor])
    
    cars = []
    for pred, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
        x1, y1, x2, y2 = map(int, pred)
        if score >= CONFIDENCE_THRESHOLD:
            centroid_X = (x1 + x2) / 2
            centroid_Y = (y1 + y2) / 2
            speed = estimate_speed((x1, y1), (x2, y2), ppm, time_unit_conversion, distance_unit_conversion)
            cars.append({
                'bbox': (x1, y1, x2, y2),
                'centroid': (centroid_X, centroid_Y),
                'speed': speed
            })
    return cars

def detect_collisions(cars):
    collisions = []
    for i in range(len(cars)):
        for j in range(i + 1, len(cars)):
            car1 = cars[i]
            car2 = cars[j]
            distance = math.sqrt((car2['centroid'][0] - car1['centroid'][0]) ** 2 + (car2['centroid'][1] - car1['centroid'][1]) ** 2)
            time_to_collision = distance / (car1['speed'] - car2['speed']) if car1['speed'] > car2['speed'] else float('inf')
            if time_to_collision <= 1:  # Assume collision if less than or equal to 1 second to collision
                collisions.append((car1, car2))
    return collisions

def calculate_collision_impact(car1, car2):
    relative_speed = abs(car1['speed'] - car2['speed'])
    max_speed = max(car1['speed'], car2['speed'])
    if max_speed == 0:
        return 0
    impact_percentage = (relative_speed / max_speed) * 100
    return impact_percentage

def calculate_survival_chance(impact_percentage, speed, airbag_timing):
    # Define a more nuanced heuristic to estimate survival chances
    base_chance = 100  # Initial survival chance
    chance_reduction = (impact_percentage + speed) / 2  # Reduce chance based on impact and speed
    survival_chance = max(0, base_chance - chance_reduction)

    # Adjust survival chance based on airbag timing
    # Define thresholds for airbag deployment timing
    early_threshold = 0.5  # If airbag deploys before this time
    mid_threshold = 1.0    # If airbag deploys between early and mid threshold
    late_threshold = 1.5   # If airbag deploys after this time

    if airbag_timing < early_threshold:
        # Increase survival chance significantly for early airbag deployment
        survival_chance += 30
    elif early_threshold <= airbag_timing < mid_threshold:
        # Moderate increase for mid-range airbag deployment
        survival_chance += 15
    elif mid_threshold <= airbag_timing < late_threshold:
        # Slight increase for late airbag deployment
        survival_chance += 5

    return survival_chance

def annotate_collisions(frame, collisions, airbag_timing):
    for car1, car2 in collisions:
        x1, y1, x2, y2 = car1['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
        x1, y1, x2, y2 = car2['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
        impact_percentage = calculate_collision_impact(car1, car2)
        speed = max(car1['speed'], car2['speed'])
        survival_chance = calculate_survival_chance(impact_percentage, speed, airbag_timing)
        cv2.putText(frame, f"Impact: {impact_percentage:.2f}%", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
        cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
        cv2.putText(frame, f"Survival Chance: {survival_chance:.2f}%", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
        cv2.putText(frame, f"Airbag Timing: {airbag_timing} s", (x1, y1 - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)  # Pause until a key is pressed
    return frame

def main():
    model = load_model()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error('Unable to load the video')
        return
    
    for airbag_timing in AIRBAG_TIMINGS:
        logger.info(f"Running simulation with airbag timing: {airbag_timing} seconds")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                breakpoint
            cars = detect_cars(frame, FRAMERATE, PPM, TIME_UNIT_CONVERSION, DISTANCE_UNIT_CONVERSION, model)
            collisions = detect_collisions(cars)
            annotated_frame = annotate_collisions(frame, collisions, airbag_timing)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


