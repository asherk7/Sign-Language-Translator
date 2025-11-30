"""
Real-time ASL Sign Language Detection using webcam.
Press 'q' to quit.
"""

import torch
import cv2
from torchvision import models, transforms
import os

CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
           'u', 'v', 'w', 'x', 'y', 'z', 'nothing', 'space']

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'mobilenet_v2_sign_language.pth')

def load_model(model_path, num_classes, device):
    """Load the trained MobileNetV2 model."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def get_transform():
    """Get the same transform used during training/testing."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first using create_model.py")
        return
    
    model = load_model(MODEL_PATH, len(CLASSES), device)
    transform = get_transform()
    print("Model loaded successfully!")
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Define ROI (region of interest) - box where hand should be placed
        h, w = frame.shape[:2]
        box_size = 300
        x1, y1 = w - box_size - 50, 50
        x2, y2 = x1 + box_size, y1 + box_size
        
        # Draw the ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand here", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Extract ROI and predict
        roi = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Transform and predict
        input_tensor = transform(roi_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = CLASSES[predicted.item()]
        conf = confidence.item()
        
        # Display prediction
        label = f"{predicted_class.upper()}: {conf:.1%}"
        cv2.putText(frame, label, (x1, y2 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Show frame
        cv2.imshow('ASL Sign Language Detector', frame)
        
        if conf > 0.5:
            print(f"Detected: {predicted_class.upper()} ({conf:.1%})")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
