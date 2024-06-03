import cv2
import mediapipe as mp

# Initialize Mediapipe Holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Colors for drawing
color_face = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
color_face_connection = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
color_hand = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4)
color_pose = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4)  # Define color_pose

display_width, display_height = 1080, 720

def capture_video():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process the image and detect landmarks
            result = holistic.process(image)

            # Convert the RGB image back to BGR for rendering using OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw the landmarks
            mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      color_face, color_face_connection)
            mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, color_hand)
            mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, color_hand)
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, color_pose)

            # Resize the image to the desired display size
            image = cv2.resize(image, (display_width, display_height))

            # Display the image
            cv2.imshow('Real-Time Object Detection with Landmarks', image)

            # Check for 'q' key press to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()