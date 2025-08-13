import os
import cv2
import mediapipe as mp


input_video_path = '/Users/marcomorandin/Desktop/ComputerVision/computer-vision/rectification/output/videos/out4.mp4'
mediapipe_outdir = 'output/'
output_video_path = mediapipe_outdir + 'video.mp4'

if not os.path.exists(mediapipe_outdir):
    os.makedirs(mediapipe_outdir)

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=2,
    enable_segmentation=True,
    static_image_mode=False
)

# Initialize MediaPipe drawing module for annotations.
mp_drawing = mp.solutions.drawing_utils


# Open the local video file.
cap = cv2.VideoCapture(input_video_path)

# Get video properties for output file.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output video.
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Write the frame into the output file.
    out.write(image)

    # # Display the annotated image (optional, can be commented out).
    # cv2.imshow('MediaPipe Pose', image)

    # # Break the loop when 'q' is pressed.
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources.
cap.release()
out.release()
cv2.destroyAllWindows()