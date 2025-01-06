import cv2
import face_recognition


def load_reference_image(file_path):
    """
    Load and encode the reference image.

    :param file_path: Path to the reference image
    :return: Encoded reference face
    """
    try:
        image = face_recognition.load_image_file(file_path)
        encoding = face_recognition.face_encodings(image)[0]
        print("Reference image loaded and encoded successfully.")
        return encoding
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit()
    except IndexError:
        print("Error: No faces found in the reference image.")
        exit()


def initialize_webcam():
    """
    Initialize the webcam.

    :return: VideoCapture object
    """
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not access the webcam.")
        exit()
    print("Webcam initialized successfully.")
    return webcam


def recognize_faces(webcam, reference_encoding):
    """
    Capture frames from the webcam and recognize faces.

    :param webcam: VideoCapture object
    :param reference_encoding: Encoded reference face
    """
    while True:
        # Capture a frame from the webcam
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Convert the frame to RGB
        rgb_frame = frame[:, :, ::-1]

        # Detect faces and encode them
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Compare detected faces with the reference encoding
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            match = face_recognition.compare_faces([reference_encoding], encoding)
            if match[0]:
                # Draw a bounding box around the recognized face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Recognized", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print("Face recognized!")

        # Display the frame in a window
        cv2.imshow("Webcam", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) == ord("q"):
            print("Exiting webcam feed.")
            break


def main():
    """
    Main function to load the reference image, initialize the webcam, and start face recognition.
    """
    # Load the reference image
    reference_image_path = "imagem_referencia.jpg"
    reference_encoding = load_reference_image(reference_image_path)

    # Initialize the webcam
    webcam = initialize_webcam()

    # Recognize faces
    recognize_faces(webcam, reference_encoding)

    # Release resources
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


