import cv2
import face_recognition

# Carregue a imagem de referência e a codifique
imagem_referencia = face_recognition.load_image_file("imagem_referencia.jpg")
codificacao_referencia = face_recognition.face_encodings(imagem_referencia)[0]

# Inicialize a webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capture um quadro da webcam
    _, frame = webcam.read()

    # Converta o quadro em RGB
    rgb_frame = frame[:, :, ::-1]

    # Detecte os rostos no quadro e codifique-os
    rostos = face_recognition.face_locations(rgb_frame)
    codificacoes = face_recognition.face_encodings(rgb_frame, rostos)

    # Compare as codificações com a imagem de referência
    for codificacao in codificacoes:
        resultado = face_recognition.compare_faces([codificacao_referencia], codificacao)
        if resultado[0]:
            print("Rosto reconhecido!")

    # Mostre o quadro na janela da webcam
    cv2.imshow("Webcam", frame)

    # Espere por uma tecla pressionada
    if cv2.waitKey(1) == ord("q"):
        break

# Libere a webcam e feche a janela
webcam.release()
cv2.destroyAllWindows()
