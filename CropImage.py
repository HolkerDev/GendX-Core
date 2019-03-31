import cv2


def crop(image_path):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print(cascade)
    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_img, 1.3, 7, minSize=(30, 30))
    print(faces)
    faces_arr = list()
    count = 0
    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        count += 1
        face_img = image[ny:ny + nr, nx:nx + nr]
        cv2.imwrite('image' + str(count) + '.png', face_img)
        faces_arr.append(face_img)
    return faces_arr
