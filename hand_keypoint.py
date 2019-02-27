import cv2


class HandKeyPoint:

    def __init__(self):
        self.protoFile = "model/hand/pose_deploy.prototxt"
        self.weightsFile = "model/hand/pose_iter_102000.caffemodel"
        self.num_points = 22
        self.threshold = 0.1
        self.points = []
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

    '''
        Do inference on a pre loaded image
    '''
    def doInference(self, image):
        imageHeight = image.shape[0]
        imageWidth = image.shape[1]
        aspect_ratio = imageWidth / imageHeight

        inHeight = 368
        inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
        inBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        self.net.setInput(inBlob)
        output = self.net.forward()

        for i in range(self.num_points):
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (imageWidth, imageHeight))

            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > self.threshold:
                self.points.append((int(point[0]), int(point[1])))
            else:
                self.points.append(None)

        return self.points

    '''
        Load the image and do inference
    '''
    def loadAndDoInference(self, image_path):
        image = cv2.imread(image_path)
        return self.doInference(image)

    '''
        Get the points
    '''
    def getPoints(self):
        return self.points
