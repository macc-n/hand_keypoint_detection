from hand_keypoint import *


def main():
    image_path = "test.jpg"
    hkp = HandKeyPoint()
    result = hkp.loadAndDoInference(image_path)
    print(result)


if __name__ == "__main__":
    main()
