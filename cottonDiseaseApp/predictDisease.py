import numpy as np

import cv2

from keras.models import load_model


def imagePred(imageName):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # class_dict = {0: 'diseased cotton leaf',
    #               1: 'diseased cotton plant',
    #               2: 'fresh cotton leaf',
    #               3: 'fresh cotton plant'}
    class_dict = {0: 'Cotton leaf is Diseased.',
                  1: 'Cotton plant is Diseased.',
                  2: 'Its a fresh Cotton leaf',
                  3: 'Its a fresh Cotton plant'}
    print(imageName, "------")
    # os.chdir("./media")
    print(os.getcwd())
    model = load_model("D:/django projects/cottonDiseaseProject/media/diseasePredictionModel.h5")
    test_image = cv2.imread(
        "D:/django projects/cottonDiseaseProject" + imageName)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (224, 224),
                            interpolation=cv2.INTER_CUBIC)
    # plt.imshow(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    probs = model.predict(test_image)
    pred_class = np.argmax(probs)

    pred_class = class_dict[pred_class]

    return (pred_class)
