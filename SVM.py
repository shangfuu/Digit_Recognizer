import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

def show_digits():
    for key, value in digits.items() :
        try:
            print(key, value.shape)
        except:
            print(key)

    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:2]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)
        plt.show()  # show image

# 將混淆矩陣圖示
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    import numpy as np
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(digits.target_names))
    plt.xticks(tick_marks, digits.target_names, rotation=45)
    plt.yticks(tick_marks, digits.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":

    # The digits dataset
    digits = datasets.load_digits()
    n_samples = len(digits.images)

    # 資料攤平:1797 x 8 x 8 -> 1797 x 64
    # 這裏的-1代表自動計算，相當於 (n_samples, 64)
    data = digits.images.reshape((n_samples, -1))

    # 產生SVC分類器
    classifier = svm.SVC(gamma=0.001)
    # 用前半部份的資料來訓練
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
    # 利用後半部份的資料來測試分類器，共 899筆資料
    predicted = classifier.predict(data[n_samples // 2:])
    expected = digits.target[n_samples // 2:]

    # 混淆矩陣
    print("Confusion matrix:\n%s"
        % metrics.confusion_matrix(expected, predicted))
    plt.figure()
    plot_confusion_matrix(metrics.confusion_matrix(expected, predicted))
    plt.show()  # show confusion_matrix

    # 統計數據
    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))

    # 觀察預測與實際結果
    images_and_predictions = list(
                            zip(digits.images[n_samples // 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)

    plt.savefig('resource/digits')
    plt.show()

