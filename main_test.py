
import joblib
from helper_function import processing, hog_extraction, flatten_extraction, load_data

## 保存好的模型地址
model_path = ""

## 数据位置
data_path = "./data/pneumoniamnist.npz"

def compute_metric(pred, labels):
    ## TODO  自行根据pred与labels计算相应的几种指标（比如acc, precision, recall, f1等），并进行返回
    acc =
    precision =
    recall =
    f1 =

    return acc, precision, recall, f1

if __name__ == '__main__':
    ## TODO  在test data上进行测试，得到最终的指标
    ## 经过了上面调参的过程，得到了不同的模型，在validate data上的结果也各不相同，
    ## 我们需要使用其中最好的模型，在test data 上进行测试，得到最终的指标(acc, precision, recall, f1 ...)。

    _, _, _, _, test_images, test_labels = load_data(data_path)

    ## 测试数据预处理
    test_images = processing(test_images)

    ## 测试数据提取特征，这里可以换成别的特征，根据效果去自由切换
    test_images = flatten_extraction(test_images)

    ## 加载训练好的模型。
    model = joblib.load(model_path)

    ## 得到预测结果
    pred = model.predict(test_images)

    ## 计算指标，传入模型预测结果与labels
    acc, precision, recall, f1 = compute_metric(pred, test_labels)

    print(f"acc is {acc}\n precision is {precision}\n recall is {recall}\n f1 is {f1} \n")