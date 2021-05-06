import optparse
import tqdm
from client import ServingClient
from datasets.data_loader import ABSADataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def test_dataset(client, dataset):
    ds = ABSADataset(dataset)

    for attr in ["train_data", "test_data"]:
        trues, preds = [], []
        for item in tqdm.tqdm(getattr(ds, attr)):
            output = client.predict(
                in_text=item["text_left"] + item["aspect"] + item["text_right"], 
                in_aspect=item["aspect"])
            item["prob"] = output["polarity"][0]
            trues.append(item["polarity"])
            preds.append("positive" if item["prob"] > 0.5 else "negative")
        acc = accuracy_score(y_true=trues, y_pred=preds)
        print(f"\n============ {attr} ============ acc: {acc}.\n")
        print(classification_report(y_true=trues, y_pred=preds))
        print(confusion_matrix(y_true=trues, y_pred=preds))


if __name__ == "__main__":
    parser = optparse.OptionParser(usage='"usage:%prog [options] arg1,arg2"')
    parser.add_option('-m', '--model_path', dest='model_path', action='store',
                      type=str, help='path to model: e.g. output/1560850786')
    parser.add_option('-p', '--processor_file', dest='processor_file', action='store',
                      type=str, help='path to processor_file: e.g. output/data_processor.json')
    parser.add_option('-d', '--dataset', dest='dataset', action='store',
                      type=str, help='camera | car | notebook | phone | ...')

    options, args = parser.parse_args()
    
    client = ServingClient(model_path=options.model_path, 
                           processor_file=options.processor_file)
    import IPython
    IPython.embed()
    test_dataset(client, options.dataset)
