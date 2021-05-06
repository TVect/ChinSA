import os


class ABSADataset:

    def __init__(self, dataset):
        basedir = os.path.dirname(__file__)
        if dataset in ["camera", "car", "notebook", "car"]:
            self.train_data = self.load_data(
                os.path.join(basedir, f"{dataset}/{dataset}.train.txt"))
            self.test_data = self.load_data(
                os.path.join(basedir, f"{dataset}/{dataset}.test.txt"))
        elif dataset == "all":
            self.train_data, self.test_data = [], []
            for name in ["camera", "car", "notebook", "car"]:
                self.train_data.extend(
                    self.load_data(os.path.join(basedir, f"{name}/{name}.train.txt")))
                self.test_data.extend(
                    self.load_data(os.path.join(basedir, f"{name}/{name}.test.txt")))

    def get_labels(self):
        return ["negative", "positive"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def load_data(self, fname):
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fr:
            lines = fr.readlines()
        
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = int(lines[i + 2].strip())
            polarity = "negative" if polarity == -1 else "positive"

            all_data.append({"text_left": text_left, 
                             "text_right": text_right, 
                             "aspect": aspect, 
                             "polarity": polarity})
        return all_data


if __name__ == "__main__":
    ds = ABSADataset("camera")
    print(f"[data size] train: {len(ds.train_data)}. test: {len(ds.test_data)}")

    print(ds.train_data[0])
    print(ds.test_data[0])
    import IPython
    IPython.embed()
