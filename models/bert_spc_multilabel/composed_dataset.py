import random
from itertools import product
from datasets.data_loader import ABSADataset


def get_composed_data(positives, negatives, random_threshold=0.2):
    random.seed(10)
    all_punctuation = '!"#$%&+,-.;=@~。，'
    composed_records = []
    for item in product(positives, negatives):
        if random.random() > random_threshold:
            continue
        if item[0]["aspect"] == item[1]["aspect"]:
            if random.random() < 0.5:
                text = f"{item[0]['text']} {random.choice(all_punctuation)} {item[1]['text']}"
            else:
                text = f"{item[1]['text']} {random.choice(all_punctuation)} {item[0]['text']}"
            composed_records.append(
                {"text": text, 
                "aspect": item[1]["aspect"], 
                "has_positive": (item[0]["polarity"] == "positive") or (item[1]["polarity"] == "positive"),
                "has_negative": (item[0]["polarity"] == "negative") or (item[1]["polarity"] == "negative")
                })
        else:
            if random.random() < 0.5:
                text = f"{item[0]['text']} {random.choice(all_punctuation)} {item[1]['text']}"
            else:
                text = f"{item[1]['text']} {random.choice(all_punctuation)} {item[0]['text']}"
            composed_records.append(
                {"text": text, 
                "aspect": item[0]["aspect"], 
                "has_positive": item[0]["polarity"] == "positive",
                "has_negative": item[0]["polarity"] == "negative"})
            composed_records.append(
                {"text": text, 
                "aspect": item[1]["aspect"], 
                "has_positive": item[1]["polarity"] == "positive",
                "has_negative": item[1]["polarity"] == "negative"})
    return composed_records


class ComposedABSADataset:

    def __init__(self, dataset):
        self.raw_data = ABSADataset(dataset)
        self.train_data = self.initialize(self.raw_data.train_data)
        self.test_data = self.initialize(self.raw_data.test_data)
    
    def get_labels(self):
        return [False, True]

    def initialize(self, records):
        text_label = {}
        for item in records:
            aspect = item['aspect']
            text = f"{item['text_left']} {item['aspect']} {item['text_right']}"
            polarity = item['polarity']
            if text not in text_label:
                text_label[text] = {}
            text_label[text][aspect] = polarity

        records_multi_aspect = []
        records_single_aspect = []
        for text, value in text_label.items():
            if len(value) > 1:
                for aspect, polarity in value.items():
                    records_multi_aspect.append(
                        {"text": text, "aspect": aspect, "polarity": polarity})
            else:
                for aspect, polarity in value.items():
                    records_single_aspect.append(
                        {"text": text, "aspect": aspect, "polarity": polarity})

        pos_examples = [example for example in records_single_aspect if example["polarity"] == "positive" and len(example["text"]) < 60]
        neg_examples = [example for example in records_single_aspect if example["polarity"] == "negative" and len(example["text"]) < 60]

        new_records = []
        for dataset in [pos_examples, neg_examples]:
            for record in dataset:
                new_records.append(
                    {"text": record["text"], 
                    "aspect": record["aspect"], 
                    "has_positive": record["polarity"] == "positive",
                    "has_negative": record["polarity"] == "negative"})
        new_records.extend(
            get_composed_data(pos_examples, neg_examples, random_threshold=0.003))
        
        return new_records


if __name__ == "__main__":
    ds = ComposedABSADataset("all")
    print(f"train_data size: {len(ds.train_data)}. test_data size: {len(ds.test_data)}")
