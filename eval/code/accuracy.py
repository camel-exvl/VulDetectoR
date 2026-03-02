import csv
import re
import numpy as np
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


def read_csv(file_path):
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            generated = row.get('generated')
            reference = row.get('reference')
            code = row.get('code')
            if generated is not None and reference is not None:
                data.append([code, generated, reference])
    return data


y_predict = []
y_label = []
label_map = {'non-vulnerable': 0, 'vulnerable': 1}

file_path = './result/RQ1_3.csv'
output_path = './result/accuracy_RQ1_3.csv'


def list_to_csv_with_headers(list, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['code', 'generated', 'reference', 'label', 'predict', 'cwehit'])
        csv_writer.writerows(list)


result_list = []
result = read_csv(file_path)
pat_label = re.compile(r'\[label\]\s*\n\s*([^\n.]+)')
pat_cwe = re.compile(r'\[cwe\]\s*\n[^\n]*?\[([^\[\]]*)\]')
cnt_hit = 0
cnt_eqa = 0
cnt_all = 0
for item in result:
    code = item[0]
    generated_result = item[1].lower()
    reference_result = item[2].lower()
    predict_label = pat_label.search(generated_result)
    true_label = pat_label.search(reference_result)
    if predict_label is not None and true_label is not None:
        try:
            predict_label = label_map[predict_label.group(1).split(" ")[-1]]
            true_label = label_map[true_label.group(1).split(" ")[-1]]
        except:
            print(predict_label.group(1))
            print(true_label.group(1))
            continue
        cwe_hit = -1
        cwe_p = pat_cwe.search(generated_result)
        cwe_t = pat_cwe.search(reference_result)
        if cwe_p is not None and cwe_t is not None:
            cwe_p = set(cwe_p.group(1).replace(" ", "").replace("'", "").split(','))
            cwe_t = set(cwe_t.group(1).replace(" ", "").replace("'", "").split(','))
            if len(cwe_t) > 0:
                cnt_all += 1
                cwe_hit = 0
                if cwe_p.issubset(cwe_t):
                    cnt_hit += 1
                    cwe_hit = 1
                if cwe_p & cwe_t == cwe_t:
                    cnt_eqa += 1
                    cwe_hit = 1
        result_list.append([code, generated_result, reference_result, true_label, predict_label, cwe_hit])
        y_predict.append(predict_label)
        y_label.append(true_label)

list_to_csv_with_headers(result_list, output_path)

y_pred = np.array(y_predict)
y_true = np.array(y_label)
print("Valid data number:", len(y_predict))

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

overall_accuracy = accuracy_score(y_true, y_pred)
print("Overall accuracy:", overall_accuracy)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
print("Precision per class:", precision)
print("Recall per class:", recall)
print("F1 Score per class:", f1)

if cnt_all > 0:
    print("CWE Hit Ratio:", cnt_hit / cnt_all)
    print("CWE Equality Ratio:", cnt_eqa / cnt_all)

precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                                      average='weighted')
print("Weighted Precision:", precision_weighted)
print("Weighted Recall:", recall_weighted)
print("Weighted F1 Score:", f1_weighted)
