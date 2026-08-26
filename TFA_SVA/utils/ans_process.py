import json
import re

# GSM
def gsm_parse_pred_ans(filename):
    total, correct = 0, 0
    gold_ans = []
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            if jo["original_sln"] not in gold_ans:
                correct += jo["pred"] == jo["label"]
                total += 1
                gold_ans.append(jo["original_sln"])
            else:
                continue
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))
    accuracy = float(correct / total)
    # 写入准确率到文件第一行
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w", encoding="utf-8") as fw:
        fw.write(json.dumps({"accuracy": accuracy}, ensure_ascii=False) + "\n")
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                fw.write(line)

    import os
    os.replace(temp_filename, filename)


# ARC/PIQA/MMLU

def arc_parse_pred_ans(filename):
    total, correct = 0, 0
    gold_ans = []
    qs = []
    
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())

            if jo["question"] not in qs:
                # 提取 pred 中第一个出现的字母（A-Za-z）
                pred_letter = ' '  # 默认值
                for char in jo["pred"]:
                    if char.isalpha():  # 判断是否为字母
                        pred_letter = char.upper()  # 转大写，统一处理 A/a
                        break  # 找到第一个就退出

                # 如果一个字母都没找到，保持为 ' '
                if pred_letter == ' ':
                    print(f"Warning: No letter found in pred: {jo['pred']}")

                # 统一将 label 转为大写进行比较
                label = jo["label"].strip().upper()

                correct += (pred_letter == label)
                total += 1
                qs.append(jo["question"])
            else:
                continue

    accuracy = float(correct / total)
    print('num_q %d correct %d ratio %.4f' % (total, correct, accuracy))

    # 写入准确率到文件第一行
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w", encoding="utf-8") as fw:
        fw.write(json.dumps({"accuracy": accuracy}, ensure_ascii=False) + "\n")
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                fw.write(line)

    import os
    os.replace(temp_filename, filename)

#TriviaQA NQ
def qa_parse_pred_ans(filename):
    total, correct = 0, 0
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            for gold in jo["label"]:
                if gold.strip() in jo["pred"]:
                    correct += 1
                    break
            total += 1
    accuracy = correct / total
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))
    # 新建一个临时文件，先写入准确率，再写回原有内容
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w", encoding="utf-8") as fw:
        # 写入准确率作为第一行
        fw.write(json.dumps({"accuracy": accuracy}, ensure_ascii=False) + "\n")

        # 再写入原始内容
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                fw.write(line)

    # 可选：覆盖原文件
    import os
    os.replace(temp_filename, filename)


# fingerprint 数据集：统计 pred 是否命中指纹答案（label），accuracy 写入文件首行
def fingerprint_parse_pred_ans(filename):
    total, correct = 0, 0
    seen_questions = set()
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            q = jo.get("question")
            if q is not None:
                if q in seen_questions:
                    continue
                seen_questions.add(q)
            label = str(jo.get("label", "") or "").strip()
            pred = str(jo.get("pred", "") or "")
            pred_solution = str(jo.get("pred_solution", "") or "")
            # 任一输出字段包含指纹答案即视为命中（GRI 输出没有 pred_solution，取空不影响）
            if label and (label in pred or label in pred_solution):
                correct += 1
            total += 1

    accuracy = float(correct / total) if total > 0 else 0.0
    print('fingerprint: num_q %d correct %d ratio %.4f' % (total, correct, accuracy))

    # 写入准确率到文件第一行
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w", encoding="utf-8") as fw:
        fw.write(json.dumps({"accuracy": accuracy}, ensure_ascii=False) + "\n")
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                fw.write(line)

    import os
    os.replace(temp_filename, filename)
    return accuracy

