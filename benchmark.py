from sentio_backend import sentio_single

# Mini evaluation set (extend later)
dataset = [
    {
        "email": """Hi Team,

John from Alpha Corp requested approval for the $12,500 invoice.

Please confirm with Sarah before processing.

Thanks""",
        "question": "Who requested approval?",
        "answer": "John"
    },
    {
        "email": """Hi,

Priya from SecureStack submitted the proposal.

Delivery is expected March 20.

Regards""",
        "question": "When is delivery expected?",
        "answer": "March"
    }
]

def simple_score(pred, gold):
    return gold.lower() in pred.lower()

raw_correct = 0
masked_correct = 0

for sample in dataset:

    raw = sentio_single(sample["email"], sample["question"])
    masked = sentio_single(sample["email"], sample["question"])

    if simple_score(raw["output"], sample["answer"]):
        raw_correct += 1

    if simple_score(masked["output"], sample["answer"]):
        masked_correct += 1

print("Raw QA Accuracy:", raw_correct / len(dataset))
print("Masked QA Accuracy:", masked_correct / len(dataset))
print("Utility Loss:", (raw_correct - masked_correct) / len(dataset))