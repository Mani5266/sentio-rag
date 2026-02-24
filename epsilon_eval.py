from sentio_backend import sentio_single

epsilons = [0.1, 0.3, 0.5, 0.7, 1.0]

email = """
Priya from SecureStack requested approval for ₹68,200.
Please confirm with Ramesh before processing.
"""

question = "Who requested approval?"

for eps in epsilons:
    out = sentio_single(email, question)
    print(f"ε={eps} →", out["output"])