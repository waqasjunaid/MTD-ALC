diagnosis = json.load(open("outputs/diagnosis_report.json"))
from repair.inference_correction import corrected_score

learn_corrected = corrected_score(learn, diagnosis)
T = w1*syntax + w2*dep + w3*learn_corrected
