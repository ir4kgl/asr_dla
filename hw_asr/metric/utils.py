import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1 if len(predicted_text) != 0 else 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)

def calc_wer(target_text, predicted_text) -> float:
    target_splitted = list(target_text.split(' '))
    pred_splitted = list(predicted_text.split(' '))
    if len(target_splitted) == 0:
        return 1 if len(pred_splitted) != 0 else 0
    return editdistance.eval(target_splitted, pred_splitted) / len(target_splitted)
