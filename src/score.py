def campaign_score(similarity, burst, spam_ratio):
    score = (
        similarity * 40 +
        (30 if burst else 0) +
        spam_ratio * 30
    )
    return round(min(score, 100), 2)

def explain_campaign(similarity, burst, spam_ratio):
    reasons = []

    if similarity > 0.3:
        reasons.append("High repeated / similar comments detected")

    if burst:
        reasons.append("Sudden comment burst detected")

    if spam_ratio > 0.3:
        reasons.append("High spam percentage in comments")

    return reasons
