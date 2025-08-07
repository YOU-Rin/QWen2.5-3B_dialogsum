from rouge_score import rouge_scorer

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
from rouge_score import rouge_scorer

def dialogsum_reward_func(completions, response, **kwargs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rewards = []
    for pred, ref in zip(completions, response):
        rewards.append(scorer.score(ref, pred)["rougeL"].fmeasure)
    return rewards


def rouge_reward_func(prompts, completions, reference_summaries, **kwargs) -> list[float]:
    """
    计算每个生成摘要与参考摘要的 Rouge-L 分数作为奖励。
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = [
        rouge.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(reference_summaries, responses)
    ]
    return rewards

def length_penalty_reward_func(completions, min_len=20, max_len=100, **kwargs) -> list[float]:
    """
    生成摘要长度越接近目标长度（如50-100字），奖励越高，超过则惩罚。
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        l = len(r.split())
        if min_len <= l <= max_len:
            rewards.append(0.5)  # 奖励合适长度
        else:
            rewards.append(0.0)  # 惩罚过短/过长
    return rewards

def repetition_penalty_reward_func(completions, **kwargs) -> list[float]:
    """
    惩罚重复 n-gram 的摘要。
    """
    def has_repetition(text, n=3):
        tokens = text.split()
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            if ngram in ngrams:
                return True
            ngrams.add(ngram)
        return False

    responses = [completion[0]["content"] for completion in completions]
    rewards = [0.0 if has_repetition(r) else 0.5 for r in responses]
    return rewards

def dialogsum_reward(prompts, completions, reference_summaries, **kwargs) -> list[float]:
    rouge_rewards = rouge_reward_func(prompts, completions, reference_summaries)
    length_rewards = length_penalty_reward_func(completions)
    rep_penalties = repetition_penalty_reward_func(completions)

    # 加权组合
    rewards = [
        1.0 * rouge_r + 0.3 * length_r + 0.3 * rep_r
        for rouge_r, length_r, rep_r in zip(rouge_rewards, length_rewards, rep_penalties)
    ]
    return rewards
