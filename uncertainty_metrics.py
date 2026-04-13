"""
extract_logit_entropy(model, tokenizer, hidden_states, direction, cases, strategy, figure_dir=None)

extract_logit_margin(model, tokenizer, hidden_states, direction, cases, strategy, figure_dir=None) 

extract_attention_entropy(model, hidden_states, direction, cases, strategy, figure_dir=None)

extract_logit_lens(model, tokenizer, hidden_states, direction, cases, strategy, figure_dir=None)

extract_tuned_lens(model, tokenizer, hidden_states, direction, cases, strategy, figure_dir=None)


Description:
extract_logit_entropy(model, tokenizer, cases)
  -> 답 생성 직전 softmax 분포의 entropy

extract_logit_margin(model, tokenizer, cases)
  -> top1 - top2 확률 차이

extract_attention_entropy(model, tokenizer, cases)
  -> context 토큰에 대한 attention entropy (레이어별)

extract_logit_lens(model, tokenizer, cases)
  -> 각 레이어 hidden state -> lm_head 통과 -> 예측 토큰 분포
  
"""