# 客服 FAQ 语义检索功能说明

## 功能描述

客服 FAQ 语义检索用于在 `customer_service` 回答链路中优先匹配 `combined_qa_generic_optimized.json` 里的标准问答。功能开启后，系统会按下面顺序执行：

1. Query2Query：用用户当前问题的 embedding 检索 FAQ 的 `question` 向量库。
2. Query2Query 高置信命中时，直接返回该历史问题对应的 `answer`。
3. Query2Query 未命中时，执行 Query2Answer：用用户当前问题的 embedding 检索 FAQ 的 `answer` 向量库。
4. Query2Answer 高置信命中时，直接返回命中的答案。
5. 两个策略都未命中时，回退原来的 `direct_llm` 客服回答链路。


## 配置方式

在 `mm_customer_agent_simple/.env` 中配置：

```env
ENABLE_CUSTOMER_QA_RETRIEVAL=true
CUSTOMER_QA_PATH=combined_qa_generic_optimized.json
CUSTOMER_QA_QDRANT_PATH=storage5/customer_qa_qdrant
CUSTOMER_QA_QUERY_COLLECTION=customer_qa_query
CUSTOMER_QA_ANSWER_COLLECTION=customer_qa_answer
CUSTOMER_QA_VECTOR_TOP_K=3
CUSTOMER_QA_MIN_VECTOR_SCORE=0.35
CUSTOMER_QA_MIN_VECTOR_SCORE_GAP=1.1
```

参数说明：

| 参数                               | 说明                                                         |
| ---------------------------------- | ------------------------------------------------------------ |
| `ENABLE_CUSTOMER_QA_RETRIEVAL`     | 总开关。`true` 启用 FAQ 语义检索，`false` 完全走原客服链路。 |
| `CUSTOMER_QA_PATH`                 | FAQ 标准问答 JSON 文件路径。                                 |
| `CUSTOMER_QA_QDRANT_PATH`          | FAQ 独立向量库路径，不要和手册 Qdrant 路径混用。             |
| `CUSTOMER_QA_QUERY_COLLECTION`     | Query2Query 使用的 question 向量 collection。                |
| `CUSTOMER_QA_ANSWER_COLLECTION`    | Query2Answer 使用的 answer 向量 collection。                 |
| `CUSTOMER_QA_VECTOR_TOP_K`         | 每个策略返回的向量候选数量。                                 |
| `CUSTOMER_QA_MIN_VECTOR_SCORE`     | 最低向量相似度阈值。低于该值不直接命中 FAQ。                 |
| `CUSTOMER_QA_MIN_VECTOR_SCORE_GAP` | 第一候选与第二候选的分差阈值。用于避免多个候选过近时误命中。 |

命中条件：

```text
top_vector_score >= CUSTOMER_QA_MIN_VECTOR_SCORE
并且 top_vector_score / second_vector_score >= CUSTOMER_QA_MIN_VECTOR_SCORE_GAP
```

如果只有一个候选，则只检查 `CUSTOMER_QA_MIN_VECTOR_SCORE`。

## 使用方式

所有命令建议在 `mm_customer_agent_simple` 目录下执行，确保 `.env` 能被正确加载：


### 1. 构建 FAQ 向量索引

首次使用，或者 `combined_qa_generic_optimized.json` 更新后，需要重新构建：

```powershell
python ingest_customer_qa.py
```

成功时会看到类似输出：

```text
读取客服 FAQ: ...\combined_qa_generic_optimized.json
原始条数: 348; 去重 question: 65; 索引条数: 283
FAQ Qdrant 独立路径: storage5/customer_qa_qdrant; collections=customer_qa_query, customer_qa_answer
生成 Query2Query question embeddings...
生成 Query2Answer answer embeddings...
重建客服 FAQ Qdrant collections...
客服 FAQ embedding 索引构建完成。
manifest 已写入: storage5/customer_qa_qdrant/customer_qa_manifest.json
```

### 2. 单条问题测试

```powershell
python answer.py --question "如何查看我的订单状态？" --debug
```

如果 Query2Query 命中，会看到：

```text
[customer_qa] enabled=true mode=embedding path=... entries=348 deduped=65 indexed=283 ready=true
[customer_qa][turn=1/1] query='如何查看我的订单状态？'
[query2query][turn=1/1] status=hit reason=threshold_pass top_vector_score=1.0000 second_vector_score=0.6926
[query2query][turn=1/1] candidate#1 vector_score=1.0000 matched_question='如何查看我的订单状态？' mapped_answer='登录账户后进入...'
[query2answer][turn=1/1] skipped reason=query2query_hit
[customer_qa][turn=1/1] final=faq_answer source=query2query
```

### 3. 批量 sample 测试

```powershell
python batch_answer.py `
  --input ..\question_public.csv `
  --output ..\question_public_output_q2qa_sample.csv `
  --start-id 1 `
  --end-id 5 `
  --context-output ..\question_public_context_q2qa_sample.txt `
  --trace-output ..\question_public_trace_q2qa_sample.jsonl
```

### 4. 全量运行

```powershell
python batch_answer.py `
  --input ..\question_public.csv `
  --output ..\question_public_output_q2qa.csv `
  --context-output ..\question_public_context_q2qa.txt `
  --trace-output ..\question_public_trace_q2qa.jsonl
```

中断后可以续跑：

```powershell
python batch_answer.py `
  --input ..\question_public.csv `
  --output ..\question_public_output_q2qa.csv `
  --context-output ..\question_public_context_q2qa.txt `
  --trace-output ..\question_public_trace_q2qa.jsonl `
  --resume
```

## 结果示例

### Query2Query 命中示例

样例来源：`question_public.csv` 的 `id=2`，第一轮问题被改写为：

```text
我想咨询一下，你们的售后维修服务范围是什么？
```

终端和 trace 中的关键检索结果：

```text
[customer_qa][turn=1/2] query='我想咨询一下，你们的售后维修服务范围是什么？'
[query2query][turn=1/2] status=hit reason=threshold_pass top_vector_score=0.8573 second_vector_score=0.7634
[query2query][turn=1/2] candidate#1 vector_score=0.8573 matched_question='商品售后维修服务范围包括哪些？' mapped_answer='售后维修通常覆盖商品在正常使用情况下出现的质量故障...'
[query2answer][turn=1/2] skipped reason=query2query_hit
[customer_qa][turn=1/2] final=faq_answer source=query2query
```

最终返回的 FAQ 答案：

```text
售后维修通常覆盖商品在正常使用情况下出现的质量故障。若商品仍在质保期内，且经检测属于非人为质量问题，一般可享受免费维修。具体维修范围以商品质保政策和售后检测结果为准。
```

### Query2Answer 命中示例

样例来源：`question_public.csv` 的 `id=2`，第二轮问题被改写为：

```text
人为损坏的，能维修吗？维修费用怎么算？
```

这轮 Q2Q 的前两名候选分数接近，因此没有直接命中；随后 Q2A 命中答案内容：

```text
[customer_qa][turn=2/2] query='人为损坏的，能维修吗？维修费用怎么算？'
[query2query][turn=2/2] status=miss reason=score_gap_below_threshold top_vector_score=0.7764 second_vector_score=0.7415 next=query2answer
[query2query][turn=2/2] candidate#1 vector_score=0.7764 matched_question='商品人为损坏后的维修费用如何计算？' mapped_answer='人为损坏或超过质保期的维修费用通常会根据检测结果...'
[query2answer][turn=2/2] status=hit reason=threshold_pass top_vector_score=0.8450 second_vector_score=0.7561
[query2answer][turn=2/2] candidate#1 vector_score=0.8450 matched_answer='人为损坏通常不属于免费质保范围，但一般可以联系售后协助付费维修...' source_question='商品出现人为损坏是否可以申请维修？'
[customer_qa][turn=2/2] final=faq_answer source=query2answer
```

最终返回的 FAQ 答案：

```text
人为损坏通常不属于免费质保范围，但一般可以联系售后协助付费维修。人为损坏包括进水、摔坏、私自拆修、使用不当等情况，具体是否可修需要以检测结果为准。
```

### 回退 direct LLM 示例

样例来源：`question_public.csv` 的 `id=1`，第一轮问题被改写为：

```text
请问你们家的商品支持7天无理由退换货吗？
```

这轮 Q2Q 和 Q2A 都召回了相关候选，但第一名和第二名分数过近，没有通过 `CUSTOMER_QA_MIN_VECTOR_SCORE_GAP=1.1` 的高置信条件，因此回退原客服链路：

```text
[customer_qa][turn=1/2] query='请问你们家的商品支持7天无理由退换货吗？'
[query2query][turn=1/2] status=miss reason=score_gap_below_threshold top_vector_score=0.8575 second_vector_score=0.8450 next=query2answer
[query2query][turn=1/2] candidate#1 vector_score=0.8575 matched_question='商品是否支持7天无理由退换货？' mapped_answer='一般情况下，符合7天无理由退换货条件的商品可以申请退换货...'
[query2answer][turn=1/2] status=miss reason=score_gap_below_threshold top_vector_score=0.7799 second_vector_score=0.7658 next=direct_llm fallback=direct_llm
[query2answer][turn=1/2] candidate#1 vector_score=0.7799 matched_answer='一般情况下，符合7天无理由退换货条件的商品可以申请退换货...' source_question='商品是否支持7天无理由退换货？'
[customer_qa][turn=1/2] final=direct_llm reason=no_confident_faq_hit
```

此时会走原客服链路：

```text
analyze_customer_turn + generate_direct
```

该样例说明：有候选不等于直接命中。只有候选达到最低相似度，并且第一名明显优于第二名，才会直接返回 FAQ。

## Trace 字段说明

批量运行时，`question_public_trace_q2qa.jsonl` 中会包含 `customer_qa_retrieval`。下面是 `id=2` 第二轮 Query2Answer 命中的真实 trace 摘要：

```json
{
  "turn_index": 2,
  "turn_total": 2,
  "original_query": "如果是人为损坏的，能维修吗？维修费用怎么算？",
  "final_source": "query2answer",
  "final_reason": "query2answer_hit",
  "query2query": {
    "hit": false,
    "reason": "score_gap_below_threshold",
    "top_vector_score": 0.7763970690163402,
    "second_vector_score": 0.7415136988198475,
    "candidates": [
      {
        "rank": 1,
        "vector_score": 0.7763970690163402,
        "matched_question": "商品人为损坏后的维修费用如何计算？",
        "mapped_answer": "人为损坏或超过质保期的维修费用通常会根据检测结果、配件成本和维修工时计算。客服或维修方会在维修前告知报价，并在你确认后再继续处理。"
      }
    ]
  },
  "query2answer": {
    "hit": true,
    "reason": "threshold_pass",
    "top_vector_score": 0.8450231708748372,
    "second_vector_score": 0.7561429243100896,
    "candidates": [
      {
        "rank": 1,
        "vector_score": 0.8450231708748372,
        "matched_answer": "人为损坏通常不属于免费质保范围，但一般可以联系售后协助付费维修。人为损坏包括进水、摔坏、私自拆修、使用不当等情况，具体是否可修需要以检测结果为准。",
        "source_question": "商品出现人为损坏是否可以申请维修？"
      }
    ]
  }
}
```

字段含义：

| 字段                  | 说明                                                      |
| --------------------- | --------------------------------------------------------- |
| `final_source`        | 最终来源：`query2query`、`query2answer` 或 `direct_llm`。 |
| `final_reason`        | 最终选择原因。                                            |
| `top_vector_score`    | 第一候选向量相似度。                                      |
| `second_vector_score` | 第二候选向量相似度。                                      |
| `matched_question`    | Q2Q 命中的历史标准问题。                                  |
| `mapped_answer`       | Q2Q 标准问题对应的答案。                                  |
| `matched_answer`      | Q2A 命中的答案内容。                                      |
| `source_question`     | Q2A 命中答案对应的原始问题。                              |
