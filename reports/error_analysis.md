# RAG Error Analysis

This report identifies the rows most useful for improving the RAG system after the dev, holdout, and stress evaluations.

## Executive Summary

- Total evaluated rows: 80
- False refusals: 7
- Rows with hallucination rate > 0: 6
- Rows with citation strict accuracy < 1: 12
- Rows with context recall < 0.5: 19

## Likely Root Causes

- Table and numeric questions are the clearest remaining retrieval weakness.
- Claim-verification rows expose false refusals when the system should classify a claim as supported, unsupported, or contradicted.
- Context precision is lower than evidence hit rate, which means the retriever often finds evidence but also passes extra irrelevant chunks to the generator.
- Refusal behavior is strong on true out-of-scope rows, but the policy is too conservative on some answerable stress rows.

## False Refusals

Rows where the model refused even though the dataset expected an answer or claim verification.

| Split | Type | Question | Metrics | Answer Snippet |
|---|---|---|---|---|
| dev | result_finding | What does Table 3 say about informality by gender? | false_refusal=1.0000, refusal=1.0000, context_recall=0.0000, answer_relevancy=0.0000 | The answer was not found in the provided context. The context mentions Tables 9, 10-13, and D.9 but does not include a Table 3. Therefore, I cannot provide information from Table 3 as requested. [S1] discusses informalit |
| dev | result_finding | What does Table 9 imply about informal firms? | false_refusal=1.0000, refusal=1.0000, context_recall=0.0000, answer_relevancy=0.0000 | The answer was not found in the provided context. The context mentions Tables 15 and 16 but does not reference Table 9. Therefore, I cannot provide an implication about informal firms based on Table 9 from the given info |
| holdout | method | What happens to informal firms when they are caught by authorities? | false_refusal=1.0000, refusal=1.0000, context_recall=0.0000, answer_relevancy=0.0000 | When informal firms are caught by authorities, they are subject to fines [S2]. The context does not provide specific details on the exact nature or amount of these fines. |
| holdout | result_finding | What happens to unemployment in Sao Paulo low-education males when informality costs rise by 10 percent? | false_refusal=1.0000, refusal=1.0000, context_recall=0.0000, answer_relevancy=0.0000 | The context does not explicitly state what happens to unemployment in Sao Paulo for low-education males when the cost of informality rises by 10 percent. The provided tables focus on productivity and wages but do not inc |
| stress | claim_verification | Verify this claim: workers in the model only search when unemployed. | false_refusal=1.0000, refusal=1.0000, context_recall=0.0000, answer_relevancy=0.0000 | The context provided does not explicitly state that "workers in the model only search when unemployed." However, it mentions that "the economy is subject to search frictions and workers search both when unemployed and wh |
| stress | claim_verification | Verify this claim: the paper studies only high education workers. | false_refusal=1.0000, refusal=1.0000, context_recall=0.0000, answer_relevancy=0.0000 | The answer was not found in the context provided. The context indicates that low education workers were focused on, defined as those with eight or less years of education [S2]. Therefore, the claim that the paper studies |
| stress | claim_verification | Verify this claim: in Sao Paulo low-education males, a 10 percent increase in informality costs reduces total welfare. | false_refusal=1.0000, refusal=1.0000, context_recall=0.0000, answer_relevancy=0.0000 | The claim that a 10 percent increase in informality costs reduces total welfare for low-education males in São Paulo is not directly supported by the provided context. The context indicates that increasing the cost of in |

## Lowest Context Recall

Rows where RAGAS judged the retrieved contexts as missing reference evidence.

| Split | Type | Question | Metrics | Answer Snippet |
|---|---|---|---|---|
| dev | result_finding | What does the paper find about unemployment-to-job offers? | context_recall=0.0000, context_precision=0.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The paper finds that unemployed workers receive job offers according to a Poisson process with an arrival rate denoted as λij, where i = 0 (unemployed) and j = 1 or 2 denote offers from the formal or informal sectors res |
| dev | result_finding | What does Table 9 imply about informal firms? | context_recall=0.0000, context_precision=0.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The answer was not found in the provided context. The context mentions Tables 15 and 16 but does not reference Table 9. Therefore, I cannot provide an implication about informal firms based on Table 9 from the given info |
| dev | result_finding | What does Table 3 say about informality by gender? | context_recall=0.0000, context_precision=0.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The answer was not found in the provided context. The context mentions Tables 9, 10-13, and D.9 but does not include a Table 3. Therefore, I cannot provide information from Table 3 as requested. [S1] discusses informalit |
| holdout | result_finding | What happens to total welfare in Sao Paulo low-education males after a 10 percent increase in informality costs? | context_recall=0.0000, context_precision=1.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The overall welfare of workers in Sao Paulo for low-education males increases after a 10 percent increase in the cost of informality [S2, S3]. This is primarily due to an increase in formal sector wages and a relocation |
| stress | true_out_of_scope | What is the current stock price of Apple today? | context_recall=0.0000, context_precision=0.0000, evidence_hit_rate=0.0000, mrr=0.0000 | The answer was not found in the provided context. [S1], [S2], [S3], and [S4] contain statistical data related to wages, firm revenue, and government revenue but do not provide information about the current stock price of |
| holdout | reasoning | Why are high-productivity firms more likely to operate formally? | context_recall=0.0000, context_precision=0.3333, evidence_hit_rate=1.0000, mrr=1.0000 | High-productivity firms are more likely to operate formally because as productivity increases, wages also increase. This leads to higher revenues from fines in the informal sector for formal firms, allowing them to captu |
| holdout | reasoning | Why can low-productivity firms operate informally? | context_recall=0.0000, context_precision=0.5000, evidence_hit_rate=1.0000, mrr=1.0000 | Low-productivity firms can operate informally because, given the various costs associated with employing workers in the formal sector (such as compliance with regulations and minimum wage requirements), these firms are m |
| holdout | method | What happens to informal firms when they are caught by authorities? | context_recall=0.0000, context_precision=0.5000, evidence_hit_rate=1.0000, mrr=1.0000 | When informal firms are caught by authorities, they are subject to fines [S2]. The context does not provide specific details on the exact nature or amount of these fines. |
| holdout | result_finding | What does the paper say about informal job mobility? | context_recall=0.0000, context_precision=0.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The paper indicates that there is significant mobility between formal and informal sectors. For instance, it cites Maloney (1999) for Mexico, Barros et al. (1990), Neri (2002), and Curi and Menezes-Filho (2006) for Brazi |
| holdout | result_finding | What happens to unemployment in Sao Paulo low-education males when informality costs rise by 10 percent? | context_recall=0.0000, context_precision=0.2500, evidence_hit_rate=1.0000, mrr=1.0000 | The context does not explicitly state what happens to unemployment in Sao Paulo for low-education males when the cost of informality rises by 10 percent. The provided tables focus on productivity and wages but do not inc |

## Lowest Context Precision

Rows where retrieved contexts included weak or irrelevant evidence.

| Split | Type | Question | Metrics | Answer Snippet |
|---|---|---|---|---|
| dev | method | What key firm decision is added to the wage-posting framework? | context_precision=0.0000, context_recall=1.0000, evidence_hit_rate=1.0000, mrr=1.0000 | In addition to wage posting, firms decide whether to operate in the formal or informal sector [S1]. This dual-sector decision-making process is a key element of the model. |
| dev | method | What kind of worker search does the model allow? | context_precision=0.0000, context_recall=1.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The model allows for endogenous choices by both workers and firms regarding where to work and locate jobs, respectively [S1]. This implies that workers can choose between formal and informal sectors based on their prefer |
| dev | result_finding | What does the paper find about unemployment-to-job offers? | context_precision=0.0000, context_recall=0.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The paper finds that unemployed workers receive job offers according to a Poisson process with an arrival rate denoted as λij, where i = 0 (unemployed) and j = 1 or 2 denote offers from the formal or informal sectors res |
| dev | result_finding | What does Table 3 say about informality by gender? | context_precision=0.0000, context_recall=0.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The answer was not found in the provided context. The context mentions Tables 9, 10-13, and D.9 but does not include a Table 3. Therefore, I cannot provide information from Table 3 as requested. [S1] discusses informalit |
| dev | result_finding | What does Table 9 imply about informal firms? | context_precision=0.0000, context_recall=0.0000, evidence_hit_rate=1.0000, mrr=1.0000 | The answer was not found in the provided context. The context mentions Tables 15 and 16 but does not reference Table 9. Therefore, I cannot provide an implication about informal firms based on Table 9 from the given info |
| holdout | reasoning | Why does the paper emphasize equilibrium effects in policy analysis? | context_precision=0.0000, context_recall=0.5000, evidence_hit_rate=1.0000, mrr=1.0000 | The paper emphasizes equilibrium effects in policy analysis because it aims to understand how reducing or eliminating informality impacts the wage structure and overall welfare, considering firms' responses to policy cha |
| stress | true_out_of_scope | What happened to Brazilian informal labor markets in 2025? | context_precision=0.0000, context_recall=0.5000, evidence_hit_rate=1.0000, mrr=1.0000 | The provided context does not contain information about Brazilian informal labor markets in 2025. The data and analysis are from a period between 2002 and 2007 [S1, S2]. Therefore, the answer was not found in the given c |
| holdout | method | How are formal-sector benefits represented in worker value? | context_precision=0.0000, context_recall=0.5000, evidence_hit_rate=1.0000, mrr=1.0000 | The formal-sector benefits are represented as part of the wage compensation after tax but before social security deductions. These deductions effectively become part of the worker's compensation because they entitle the |
| stress | true_out_of_scope | What is the current stock price of Apple today? | context_precision=0.0000, context_recall=0.0000, evidence_hit_rate=0.0000, mrr=0.0000 | The answer was not found in the provided context. [S1], [S2], [S3], and [S4] contain statistical data related to wages, firm revenue, and government revenue but do not provide information about the current stock price of |
| stress | true_out_of_scope | What is Renata Narita's private phone number? | context_precision=0.0000, context_recall=0.5000, evidence_hit_rate=1.0000, mrr=0.2500 | The private phone number for Renata Narita was not found in the provided context [S1]-[S4]. |

## Lowest Faithfulness

Rows where answer claims were least grounded in retrieved context.

| Split | Type | Question | Metrics | Answer Snippet |
|---|---|---|---|---|
| dev | result_finding | What happens when informality is abolished and contact rates adjust endogenously? | faithfulness=0.0000, hallucination_rate=0.0000, citation_strict_accuracy=1.0000 | When informality is abolished and contact rates adjust endogenously, unemployment declines in all markets, and overall welfare increases as does workers' welfare. In Sao Paulo specifically, about 18% of firms close down, |
| holdout | result_finding | What happens to mean informal wages in Sao Paulo low-education males after informality costs rise? | faithfulness=0.0000, hallucination_rate=0.0000, citation_strict_accuracy=1.0000 | The mean informal wage for low-education males in Sao Paulo decreases after the cost of informality rises by 10 percent. According to Table 16 from [S2], the mean informal wage (log) drops from approximately 6.71 when η |
| holdout | result_finding | What happens to unemployment in Sao Paulo low-education males when informality costs rise by 10 percent? | faithfulness=0.0000, hallucination_rate=0.0000, citation_strict_accuracy=1.0000 | The context does not explicitly state what happens to unemployment in Sao Paulo for low-education males when the cost of informality rises by 10 percent. The provided tables focus on productivity and wages but do not inc |
| stress | claim_verification | Verify this claim: in Sao Paulo low-education males, a 10 percent increase in informality costs reduces total welfare. | faithfulness=0.1667, hallucination_rate=0.0000, citation_strict_accuracy=1.0000 | The claim that a 10 percent increase in informality costs reduces total welfare for low-education males in São Paulo is not directly supported by the provided context. The context indicates that increasing the cost of in |
| holdout | reasoning | Why are high-productivity firms more likely to operate formally? | faithfulness=0.3333, hallucination_rate=0.0000, citation_strict_accuracy=1.0000 | High-productivity firms are more likely to operate formally because as productivity increases, wages also increase. This leads to higher revenues from fines in the informal sector for formal firms, allowing them to captu |
| holdout | method | What are the three worker states used in the model? | faithfulness=0.3333, hallucination_rate=0.0000, citation_strict_accuracy=1.0000 | The three worker states used in the model are unemployment, formal employment, and informal employment. This is described in [S2] where it states: "To derive them we need to define the steady state flow of workers betwee |
| stress | unsupported_claim | Does the paper prove that the same welfare results apply to every developing country? | faithfulness=0.3333, hallucination_rate=0.5000, citation_strict_accuracy=0.0000 | The answer was not found in the provided context. The document discusses the effects of abolishing informality using Brazilian micro data and does not explicitly state whether similar welfare results would apply to every |
| holdout | factual | What JEL codes are listed for the paper? | faithfulness=0.3333, hallucination_rate=0.0000, citation_strict_accuracy=0.0000 | The JEL codes listed for the paper are J24, J3, J42, J6, and O17 [S3]. |
| stress | unsupported_claim | Can you infer the missing productivity threshold for a country not studied in the paper? | faithfulness=0.3333, hallucination_rate=0.0000, citation_strict_accuracy=1.0000 | The answer was not found in the context provided. The tables and notes only contain data for Sao Paulo, Salvador, low education males, and low education females, but do not provide any information about other countries o |
| dev | result_finding | What does Table 9 imply about informal firms? | faithfulness=0.5000, hallucination_rate=0.2500, citation_strict_accuracy=0.7500 | The answer was not found in the provided context. The context mentions Tables 15 and 16 but does not reference Table 9. Therefore, I cannot provide an implication about informal firms based on Table 9 from the given info |

## Citation Alignment Issues

Rows where citations were syntactically valid but did not strongly align with the ground truth.

| Split | Type | Question | Metrics | Answer Snippet |
|---|---|---|---|---|
| holdout | factual | What JEL codes are listed for the paper? | citation_strict_accuracy=0.0000, citation_accuracy=1.0000, context_recall=1.0000 | The JEL codes listed for the paper are J24, J3, J42, J6, and O17 [S3]. |
| holdout | result_finding | What happens to mean formal wages in Sao Paulo low-education males after informality costs rise? | citation_strict_accuracy=0.0000, citation_accuracy=0.0000, context_recall=1.0000 | The mean formal wages for low-education males in Sao Paulo increase from 7.18 (η = 0.3) to 7.29 (η = 0.5) when the cost of informality rises by 10 percent [S1, S3]. |
| stress | true_out_of_scope | What is the current stock price of Apple today? | citation_strict_accuracy=0.0000, citation_accuracy=1.0000, context_recall=0.0000 | The answer was not found in the provided context. [S1], [S2], [S3], and [S4] contain statistical data related to wages, firm revenue, and government revenue but do not provide information about the current stock price of |
| stress | unsupported_claim | Does the paper prove that the same welfare results apply to every developing country? | citation_strict_accuracy=0.0000, citation_accuracy=1.0000, context_recall=0.6667 | The answer was not found in the provided context. The document discusses the effects of abolishing informality using Brazilian micro data and does not explicitly state whether similar welfare results would apply to every |
| stress | format_check | Give a one-sentence answer: what is the main dataset? | citation_strict_accuracy=0.0000, citation_accuracy=1.0000, context_recall=0.0000 | The main dataset consists of employment data across different regions in Brazil, including Recife, Salvador, Belo Horizonte, Rio de Janeiro, São Paulo, Porto Alegre, and their respective subcategories such as unemployed |
| stress | unsupported_claim | Does the paper show that minimum wage laws are the only cause of informality? | citation_strict_accuracy=0.0000, citation_accuracy=0.0000, context_recall=1.0000 | The paper does not show that minimum wage laws are the only cause of informality. The authors extend the wage-posting framework to include costs of informality, such as penalties when caught, and search frictions [S2, S3 |
| stress | true_out_of_scope | Which policy did Brazil actually implement after this 2012 working paper was published? | citation_strict_accuracy=0.5000, citation_accuracy=1.0000, context_recall=1.0000 | The answer was not found in the provided context. The document discusses theoretical models and potential policies but does not mention any specific policy implemented by Brazil after the publication of this 2012 working |
| stress | true_out_of_scope | What is Renata Narita's private phone number? | citation_strict_accuracy=0.5000, citation_accuracy=1.0000, context_recall=0.5000 | The private phone number for Renata Narita was not found in the provided context [S1]-[S4]. |
| stress | true_out_of_scope | What are the authors' current institutional affiliations today? | citation_strict_accuracy=0.5000, citation_accuracy=1.0000, context_recall=1.0000 | The provided context does not contain information about the authors' current institutional affiliations. The contact information given is from 2012, and it may have changed since then. To find their current affiliations |
| stress | true_out_of_scope | What is the exact replication code used by the authors? | citation_strict_accuracy=0.5000, citation_accuracy=1.0000, context_recall=0.5000 | The provided context does not contain specific information about the exact replication code used by the authors. The documents describe the methodology and mathematical derivations but do not provide details on the softw |

## Hallucination Candidates

Rows where the custom unsupported-claim heuristic found unsupported answer claims.

| Split | Type | Question | Metrics | Answer Snippet |
|---|---|---|---|---|
| stress | unsupported_claim | Does the paper prove that the same welfare results apply to every developing country? | hallucination_rate=0.5000, faithfulness=0.3333, citation_strict_accuracy=0.0000 | The answer was not found in the provided context. The document discusses the effects of abolishing informality using Brazilian micro data and does not explicitly state whether similar welfare results would apply to every |
| stress | true_out_of_scope | What is the current stock price of Apple today? | hallucination_rate=0.5000, faithfulness=1.0000, citation_strict_accuracy=0.0000 | The answer was not found in the provided context. [S1], [S2], [S3], and [S4] contain statistical data related to wages, firm revenue, and government revenue but do not provide information about the current stock price of |
| stress | true_out_of_scope | What are the authors' current institutional affiliations today? | hallucination_rate=0.3333, faithfulness=1.0000, citation_strict_accuracy=0.5000 | The provided context does not contain information about the authors' current institutional affiliations. The contact information given is from 2012, and it may have changed since then. To find their current affiliations |
| stress | language_check | Summarize the conclusion in English only, even though this instruction mentions tieng Viet. | hallucination_rate=0.3000, faithfulness=1.0000, citation_strict_accuracy=0.6667 | The document discusses different labor market regimes based on firm productivity levels. Key conclusions include: 1. **Inactivity**: Firms are inactive when productivity is low. 2. **Informal Sector Only**: Firms operat |
| dev | result_finding | What does Table 9 imply about informal firms? | hallucination_rate=0.2500, faithfulness=0.5000, citation_strict_accuracy=0.7500 | The answer was not found in the provided context. The context mentions Tables 15 and 16 but does not reference Table 9. Therefore, I cannot provide an implication about informal firms based on Table 9 from the given info |
| dev | result_finding | What does Table 4 show about wages by gender and sector? | hallucination_rate=0.0909, faithfulness=0.8000, citation_strict_accuracy=1.0000 | Table 4 from [S2] shows that in Sao Paulo, male formal workers have a higher mean log wage of 6.67 compared to female formal workers at 6.38. Similarly, the standard deviation for male formal wages is 0.42 while for fema |


## Recommended Next Step: Table/Numeric Retrieval

The most portfolio-worthy next improvement is table-aware retrieval:

- Extract PDF tables separately during ingestion.
- Store table chunks with metadata such as `page`, `table_number`, `section`, and `caption`.
- Detect table/numeric queries such as `Table 3`, `10 percent`, `welfare`, or `unemployment`.
- Retrieve normal text and table chunks separately, then merge or rerank the candidates.
- Add gold evidence labels such as `expected_page`, `expected_section`, or `expected_table` for stronger retrieval metrics.
