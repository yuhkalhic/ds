# 这是计算机2204 20221251 余嘉阳的数据科学导论作业

> baseline.py 是基础版本
>
> 
> cot.py 是思维链推理增强
>
> 
> rag.py 是RAG推理增强
>
> 
> inter.py 是INTERVENOR推理增强
>
> 
> all.py 是三种方法的综合框架


这是几个脚本的推理bleu分数
|method|bleu|
|---|---|
|baseline| 0.3398|
|RAG|0.3704|
|CoT|0.4420|
|INTERVENOR|0.2603|
|all|0.3650|

这是INTERVENOR技术应用前后代码成功运行的概率变化
|method|before|after|
|--|--|--|
|INTERVENOR|0.57|0.81|
|all|0.18|0.5|
