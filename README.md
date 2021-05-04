# Semantic Kernel: Concept Graphs for EDA #


sudo apt-get install graphviz graphviz-dev
pip install pygraphviz

```bash
python -m spacy download da_core_news_sm


```

```python
import nltk
nltk.download('punkt')
```


pip install python-Levenshtein



####################

EXPERIMENT 1: vectors_expr1
dat-error, no-lemma 128, 4 threads, window 5
['kirke', 'menighed', 'præst']
['aabenbaring', 'BEKIENDELSE', 'bekjendelse', 'bibelen', 'DIGTER', 'forfatter', 'GIERNING', 'gjerning', 'grundsætning', 'KIRKE', 'KIRKEN', 'MENIGHED', 'menigheden', 'person', 'PROPHET', 'PRÆST', 'skribent', 'troes']
[INFO] saved 19465 vectors to mdl/vectors_test.pcl
[INFO] runtime 4914.694873809814 seconds.

nucle_types
{'kirke': ['kirke', 'kirken', 'menighed', 'kirkes', 'skole'], 'menighed': ['menighed', 'gierning', 'bekiendelse', 'grundvold', 'tjeneste'], 'præst': ['præst', 'prophet', 'digter', 'person', 'mand']}

nucle_tokens
{'bekiendelse': ['bekiendelse', 'bekjendelse', 'troes', 'regel', 'menigheds', 'eenhed', 'uforanderlige', 'børnelærdom', 'artikel', 'skrifts'], 'digter': ['digter', 'forfatter', 'skribent', 'historiker', 'begivenhed', 'tyran', 'hjemmel', 'fortælling', 'rolle', 'idee'], 'gierning': ['gierning', 'grundsætning', 'gjerning', 'tilværelse', 'bevidsthed', 'personlighed', 'skabning', 'menighed', 'bekiendelse', 'aabenbaring'], 'kirke': ['kirke', 'kirken', 'menighed', 'kirkes', 'skole', 'stat', 'stats', 'bekiendelse', 'grundvold', 'forfatning'], 'kirken': ['kirken', 'menigheden', 'bibelen', 'staten', 'kirke', 'troen', 'skriften', 'skolen', 'christendommen', 'historien'], 'menighed': ['menighed', 'gierning', 'bekiendelse', 'grundvold', 'tjeneste', 'tilværelse', 'menigheds', 'grundsætning', 'guddom', 'personlighed'], 'prophet': ['prophet', 'person', 'aabenbaring', 'præst', 'digter', 'talsmand', 'tyran', 'pave', 'forfatter', 'fjende'], 'præst': ['præst', 'prophet', 'digter', 'person', 'mand', 'tyran', 'kristen', 'christen', 'biskop', 'catechismus']}


EXPERIMENT 2: vectors_expr2
dat, no-lemma, 128, 4 threads, window 5
['kirke', 'menighed', 'præst']
['BISP', 'evangeliske', 'farfader', 'folkekirke', 'forfatningen', 'forkyndelse', 'FRIMENIGHED', 'johan', 'KIRKE', 'KIRKEN', 'KIRKES', 'MENIGHED', 'MENIGHEDS', 'PRÆST', 'PRÆSTEN', 'præsterne', 'statskirke']
[INFO] saved 34091 vectors to mdl/vectors_test.pcl
[INFO] runtime 10227.938758134842 seconds.

nucle_types
{'kirke': ['kirke', 'kirken', 'kirkes'], 'menighed': ['menighed', 'frimenighed', 'menigheds'], 'præst': ['præst', 'bisp', 'præsten']}

nucle_tokens
{'bisp': ['bisp', 'johan', 'farfader'], 'frimenighed': ['frimenighed', 'folkekirke', 'statskirke'], 'kirke': ['kirke', 'kirken', 'kirkes'], 'kirken': ['kirken', 'kirke', 'forfatningen'], 'kirkes': ['kirkes', 'evangeliske', 'menigheds'], 'menighed': ['menighed', 'frimenighed', 'menigheds'], 'menigheds': ['menigheds', 'kirkes', 'forkyndelse'], 'præst': ['præst', 'bisp', 'præsten'], 'præsten': ['præsten', 'præsterne', 'præst']}


EXPERIMENT 3: vectors_expr3
dat, lemma, 128, 4 threads, window 5

nucle_types
{'kirke': ['kirke', 'menighed', 'folkekirke'], 'menighed': ['menighed', 'frimenighed', 'evangelium'], 'præst': ['præst', 'lærer', 'bisp']}

nucle_tokens
{'bisp': ['bisp', 'ærke', 'præst'], 'evangelium': ['evangelium', 'apostel', 'forkyndelse'], 'folkekirke': ['folkekirke', 'statskirke', 'religion'], 'frimenighed': ['frimenighed', 'folkekirke', 'statskirke'], 'kirke': ['kirke', 'menighed', 'folkekirke'], 'lærer': ['lærer', 'embedsmand', 'præst'], 'menighed': ['menighed', 'frimenighed', 'evangelium'], 'præst': ['præst', 'lærer', 'bisp']}

[INFO] saved 24450 vectors to mdl/vectors_expr3.pcl
[INFO] runtime 14314.589876413345 seconds.

EXPERIMENT 4: vectors_expr4
dat, lemma, 100, 4 threads, window 5

nucle_types
{'kirke': ['kirke', 'menighed', 'folkekirke'], 'menighed': ['menighed', 'frimenighed', 'evangelium'], 'præst': ['præst', 'lærer', 'bisp']}

nucle_tokens
{'bisp': ['bisp', 'ærke', 'præst'], 'evangelium': ['evangelium', 'apostel', 'forkyndelse'], 'folkekirke': ['folkekirke', 'statskirke', 'religion'], 'frimenighed': ['frimenighed', 'folkekirke', 'statskirke'], 'kirke': ['kirke', 'menighed', 'folkekirke'], 'lærer': ['lærer', 'embedsmand', 'præst'], 'menighed': ['menighed', 'frimenighed', 'evangelium'], 'præst': ['præst', 'lærer', 'bisp']}

EXPERIMENT 5: vectors_expr5
dat, lemma, 100, 4 threads, window 10, min_count=10

[INFO] writing vectors to disc ...
[INFO] saved 24450 vectors to mdl/vectors_expr5.pcl

nucle_types

nucle_tokens


EXPERIMENT 6: vectors_expr6
dat, lemma, 100, 4 threads, window 3, min_count=1

[INFO] writing vectors to disc ...
[INFO] saved 147859 vectors to mdl/vectors_expr6.pcl

nucle_types

nucle_tokens


EXPERIMENT 7: vectors_expr7 --> KEEP max cores
dat, lemma, 100, 8 threads, window 5, min_count=10

[INFO] saved 24450 vectors to mdl/vectors_expr7.pcl

[INFO] runtime 9580.704177856445 seconds.

nucle_types

nucle_tokens

EXPERIMENT 8: vectors_expr8 (run in batch mode)

dat, lemma, 50, 8 threads, window 5, min_count=10