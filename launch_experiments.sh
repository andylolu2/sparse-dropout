python eval/llm/train.py -c eval/llm/config.py --config.model.dropout.p=0.0 --config.model.dropout.variant="blockwise[cuda]"

python eval/llm/train.py -c eval/llm/config.py --config.model.dropout.p=0.2 --config.model.dropout.variant="blockwise[cuda]"

python eval/llm/train.py -c eval/llm/config.py --config.model.dropout.p=0.4 --config.model.dropout.variant="blockwise[cuda]"

python eval/llm/train.py -c eval/llm/config.py --config.model.dropout.p=0.5 --config.model.dropout.variant="blockwise[cuda]"

python eval/llm/train.py -c eval/llm/config.py --config.model.dropout.p=0.6 --config.model.dropout.variant="blockwise[cuda]"

python eval/llm/train.py -c eval/llm/config.py --config.model.dropout.p=0.4 --config.model.dropout.variant="vanilla"

python eval/llm/train.py -c eval/llm/config.py --config.model.dropout.p=0.6 --config.model.dropout.variant="vanilla"

python eval/llm/train.py -c eval/llm/config.py --config.model.dropout.p=0.8 --config.model.dropout.variant="vanilla"