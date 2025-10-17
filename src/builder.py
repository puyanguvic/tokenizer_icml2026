from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
import os, json
from tqdm import tqdm
from .http_tokenizer import http_clean_line

def train_domain_tokenizer(corpus_files, save_dir="tokenizer_output", vocab_size=None):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=1, vocab_size=vocab_size
    )

    def lines():
        for path in corpus_files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield http_clean_line(line)

    tokenizer.train_from_iterator(tqdm(lines()), trainer=trainer)
    tokenizer.decoder = decoders.WordLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    vocab = tokenizer.get_vocab()
    with open(os.path.join(save_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for token, _ in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(token + "\n")

    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "wordlevel", "unk_token": "[UNK]"}, f, indent=2)

    print(f"✅ Tokenizer exported to {save_dir}")
    return tokenizer

