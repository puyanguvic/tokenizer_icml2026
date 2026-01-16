from ctok.tokenization.tokenizer import CtokTokenizer
from ctok.tokenization.vocab import Vocabulary


def test_encode_decode_roundtrip():
    vocab = Vocabulary(tokens=["a", "b", "ab"], special_tokens={})
    tokenizer = CtokTokenizer(vocab)
    ids = tokenizer.encode("ab")
    assert tokenizer.decode(ids) == "ab"


def test_call_padding_and_attention():
    vocab = Vocabulary(tokens=["a", "b", "<pad>"], special_tokens={"pad": "<pad>"})
    tokenizer = CtokTokenizer(vocab)
    batch = tokenizer(["a", "ab"], padding=True)
    assert len(batch["input_ids"]) == 2
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert batch["attention_mask"][0][-1] == 0


def test_pair_special_tokens_roberta_style():
    vocab = Vocabulary(tokens=["a", "b", "<s>", "</s>"], special_tokens={"cls": "<s>", "sep": "</s>"})
    tokenizer = CtokTokenizer(vocab)
    output = tokenizer("a", text_pair="b", add_special_tokens=True, padding=False)
    ids = output["input_ids"]
    cls_id = vocab.id_for("<s>")
    sep_id = vocab.id_for("</s>")
    assert ids == [cls_id, vocab.id_for("a"), sep_id, sep_id, vocab.id_for("b"), sep_id]


def test_save_and_load(tmp_path):
    vocab = Vocabulary(tokens=["a", "b", "<pad>"], special_tokens={"pad": "<pad>"})
    tokenizer = CtokTokenizer(vocab)
    tokenizer.save_pretrained(tmp_path)
    loaded = CtokTokenizer.from_pretrained(tmp_path)
    assert loaded.encode("ab") == tokenizer.encode("ab")
    assert loaded.pad_token == "<pad>"
