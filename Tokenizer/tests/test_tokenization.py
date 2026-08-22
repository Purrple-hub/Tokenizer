from anatok.tokenizer import (
    TextChunkStreamer,
    byte_based_tokenize,
    detokenize,
    get_hf_tokenizer,
    iter_word_group_tokens,
    merge_tokens,
    text_tokenize,
    tokenize_file,
    tokenize_with_hf,
)

def test_text_and_byte_tokenize():
    tokens = text_tokenize("hello world this is a test")
    assert isinstance(tokens, list) and tokens
    assert isinstance(byte_based_tokenize(b"test data"), list)

def test_merge_detokenize():
    assert merge_tokens([["a", "b"], ["c"]]) == ["a", "b", "c"]
    assert detokenize(["hello", "world"]) == "hello world"

def test_streamer_splits_at_boundaries():
    s = TextChunkStreamer(target_size=4096)
    pieces = s.feed(b"word " * 2000)
    assert all(len(p) <= 4096 for p in pieces)
    tail = s.finish()
    joined = "".join(pieces + tail)
    assert joined == "word " * 2000

def test_hf_wrapper_encode_decode_vocab():
    wrapper = get_hf_tokenizer()
    ids = wrapper.encode("Hello world")
    assert isinstance(ids, list) and ids
    vocab_size = wrapper.get_vocab_size()
    assert isinstance(vocab_size, int) and vocab_size > 0
    token_str = wrapper.id_to_token(ids[0])
    assert isinstance(token_str, str)

def test_iter_word_group_tokens(sample_file):
    groups = list(iter_word_group_tokens(sample_file, group_size=10))
    assert len(groups) >= 1
    for group in groups[:-1]:
        assert len(group.split()) == 10

def test_tokenize_file(sample_file):
    result = tokenize_file(sample_file)
    assert result["token_count"] > 0
    assert result["tokens"]
    assert result["filepath"] == sample_file

def test_tokenize_with_hf_offsets():
    t = tokenize_with_hf("offsets please", return_offsets=True)
    assert len(t["offset_mapping"]) == len(t["input_ids"])
