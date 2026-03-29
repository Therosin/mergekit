from mergekit.architecture.json_definitions import NAME_TO_ARCH


def test_qwen3_5_architecture_exposes_nested_vocab_size_key():
    arch_info = NAME_TO_ARCH["Qwen3_5ForConditionalGeneration"][0]
    assert arch_info.expected_model_type == "qwen3_5"
    assert arch_info.vocab_size_config_key == "text_config.vocab_size"


def test_qwen3_5_text_architecture_uses_text_only_tensor_prefixes():
    arch_info = NAME_TO_ARCH["Qwen3_5ForCausalLM"][0]
    pre_weights = arch_info.modules["default"].architecture.definition.pre_weights
    assert pre_weights[0].name == "model.embed_tokens.weight"
