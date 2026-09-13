from voice.permissions import (
    RiskTier,
    ToolCallInfo,
    check,
    is_medium_confirmation,
    is_high_confirmation,
)


def test_normal_read_tool_is_low():
    tool = ToolCallInfo(
        tool_name="read_file",
        server_name="filesystem",
    )

    assert check(tool) == RiskTier.LOW


def test_delete_file_is_medium():
    tool = ToolCallInfo(
        tool_name="delete_file",
        server_name="filesystem",
    )

    assert check(tool) == RiskTier.MEDIUM


def test_trading_server_is_always_high():
    tool = ToolCallInfo(
        tool_name="get_account_balance",
        server_name="trading",
    )

    assert check(tool) == RiskTier.HIGH


def test_trading_server_is_high_even_for_normal_tool_name():
    tool = ToolCallInfo(
        tool_name="get_price",
        server_name="broker_mcp",
    )

    assert check(tool) == RiskTier.HIGH


def test_place_order_is_high():
    tool = ToolCallInfo(
        tool_name="place_order",
        server_name="some_server",
    )

    assert check(tool) == RiskTier.HIGH


def test_medium_yes_confirmation():
    assert is_medium_confirmation("yes")
    assert is_medium_confirmation("confirm")
    assert is_medium_confirmation("do it")


def test_high_confirmation_rejects_bare_yes():
    assert not is_high_confirmation(
        text="yes",
        expected_phrase="yes place the order",
        stt_confidence=0.95,
        confidence_threshold=0.75,
    )


def test_high_confirmation_requires_confidence():
    assert not is_high_confirmation(
        text="yes place the order",
        expected_phrase="yes place the order",
        stt_confidence=0.60,
        confidence_threshold=0.75,
    )


def test_high_confirmation_accepts_valid_phrase():
    assert is_high_confirmation(
        text="yes, place the order",
        expected_phrase="yes, place the order",
        stt_confidence=0.90,
        confidence_threshold=0.75,
    )