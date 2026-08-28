from permissions import ToolCall, check


def test_filesystem_read():
    """Low-risk, read-only filesystem operation."""

    result = check(
        ToolCall(
            server_id="filesystem",
            method_name="read_text_file",
            arguments={
                "path": "test.txt"
            },
        )
    )

    print("READ:", result)


def test_filesystem_move():
    """Filesystem operation that modifies state — should be gated higher than read."""

    result = check(
        ToolCall(
            server_id="filesystem",
            method_name="move_file",
            arguments={
                "source": "a.txt",
                "destination": "b.txt",
            },
        )
    )

    print("MOVE:", result)


def test_github_push():
    """Remote, state-changing operation — should require the highest tier."""

    result = check(
        ToolCall(
            server_id="github",
            method_name="push_files",
            arguments={},
        ),
    )

    print("GITHUB:", result)


def test_unregistered_server_defaults_high():
    """
    A server that isn't in the permission registry at all.

    Per the intended design, this should default to HIGH
    (fail safe) and log a warning, rather than silently
    treating it as low-risk.
    """

    result = check(
        ToolCall(
            server_id="some_unregistered_server",
            method_name="do_anything",
            arguments={},
        ),
    )

    print("UNREGISTERED:", result)


def test_unknown_method_on_known_server():
    """
    A known server (filesystem) called with a method not in
    its registry entry — checks the fallback behavior for
    partially-registered servers, not just fully-unknown ones.
    """

    result = check(
        ToolCall(
            server_id="filesystem",
            method_name="totally_made_up_method",
            arguments={},
        ),
    )

    print("UNKNOWN METHOD:", result)


if __name__ == "__main__":

    test_filesystem_read()
    test_filesystem_move()
    test_github_push()
    test_unregistered_server_defaults_high()
    test_unknown_method_on_known_server()