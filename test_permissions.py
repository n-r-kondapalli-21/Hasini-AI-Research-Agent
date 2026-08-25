from permissions import ToolCall, check


def test_filesystem_read():

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

    result = check(
        ToolCall(
            server_id="github",
            method_name="push_files",
            arguments={},
        ),
    )

    print("GITHUB:", result)


if __name__ == "__main__":

    test_filesystem_read()
    test_filesystem_move()
    test_github_push()