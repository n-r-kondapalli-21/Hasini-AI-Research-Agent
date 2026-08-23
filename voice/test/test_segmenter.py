from ..segmenter import ResponseSegmenter


def test_sentence_boundary():
    print("\nTEST 1: Sentence boundary")

    s = ResponseSegmenter()

    tokens = [
        "Hello ",
        "there. ",
        "This ",
        "is ",
        "another ",
        "sentence!"
    ]

    chunks = []

    for token in tokens:
        chunk = s.add(token)

        if chunk:
            chunks.append(chunk)
            print("CHUNK:", repr(chunk))

    chunks.append(s.flush()) if s.flush() else None

    assert chunks[0] == "Hello there."
    assert "This is another sentence!" in chunks

    print("PASS")


def test_question_and_exclamation():
    print("\nTEST 2: ! and ? boundaries")

    s = ResponseSegmenter()

    chunks = []

    for token in [
        "Are ",
        "you ",
        "ready? ",
        "Let's ",
        "go!"
    ]:
        chunk = s.add(token)

        if chunk:
            chunks.append(chunk)
            print("CHUNK:", repr(chunk))

    assert chunks == [
        "Are you ready?",
        "Let's go!"
    ]

    print("PASS")


def test_comma_boundary():
    print("\nTEST 3: Comma boundary")

    s = ResponseSegmenter(
        min_chunk_length=10
    )

    chunks = []

    for token in [
        "This is a long clause, ",
        "and this continues."
    ]:
        chunk = s.add(token)

        if chunk:
            chunks.append(chunk)
            print("CHUNK:", repr(chunk))

    assert chunks[0] == "This is a long clause,"

    print("PASS")


def test_short_comma_does_not_split():
    print("\nTEST 4: Short comma should NOT split")

    s = ResponseSegmenter(
        min_chunk_length=25
    )

    chunks = []

    for token in [
        "Hello, ",
        "this continues ",
        "with more text."
    ]:
        chunk = s.add(token)

        if chunk:
            chunks.append(chunk)
            print("CHUNK:", repr(chunk))

    assert chunks[0] == "Hello, this continues with more text."

    print("PASS")


def test_final_flush():
    print("\nTEST 5: Final flush")

    s = ResponseSegmenter()

    s.add("This response has no punctuation")

    chunk = s.flush()

    print("FINAL:", repr(chunk))

    assert chunk == "This response has no punctuation"

    print("PASS")


def test_empty_input():
    print("\nTEST 6: Empty input")

    s = ResponseSegmenter()

    assert s.add("") is None
    assert s.flush() is None

    print("PASS")


def test_long_response():
    print("\nTEST 7: Long response")

    s = ResponseSegmenter(
        min_chunk_length=20,
        max_buffer_time=3.0
    )

    chunks = []

    tokens = [
        "RAG ",
        "is ",
        "a ",
        "retrieval ",
        "augmented ",
        "generation ",
        "system. ",
        "It ",
        "retrieves ",
        "information ",
        "from ",
        "a ",
        "knowledge ",
        "base. ",
        "The ",
        "LLM ",
        "then ",
        "uses ",
        "that ",
        "information ",
        "to ",
        "generate ",
        "an ",
        "answer."
    ]

    for token in tokens:

        chunk = s.add(token)

        if chunk:
            chunks.append(chunk)
            print("CHUNK:", repr(chunk))

    final = s.flush()

    if final:
        chunks.append(final)
        print("FINAL:", repr(final))

    assert len(chunks) >= 2

    print("PASS")


def main():

    print("=" * 60)
    print("PHASE 2.3 - RESPONSE SEGMENTER TESTS")
    print("=" * 60)

    test_sentence_boundary()
    test_question_and_exclamation()
    test_comma_boundary()
    test_short_comma_does_not_split()
    test_final_flush()
    test_empty_input()
    test_long_response()

    print("\n" + "=" * 60)
    print("ALL PHASE 2.3 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()