from retriever_arabic import dense_retrieve

TEST_QUESTIONS = [
    "قعدت فى مكان فى دخان و صدرى واجعنى",

    "عندى تعب فى المعده بقالى فتره",

    "عندى صداع مستمر من كام يوم",

    "عندى ارتفاع فى ضغط الدم",

    "عندى ألم فى المعدة وبشرب قهوة كتير",
]

def print_results(question, results):

    print("\n")
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\n")
    print("=" * 80)
    print("FINAL DENSE RETRIEVAL RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, 1):

        print("\n" + "-" * 80)

        print(f"Rank: {i}")

        print(f"Chunk ID: {result['id']}")

        print(
            f"Dense Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Category: "
            f"{result['meta'].get('category', '')}"
        )

        print(
            f"Authority: "
            f"{result['meta'].get('authority_family', '')}"
        )

        print("\nText:")

        print(
            result["text"][:1000]
        )

def main():

    print("=" * 80)
    print("ARABIC DENSE RETRIEVAL TEST")
    print("=" * 80)

    print(
        "\nUsing the same dense_retrieve() "
        "from retriever_arabic.py"
    )

    for question in TEST_QUESTIONS:

        try:

            results = dense_retrieve(
                question,
                top_k=5
            )

            print_results(
                question,
                results
            )

        except Exception as e:

            print("\nERROR")
            print("-" * 80)
            print(e)

    print("\n")
    print("=" * 80)
    print("DENSE RETRIEVAL TEST FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()