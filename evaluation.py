from retriever_arabic import dense_retrieve


# ============================================================
# Evaluation Questions
# ============================================================

TEST_QUESTIONS = [

    # Chest / Smoking
    "قعدت فى مكان فى دخان و صدرى واجعنى",
    "انا بدخن وعندى ألم فى الصدر",
    "عندى ضيق فى التنفس مع ألم فى الصدر",

    # Stomach
    "عندى تعب فى المعده بقالى فتره",
    "عندى ألم فى المعدة وبشرب قهوة كتير",
    "عندى حرقان فى المعدة",
    "معدتى بتوجعنى بعد الأكل",

    # Headache
    "عندى صداع مستمر من كام يوم",
    "عندى صداع ودوخة",
    "عندى صداع يومى",

    # Blood pressure
    "عندى ارتفاع فى ضغط الدم",
    "ضغط الدم عندى مرتفع",
    "ضغطى 160 على 100",

    # General
    "عندى دوخة مستمرة",
    "عندى غثيان وترجيع",
    "عندى ألم فى البطن",
    "عندى كحة مستمرة",
    "عندى إرهاق وتعب مستمر",
    "عندى فقدان شهية",

]


def ask_relevance(question, result):

    print("\n" + "-" * 80)

    print("QUESTION:")
    print(question)

    print("\nRESULT:")
    print(result["text"][:700])

    print("\nSimilarity:")
    print(f"{result['score']:.4f}")

    while True:

        answer = input(
            "\nRelevant? "
            "[y = yes / n = no / s = skip]: "
        ).strip().lower()

        if answer in ["y", "n", "s"]:
            return answer


def evaluate_question(question):

    results = dense_retrieve(
        question,
        top_k=5
    )

    print("\n")
    print("=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, 1):

        print("\n" + "-" * 80)

        print(f"Rank: {i}")
        print(f"Chunk ID: {result['id']}")
        print(f"Score: {result['score']:.4f}")
        print(
            f"Category: "
            f"{result['meta'].get('category', '')}"
        )

        print("\nText:")
        print(result["text"][:500])


    relevant_ranks = []

    print("\n")
    print("=" * 80)
    print("MANUAL RELEVANCE EVALUATION")
    print("=" * 80)

    for rank, result in enumerate(results, 1):

        label = ask_relevance(
            question,
            result
        )

        if label == "y":

            relevant_ranks.append(rank)

        elif label == "s":

            break

    return results, relevant_ranks


def calculate_metrics(evaluations):

    total = len(evaluations)

    if total == 0:
        return

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0

    reciprocal_ranks = []

    for evaluation in evaluations:

        relevant_ranks = evaluation["relevant_ranks"]

        if not relevant_ranks:
            continue

        first_relevant = min(
            relevant_ranks
        )

        if first_relevant <= 1:
            hit_at_1 += 1

        if first_relevant <= 3:
            hit_at_3 += 1

        if first_relevant <= 5:
            hit_at_5 += 1

        reciprocal_ranks.append(
            1 / first_relevant
        )

    print("\n")
    print("=" * 80)
    print("DENSE RETRIEVAL EVALUATION")
    print("=" * 80)

    print(f"\nTotal Questions: {total}")

    print(
        f"\nHit@1: "
        f"{hit_at_1 / total:.4f}"
        f" ({hit_at_1}/{total})"
    )

    print(
        f"Hit@3: "
        f"{hit_at_3 / total:.4f}"
        f" ({hit_at_3}/{total})"
    )

    print(
        f"Hit@5: "
        f"{hit_at_5 / total:.4f}"
        f" ({hit_at_5}/{total})"
    )

    if reciprocal_ranks:

        mrr = (
            sum(reciprocal_ranks)
            / total
        )

        print(
            f"MRR: "
            f"{mrr:.4f}"
        )

    print("\n")


def main():

    print("=" * 80)
    print("ARABIC DENSE RETRIEVAL EVALUATION")
    print("=" * 80)

    print(
        "\nWe will manually evaluate "
        "the Top-5 results."
    )

    print(
        "\nFor every result:"
        "\n  y = relevant"
        "\n  n = not relevant"
        "\n  s = stop evaluating this question"
    )

    evaluations = []

    for index, question in enumerate(
        TEST_QUESTIONS,
        1
    ):

        print("\n\n")
        print("#" * 80)

        print(
            f"QUESTION "
            f"{index}/{len(TEST_QUESTIONS)}"
        )

        print("#" * 80)

        try:

            results, relevant_ranks = (
                evaluate_question(question)
            )

            evaluations.append({

                "question": question,

                "relevant_ranks":
                    relevant_ranks

            })

        except Exception as e:

            print("\nERROR:")
            print(e)


    calculate_metrics(
        evaluations
    )

    print("=" * 80)
    print("EVALUATION FINISHED")
    print("=" * 80)


if __name__ == "__main__":

    main()