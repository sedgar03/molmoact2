import re


non_countable_quantities = [
    # time
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "[a-z]*seconds",

    # length/area
    "(tera|giga|mega|deci|kilo||micro|centi|milli|nano|pico|deca)meters",
    "meters",
    "metres",  # mispelt meters
    "acres",
    "leagues",
    "fathoms",
    "nautical miles",
    "hectares",
    "(square |SQ )?inches",
    "(square |SQ )feet",  # Just feet can be a false positive
    "(square |SQ )?ft",
    "(square |SQ )?miles",
    "(square | SQ)?yards",
    "passing yards",

    # currency
    "dollars",
    "cents",
    "pounds",
    "euros",

    # speed
    "seed",
    "mph",
    "kph",

    # Comparisons "how many more..."
    "more",
    "fewer",
    "less",

    "likes",  # almost always from a screenshot

    # volume
    "cubic",
    "gallons",
    "quarts",
    "pints",
    "fluid ounces",
    "[a-z]*liters",
    # ambiguous and probably more often used as an object then a volume
    # cup
    # tablespoons
    # teaspoons

    # weight
    "weight",
    "[a-z]*grams",
    "pounds",
    "tons",
    "ounces",

    "ways", "different ways",

    # other
    "degrees", "calories",
    "hertz", "horsepower", "[a-z]*bytes",
    "psi", "atmospheres", "[a-z]*watts",
]
non_countable_re_str = "|".join(non_countable_quantities)
# We avoid "the" for things like "the number of the player who scored a goal
non_countable_end_re_str = "|".join(non_countable_quantities + ["money", "the"])

counting_patterns = [
    f'how ?many (?!{non_countable_re_str})',
    r'(count|tally) (all|every|each|the) ',
    f"(what|(what's|what (is|was|were)|states?|indicates?|tell me|say) the( exact| precise)?) ((total count|count|total|number|total number|total amount|amount) of (?!{non_countable_end_re_str}))",
]
count_any = re.compile("^(?!approximately).*(\\b|^|\n)(?P<all>" + "|".join(counting_patterns) + ")\\b.*", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def is_pixmo_point_and_count_question(question: str) -> bool:
    """
    Returns whether the question could be a counting question that would use pointing for

    This check is conservative by design, so it will have a high recall but low precision

    The main goal is to flag questions that closely resemble the format we use for counting, like
    "how many cats?", while not flagging question the closely resemble that format but are
    not counting, like "how many days old is it?"

    We accept other false positives instead of using a more complex detector, since, for questions
    that do not resemble pixmo-point format, is pretty harmless to add "a no pointing" instruction
    The trained will still learn to avoid pointing is those cases anyway
    """
    return bool(count_any.fullmatch(question))


def test_is_counting_question():
    assert is_pixmo_point_and_count_question("how many times does he smile?")
    assert is_pixmo_point_and_count_question("Count all the cats")
    assert is_pixmo_point_and_count_question("Count the cats")
    assert is_pixmo_point_and_count_question("What is the exact number of performers in the video?")
    assert is_pixmo_point_and_count_question("Tell me, how many cats?")
    assert is_pixmo_point_and_count_question("Count the cats?")
    assert is_pixmo_point_and_count_question("What's the number of dogs?")
    assert not is_pixmo_point_and_count_question("What is the number of degrees in the cricle?")
    assert is_pixmo_point_and_count_question("What number of zebras are standing in front of the tree surrounded by a chain link fence?")
    assert is_pixmo_point_and_count_question("What is the number of nice elephants who are living inside the zoo enclosure?")
    assert is_pixmo_point_and_count_question("What amount of children are sitting in front of the TV, when Mrs. Allen opens the door?")
    assert is_pixmo_point_and_count_question("How many cup are shown in this video?")
    assert not is_pixmo_point_and_count_question("What amount of money was spent?")
    assert not is_pixmo_point_and_count_question("What is the maximum number of shoes present?")
    assert not is_pixmo_point_and_count_question("What is the number written on top of the middle green bananas?")
    assert not is_pixmo_point_and_count_question("What number is on the yellow train?")
    assert not is_pixmo_point_and_count_question("What country is likely hosting this vehicle evident by the writing on its side?")
    assert not is_pixmo_point_and_count_question("Approximately how many people live in this city?")
    assert not is_pixmo_point_and_count_question("How many watts does a night lamp use?")
    assert not is_pixmo_point_and_count_question("How many miles are there?")
    assert is_pixmo_point_and_count_question("How many thermometers are there?")
    assert not is_pixmo_point_and_count_question("What is one change to the ecosystem that would increase the number of frogs?")


if __name__ == '__main__':
    test_is_counting_question()