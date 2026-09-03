# Question bank for the expanded harness corpus (writing_eval_harness
# coverage expansion). Every question here is self-authored for this test
# suite - none copied from any IELTS/Cambridge publication, same discipline
# as clean_corpus.py.
#
# Structure: QUESTION_BANK is a flat list; each entry carries a "variant"
# tag used to build the coverage matrix's rows. At least 2 questions per
# variant, per the coverage requirement.
#
# "topic" is a short (2-6 word) noun phrase naming what the question is
# about - used only by answer_profiles.py's p6_memorised_template profile
# to bolt a real topic onto a generic template frame (see that module's
# make_template_answer()). Every frame uses "topic" as the object of a
# preposition ("about {topic}", "connected to {topic}"), never as a bare
# grammatical subject, so the phrase's internal singular/plural form never
# has to agree with a frame verb.

QUESTION_BANK = [
    # ---- Task 1 Academic: line graph (2) ----
    {
        "id": "q_t1a_line_01",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "line_graph",
        "topic": "internet access rates",
        "question": (
            "The line graph below shows the percentage of households with "
            "internet access in three regions between 2000 and 2020. "
            "Summarise the information by selecting and reporting the main "
            "features, and make comparisons where relevant."
        ),
    },
    {
        "id": "q_t1a_line_02",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "line_graph",
        "topic": "monthly rainfall levels",
        "question": (
            "The graph below shows average monthly rainfall in three "
            "cities over the course of one year. Summarise the "
            "information by selecting and reporting the main features, "
            "and make comparisons where relevant."
        ),
    },
    # ---- Task 1 Academic: bar chart (2) ----
    {
        "id": "q_t1a_bar_01",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "bar_chart",
        "topic": "leisure activity hours",
        "question": (
            "The bar chart below shows the average number of hours per "
            "week that adults in four countries spent on leisure "
            "activities in 2015 and 2020. Summarise the information by "
            "selecting and reporting the main features, and make "
            "comparisons where relevant."
        ),
    },
    {
        "id": "q_t1a_bar_02",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "bar_chart",
        "topic": "university graduate numbers",
        "question": (
            "The chart below shows the number of new university graduates "
            "in four fields of study in one country in 2010 and 2020. "
            "Summarise the information by selecting and reporting the "
            "main features, and make comparisons where relevant."
        ),
    },
    # ---- Task 1 Academic: pie chart (2) ----
    {
        "id": "q_t1a_pie_01",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "pie_chart",
        "topic": "electricity generation sources",
        "question": (
            "The pie charts below show the main sources of electricity "
            "generation in one country in 2000 and 2020. Summarise the "
            "information by selecting and reporting the main features, "
            "and make comparisons where relevant."
        ),
    },
    {
        "id": "q_t1a_pie_02",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "pie_chart",
        "topic": "household income spending",
        "question": (
            "The pie chart below shows how a household's monthly income "
            "was spent in one country in 2022. Summarise the information "
            "by selecting and reporting the main features, and make "
            "comparisons where relevant."
        ),
    },
    # ---- Task 1 Academic: table (2) ----
    {
        "id": "q_t1a_table_01",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "table",
        "topic": "museum visitor numbers",
        "question": (
            "The table below shows the number of visitors, in thousands, "
            "to four museums in one city in 2018, 2019, and 2021. "
            "Summarise the information by selecting and reporting the "
            "main features, and make comparisons where relevant."
        ),
    },
    {
        "id": "q_t1a_table_02",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "table",
        "topic": "life expectancy figures",
        "question": (
            "The table below shows average life expectancy, in years, in "
            "five countries in 1990 and 2020. Summarise the information "
            "by selecting and reporting the main features, and make "
            "comparisons where relevant."
        ),
    },
    # ---- Task 1 Academic: process diagram (2) ----
    {
        "id": "q_t1a_process_01",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "process_diagram",
        "topic": "orange juice production",
        "question": (
            "The diagram below shows the process of producing bottled "
            "orange juice. Summarise the information by selecting and "
            "reporting the main features, and make comparisons where "
            "relevant."
        ),
    },
    {
        "id": "q_t1a_process_02",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "process_diagram",
        "topic": "the water cycle",
        "question": (
            "The diagram below shows the water cycle, from evaporation to "
            "precipitation and collection. Summarise the information by "
            "selecting and reporting the main features."
        ),
    },
    # ---- Task 1 Academic: map (change over time) (2) ----
    {
        "id": "q_t1a_map_01",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "map",
        "topic": "changes in Millbrook",
        "question": (
            "The maps below show the town of Millbrook in 1990 and the "
            "same town in 2020. Summarise the information by selecting "
            "and reporting the main features, and make comparisons where "
            "relevant."
        ),
    },
    {
        "id": "q_t1a_map_02",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "map",
        "topic": "coastal village development",
        "question": (
            "The maps below show a coastal village in 1980 and the same "
            "area in 2020, after significant tourism development. "
            "Summarise the information by selecting and reporting the "
            "main features, and make comparisons where relevant."
        ),
    },
    # ---- Task 1 Academic: mixed/multiple charts (2) ----
    {
        "id": "q_t1a_mixed_01",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "mixed_charts",
        "topic": "meal nutrient content",
        "question": (
            "The charts below show the average percentages in typical "
            "meals of three types of nutrients, all of which may be "
            "unhealthy if eaten too much. Summarise the information by "
            "selecting and reporting the main features, and make "
            "comparisons where relevant."
        ),
    },
    {
        "id": "q_t1a_mixed_02",
        "task_type": "task_1",
        "t1_variant": "academic",
        "chart_type": "mixed_charts",
        "topic": "commuting method choices",
        "question": (
            "The bar chart and pie chart below show information about "
            "commuting methods in one city: the bar chart shows the "
            "percentage using each method in 2010 and 2020, and the pie "
            "chart shows the breakdown of reasons commuters gave for "
            "choosing their main method in 2020. Summarise the "
            "information by selecting and reporting the main features, "
            "and make comparisons where relevant."
        ),
    },

    # ---- Task 1 General Training: formal letter (2) ----
    {
        "id": "q_t1gt_formal_01",
        "task_type": "task_1",
        "t1_variant": "general",
        "letter_register": "formal",
        "topic": "the damaged furniture item",
        "question": (
            "You recently bought a piece of furniture online, but it "
            "arrived damaged. Write a letter to the company. In your "
            "letter: describe the item you bought, explain what is wrong "
            "with it, say what you would like the company to do."
        ),
    },
    {
        "id": "q_t1gt_formal_02",
        "task_type": "task_1",
        "t1_variant": "general",
        "letter_register": "formal",
        "topic": "the unsatisfactory hotel stay",
        "question": (
            "You recently stayed in a hotel and were unhappy with the "
            "service. Write a letter to the hotel manager. In your "
            "letter: explain when you stayed there, describe the problems "
            "you experienced, say what you would like the manager to do."
        ),
    },
    # ---- Task 1 General Training: semi-formal letter (2) ----
    {
        "id": "q_t1gt_semiformal_01",
        "task_type": "task_1",
        "t1_variant": "general",
        "letter_register": "semi_formal",
        "topic": "the sports club improvement",
        "question": (
            "You are a member of a local sports club, and you want to "
            "suggest an improvement. Write a letter to the club "
            "secretary, who you have met a few times. In your letter: "
            "introduce yourself, describe the improvement you would like, "
            "explain how it would benefit other members."
        ),
    },
    {
        "id": "q_t1gt_semiformal_02",
        "task_type": "task_1",
        "t1_variant": "general",
        "letter_register": "semi_formal",
        "topic": "the training course",
        "question": (
            "You attended a training course run by a former colleague who "
            "now works for a training company. Write a letter to this "
            "person. In your letter: thank them for the course, mention "
            "what you found most useful, ask about future courses."
        ),
    },
    # ---- Task 1 General Training: informal letter (2) ----
    {
        "id": "q_t1gt_informal_01",
        "task_type": "task_1",
        "t1_variant": "general",
        "letter_register": "informal",
        "topic": "moving to a new city",
        "question": (
            "You are moving to a new city for work. Write a letter to a "
            "friend who lives there. In your letter: tell them your news, "
            "explain why you are moving, ask if they can help you settle "
            "in."
        ),
    },
    {
        "id": "q_t1gt_informal_02",
        "task_type": "task_1",
        "t1_variant": "general",
        "letter_register": "informal",
        "topic": "your friend's help with moving house",
        "question": (
            "A close friend recently helped you move house. Write a "
            "letter to thank them. In your letter: say what they did to "
            "help, explain how much you appreciated it, suggest a way to "
            "thank them properly."
        ),
    },

    # ---- Task 2: opinion (agree/disagree) (2, reusing 2 existing) ----
    {
        "id": "q_t2_opinion_01",
        "task_type": "task_2",
        "t2_variant": "opinion",
        "topic": "technology's effect on social skills",
        "question": (
            "Some people believe that increased use of technology in "
            "daily life has weakened face-to-face social skills, "
            "particularly among young people. To what extent do you "
            "agree or disagree?"
        ),
    },
    {
        "id": "q_t2_opinion_02",
        "task_type": "task_2",
        "t2_variant": "opinion",
        "topic": "compulsory music education",
        "question": (
            "Some people believe that children should be required to "
            "learn a musical instrument at school. To what extent do you "
            "agree or disagree?"
        ),
    },
    # ---- Task 2: discussion (both views) (2, reusing 2 existing) ----
    {
        "id": "q_t2_discussion_01",
        "task_type": "task_2",
        "t2_variant": "discussion",
        "topic": "parks versus new housing",
        "question": (
            "Some city planners argue that public parks and green spaces "
            "should be prioritised over new housing developments. Discuss "
            "both views and give your own opinion."
        ),
    },
    {
        "id": "q_t2_discussion_02",
        "task_type": "task_2",
        "t2_variant": "discussion",
        "topic": "free museum entry",
        "question": (
            "Some people think museums and art galleries should always be "
            "free to enter. Others believe visitors should pay for "
            "tickets. Discuss both views and give your own opinion."
        ),
    },
    # ---- Task 2: advantages-disadvantages (2, reusing 2 existing) ----
    {
        "id": "q_t2_advdis_01",
        "task_type": "task_2",
        "t2_variant": "advantages_disadvantages",
        "topic": "remote work",
        "question": (
            "Remote work has become far more common since 2020. Do the "
            "advantages of this trend outweigh the disadvantages?"
        ),
    },
    {
        "id": "q_t2_advdis_02",
        "task_type": "task_2",
        "t2_variant": "advantages_disadvantages",
        "topic": "the four-day working week",
        "question": (
            "Several companies have experimented with a four-day working "
            "week without reducing pay. What are the advantages and "
            "disadvantages of this practice?"
        ),
    },
    # ---- Task 2: problem-solution (2, both new) ----
    {
        "id": "q_t2_problemsolution_01",
        "task_type": "task_2",
        "t2_variant": "problem_solution",
        "topic": "traffic congestion",
        "question": (
            "Traffic congestion has become a serious problem in many "
            "large cities. What problems does this cause, and what "
            "solutions can you suggest?"
        ),
    },
    {
        "id": "q_t2_problemsolution_02",
        "task_type": "task_2",
        "t2_variant": "problem_solution",
        "topic": "the practical-skills gap among graduates",
        "question": (
            "Many students graduate from university without the "
            "practical skills employers are looking for. What problems "
            "does this cause, and what could universities do to address "
            "it?"
        ),
    },
    # ---- Task 2: two-part question (2, reusing 1 + 1 new) ----
    {
        "id": "q_t2_twopart_01",
        "task_type": "task_2",
        "t2_variant": "two_part",
        "topic": "food waste",
        "question": (
            "A large amount of food produced worldwide is wasted every "
            "year. What are the causes of this problem, and what measures "
            "could be taken to address it?"
        ),
    },
    {
        "id": "q_t2_twopart_02",
        "task_type": "task_2",
        "t2_variant": "two_part",
        "topic": "declining interest in skilled trades",
        "question": (
            "Fewer young people are choosing careers in skilled trades "
            "such as plumbing and electrical work. Why do you think this "
            "is happening, and what can be done to encourage more young "
            "people into these careers?"
        ),
    },
]

QUESTION_BANK_BY_ID = {q["id"]: q for q in QUESTION_BANK}

# Convenience groupings for building the coverage matrix's row labels.
TASK1_ACADEMIC_CHART_TYPES = [
    "line_graph", "bar_chart", "pie_chart", "table", "process_diagram",
    "map", "mixed_charts",
]
TASK1_GT_REGISTERS = ["formal", "semi_formal", "informal"]
TASK2_VARIANTS = [
    "opinion", "discussion", "advantages_disadvantages", "problem_solution",
    "two_part",
]

ALL_VARIANT_ROW_LABELS = (
    [f"t1_academic_{c}" for c in TASK1_ACADEMIC_CHART_TYPES]
    + [f"t1_gt_{r}" for r in TASK1_GT_REGISTERS]
    + [f"t2_{v}" for v in TASK2_VARIANTS]
)
