# Clean corpus for the false-positive check (writing_eval_harness Step 0).
#
# Every text here is self-authored for this test suite, on IELTS-style
# topics, in IELTS essay format - NONE of it is copied from any IELTS,
# Cambridge, British Council, or IDP publication. That distinction matters:
# this file is safe to commit; official examiner-scored scripts are not
# (see official_samples_manifest.json + official_samples_extractor.py for
# how those are handled instead - local-only, gitignored, metadata-only in
# git).
#
# Each entry is deliberately written to be genuinely free of grammar,
# lexical, and coherence errors - a careful human editor would have
# nothing to correct. That is the entire premise of the false-positive
# check: any mistake object the evaluator returns against one of these is,
# by construction, a false positive.
#
# THIS IS BAND 8-LEVEL MATERIAL, NOT BAND 9 - error-free but not
# exemplary. Confirmed directly: asked (with no band framing at all)
# what would need to improve for this text to be top-tier, and got
# genuine, specific, textually-grounded critique every time (vocabulary
# precision, sentence variety, naturalness, argument subtlety) - never
# "nothing to improve." Band 9 requires range, precision, and
# sophistication beyond correctness (official descriptor language: "wide
# range used with full flexibility and precision", "cohesion... attracts
# no attention"), which this corpus was never written for and does not
# claim to have. A top-band (9) check run against this corpus will
# correctly fail - that is not a scoring bug, and this corpus is not a
# Band 9 reference set. Do not repeat the assumption that "error-free"
# means "Band 9."

CLEAN_CORPUS = [
    {
        "id": "clean_t2_01",
        "task_type": "task_2",
        "topic": "technology and social skills",
        "question": (
            "Some people believe that increased use of technology in daily "
            "life has weakened face-to-face social skills, particularly "
            "among young people. To what extent do you agree or disagree?"
        ),
        "text": (
            "It is often claimed that technology has eroded young people's "
            "ability to interact with one another in person, and I largely "
            "agree with this view, although I believe the picture is more "
            "nuanced than it first appears.\n\n"
            "On one hand, there is clear evidence that reliance on digital "
            "communication has changed how young people relate to each "
            "other. Text messages and social media allow a person to "
            "compose and edit a response before sending it, which removes "
            "the spontaneity and vulnerability that characterise face-to-"
            "face conversation. As a result, some young people report "
            "feeling anxious in situations that require them to respond "
            "immediately, such as a job interview or a chance encounter "
            "with a stranger. Over time, this avoidance can weaken the "
            "very skills that only practice can build.\n\n"
            "On the other hand, technology has not eliminated social "
            "interaction so much as redirected it. Many young people "
            "maintain rich friendships that begin online and later move "
            "into the physical world, and video calls allow people to "
            "read facial expressions and tone of voice in ways that text "
            "alone cannot capture. It would therefore be inaccurate to "
            "claim that technology has replaced social skills altogether; "
            "rather, it has changed which skills are exercised most often.\n\n"
            "In my view, the responsibility lies less with technology "
            "itself than with how it is used. Parents and schools that "
            "actively encourage unstructured, in-person time among young "
            "people can offset much of the risk, while treating every "
            "screen as inherently harmful oversimplifies a genuinely "
            "complex issue.\n\n"
            "In conclusion, while I accept that heavy reliance on digital "
            "communication can weaken certain in-person social skills, I "
            "do not believe this outcome is inevitable, and thoughtful "
            "guidance from parents and educators can help young people "
            "benefit from technology without losing the ability to "
            "connect with others directly."
        ),
    },
    {
        "id": "clean_t2_02",
        "task_type": "task_2",
        "topic": "urban green space",
        "question": (
            "Some city planners argue that public parks and green spaces "
            "should be prioritised over new housing developments. Discuss "
            "both views and give your own opinion."
        ),
        "text": (
            "As cities continue to expand, planners face a persistent "
            "tension between building enough housing and preserving green "
            "space for residents. Both priorities have strong justification, "
            "and I believe the most sensible approach is to pursue them "
            "together rather than treating the choice as binary.\n\n"
            "Those who favour prioritising housing point to the acute "
            "shortages many cities now face. Rising rents have pushed "
            "families further from city centres, lengthening commutes and "
            "straining household budgets. From this perspective, land "
            "that could house hundreds of families should not be reserved "
            "for recreational use, however pleasant that use might be.\n\n"
            "Advocates for green space, meanwhile, point to its broader "
            "value. Parks lower local temperatures during heatwaves, "
            "provide a free venue for exercise and social contact, and "
            "have been linked in numerous studies to improved mental "
            "health among nearby residents. Removing green space to build "
            "housing may solve one problem while quietly creating others, "
            "particularly for people who cannot afford private gardens or "
            "gym memberships.\n\n"
            "In my opinion, the apparent conflict between these two goals "
            "is often exaggerated. Many cities have successfully increased "
            "housing density by building upward rather than outward, "
            "which allows more residents to be housed without consuming "
            "additional land. Rooftop gardens, pocket parks, and green "
            "corridors along waterways can also deliver many of the "
            "benefits of larger parks within a smaller footprint.\n\n"
            "In conclusion, rather than forcing planners to choose between "
            "housing and green space, cities that invest in creative, "
            "higher-density design can expand both simultaneously, "
            "delivering the practical and psychological benefits that "
            "each provides."
        ),
    },
    {
        "id": "clean_t2_03",
        "task_type": "task_2",
        "topic": "remote work",
        "question": (
            "Remote work has become far more common since 2020. Do the "
            "advantages of this trend outweigh the disadvantages?"
        ),
        "text": (
            "The shift toward remote work that began in 2020 has "
            "fundamentally altered how millions of people structure their "
            "working lives, and on balance I believe its advantages "
            "outweigh its drawbacks, though the disadvantages are real "
            "enough to deserve careful attention.\n\n"
            "The benefits of remote work are considerable. Employees no "
            "longer lose hours each week commuting, time that can instead "
            "be spent with family, exercising, or simply resting. "
            "Employers, in turn, can recruit talent from a much wider "
            "geographic pool rather than limiting themselves to people "
            "willing to relocate. Many workers also report greater "
            "autonomy over their schedules, which allows them to manage "
            "personal responsibilities alongside professional ones far "
            "more flexibly than a traditional office schedule permits.\n\n"
            "Nevertheless, remote work carries genuine costs. Spontaneous "
            "collaboration, the kind that often happens when colleagues "
            "overhear one another or meet by chance near a shared coffee "
            "machine, becomes harder to replicate over a screen. New "
            "employees in particular can struggle to absorb an "
            "organisation's culture without regular in-person contact, "
            "and some workers report feeling isolated when their entire "
            "professional life is mediated through a laptop.\n\n"
            "Weighing these considerations, I believe the productivity "
            "and wellbeing gains reported by most remote workers, combined "
            "with employers' expanded access to talent, generally exceed "
            "the costs, particularly when organisations make deliberate "
            "efforts to bring teams together periodically in person.\n\n"
            "In conclusion, although remote work introduces real "
            "challenges around collaboration and belonging, its benefits "
            "for both individuals and organisations are substantial "
            "enough that, overall, the trend represents a net positive "
            "change to how people work."
        ),
    },
    {
        "id": "clean_t2_04",
        "task_type": "task_2",
        "topic": "museum funding",
        "question": (
            "Some people think museums and art galleries should always be "
            "free to enter. Others believe visitors should pay for tickets. "
            "Discuss both views and give your own opinion."
        ),
        "text": (
            "Whether museums and galleries should charge for admission is "
            "a question that touches on both cultural access and "
            "practical funding, and reasonable people disagree about how "
            "to balance the two.\n\n"
            "Supporters of free admission argue that culture belongs to "
            "everyone, not only to those who can afford a ticket. When "
            "entry is free, a family on a limited budget can visit as "
            "easily as a wealthy tourist, and children in particular "
            "benefit from repeated, low-stakes exposure to art and "
            "history rather than a single rushed visit. Free admission "
            "also tends to increase overall visitor numbers, which can "
            "strengthen a museum's role as a genuine public institution "
            "rather than a niche attraction.\n\n"
            "Those in favour of paid entry, however, point to the "
            "financial realities museums face. Maintaining collections, "
            "preserving fragile artefacts, and mounting new exhibitions "
            "all require significant funding, and ticket revenue can "
            "reduce a museum's dependence on government grants or private "
            "donors, which may come with their own conditions attached. "
            "A reasonable ticket price, in this view, is a fair exchange "
            "for the experience and knowledge on offer.\n\n"
            "In my opinion, a hybrid approach best serves both goals. "
            "Permanent collections, which represent a shared cultural "
            "inheritance, are well suited to free admission, while "
            "temporary or especially elaborate exhibitions can reasonably "
            "charge a fee to help cover their additional costs. This way, "
            "access is preserved for those who need it most, while "
            "museums retain a meaningful source of independent funding.\n\n"
            "In conclusion, rather than choosing one policy for every "
            "kind of exhibition, museums that combine free general access "
            "with selective ticketing for special events are more likely "
            "to remain both accessible and financially sustainable."
        ),
    },
    {
        "id": "clean_t2_05",
        "task_type": "task_2",
        "topic": "standardised testing in schools",
        "question": (
            "Standardised tests are often used to measure students' "
            "academic ability. Do the advantages of this approach outweigh "
            "the disadvantages?"
        ),
        "text": (
            "Standardised testing has long been the primary tool schools "
            "use to measure academic performance, and while it offers "
            "real practical benefits, I believe its disadvantages are "
            "significant enough that its advantages do not clearly "
            "outweigh them.\n\n"
            "The case for standardised testing rests largely on "
            "consistency. Because every student answers the same "
            "questions under the same conditions, results can be compared "
            "fairly across schools, regions, and even countries, which "
            "allows policymakers to identify where additional resources "
            "are needed. Standardised tests are also relatively "
            "inexpensive to administer at scale compared with more "
            "individualised forms of assessment.\n\n"
            "However, this consistency comes at a cost. A single test "
            "score cannot capture creativity, collaboration, or the kind "
            "of sustained critical thinking that develops over an entire "
            "term of coursework. Students who struggle with timed, high-"
            "pressure conditions may perform poorly despite having a "
            "genuine grasp of the material, while teachers, under "
            "pressure to raise scores, can end up narrowing their "
            "teaching to whatever the test happens to measure, at the "
            "expense of a richer curriculum.\n\n"
            "Given these drawbacks, I believe standardised tests are most "
            "useful as one input among several rather than as the primary "
            "measure of a student's ability. Portfolios of student work, "
            "teacher assessments built up over the course of a term, and "
            "project-based evaluation all capture dimensions of learning "
            "that a single test cannot.\n\n"
            "In conclusion, although standardised testing offers "
            "efficiency and comparability, its narrow view of what "
            "counts as achievement means that, on balance, its "
            "disadvantages outweigh its administrative convenience."
        ),
    },
    {
        "id": "clean_t2_06",
        "task_type": "task_2",
        "topic": "food waste",
        "question": (
            "A large amount of food produced worldwide is wasted every "
            "year. What are the causes of this problem, and what measures "
            "could be taken to address it?"
        ),
        "text": (
            "Food waste on a global scale has become an issue that "
            "concerns economists, environmentalists, and policymakers "
            "alike, and understanding its causes is the first step toward "
            "reducing it.\n\n"
            "One major cause lies at the retail level, where supermarkets "
            "often reject fruit and vegetables that fail to meet strict "
            "cosmetic standards, even though the produce itself remains "
            "perfectly edible. Confusing date labelling compounds the "
            "problem, since many consumers discard food once a \"best "
            "before\" date has passed, mistaking a quality indicator for "
            "a genuine safety warning. At the household level, over-"
            "purchasing driven by bulk discounts frequently leads to food "
            "spoiling before it can be used.\n\n"
            "Several measures could meaningfully reduce this waste. "
            "Retailers could relax cosmetic standards and sell imperfect "
            "produce at a discount rather than discarding it outright, an "
            "approach some supermarket chains have already adopted with "
            "encouraging results. Clearer, standardised labelling that "
            "distinguishes safety dates from quality dates would help "
            "consumers make better decisions about what to keep and what "
            "to discard. Public education campaigns, meanwhile, could "
            "teach households practical skills such as meal planning and "
            "proper food storage, both of which directly reduce the "
            "amount of food that spoils unused.\n\n"
            "Governments also have a role to play. Tax incentives for "
            "businesses that donate surplus food to charities, combined "
            "with clearer legal protection for such donations, would "
            "remove a common excuse for discarding edible food rather "
            "than redistributing it.\n\n"
            "In conclusion, food waste arises from a combination of "
            "retail practices, consumer misunderstanding, and household "
            "habits, and addressing it will require coordinated action "
            "from retailers, educators, and governments rather than any "
            "single intervention alone."
        ),
    },
    {
        "id": "clean_t2_07",
        "task_type": "task_2",
        "topic": "learning a musical instrument",
        "question": (
            "Some people believe that children should be required to "
            "learn a musical instrument at school. To what extent do you "
            "agree or disagree?"
        ),
        "text": (
            "Whether schools should require every child to learn a "
            "musical instrument is a question that raises both "
            "educational and practical concerns, and I only partially "
            "agree with making such instruction compulsory.\n\n"
            "There is genuine evidence supporting music education. "
            "Learning an instrument demands sustained concentration, "
            "discipline, and the ability to interpret abstract notation, "
            "skills that plausibly transfer to other academic subjects. "
            "Music also offers a form of creative expression that many "
            "children might never otherwise encounter, and some students "
            "who struggle in traditional academic subjects find "
            "unexpected confidence through musical achievement.\n\n"
            "At the same time, mandating instrumental instruction for "
            "every child overlooks the wide variation in individual "
            "interests and aptitudes. A student with little natural "
            "inclination toward music may experience the requirement as "
            "a source of stress rather than enrichment, and the time "
            "spent on compulsory lessons could, for that student, be "
            "better directed toward an activity more suited to their "
            "talents, whether that is visual art, sport, or another "
            "creative pursuit entirely.\n\n"
            "A more balanced policy, in my view, would guarantee every "
            "child meaningful exposure to music, through general "
            "listening, singing, and basic music theory, while making "
            "sustained instrumental study an elective option rather than "
            "an obligation. This preserves the benefits of early musical "
            "exposure for all students while allowing those with a "
            "genuine interest to pursue an instrument in greater depth.\n\n"
            "In conclusion, while I recognise the real benefits of "
            "learning an instrument, I do not believe every child should "
            "be required to do so, and a system that offers broad "
            "exposure alongside optional deeper study would serve "
            "students better than a blanket requirement."
        ),
    },
    {
        "id": "clean_t2_08",
        "task_type": "task_2",
        "topic": "advertising aimed at children",
        "question": (
            "Some countries have introduced restrictions on advertising "
            "aimed at children. Do you think this is a positive or "
            "negative development?"
        ),
        "text": (
            "Restrictions on advertising aimed at children have become "
            "increasingly common in recent years, and I regard this as a "
            "largely positive development, even though it raises some "
            "legitimate concerns worth acknowledging.\n\n"
            "Young children are generally unable to distinguish between "
            "factual content and persuasive marketing in the way that "
            "adults can, which makes them particularly susceptible to "
            "advertising for sugary snacks, expensive toys, and other "
            "products that may not serve their genuine interests. "
            "Limiting such advertising, especially during programming "
            "aimed specifically at young audiences, reduces the pressure "
            "children place on parents and can contribute to healthier "
            "habits, particularly around food.\n\n"
            "Critics of these restrictions argue that they interfere with "
            "free commercial speech and that responsibility for managing "
            "a child's exposure to advertising should rest with parents "
            "rather than regulators. There is some merit to this "
            "position; no regulation can fully replace attentive "
            "parenting, and children will inevitably encounter "
            "advertising through other channels regardless of what "
            "broadcasters are permitted to show.\n\n"
            "Nevertheless, I believe the imbalance of power between a "
            "sophisticated advertising industry and a young child "
            "justifies some degree of regulatory protection, in the same "
            "way that other protections exist for groups less able to "
            "advocate for their own interests. Sensible restrictions do "
            "not eliminate parental responsibility; they simply reduce "
            "the volume of persuasive material parents must actively "
            "counter.\n\n"
            "In conclusion, while such restrictions are not a complete "
            "solution on their own, I view them as a reasonable and "
            "largely positive step that works alongside, rather than "
            "instead of, parental guidance."
        ),
    },
    {
        "id": "clean_t2_09",
        "task_type": "task_2",
        "topic": "four-day work week",
        "question": (
            "Several companies have experimented with a four-day working "
            "week without reducing pay. What are the advantages and "
            "disadvantages of this practice?"
        ),
        "text": (
            "The idea of compressing a standard working week into four "
            "days rather than five, without any corresponding reduction "
            "in pay, has moved from a fringe experiment to a genuine "
            "policy discussion in several countries, and it brings both "
            "clear advantages and real challenges.\n\n"
            "Among the advantages, employees consistently report improved "
            "wellbeing when granted an additional day away from work each "
            "week, with more time available for rest, family, and "
            "personal pursuits. Several trials have also found that "
            "productivity per hour worked actually increased under a "
            "four-day arrangement, suggesting that employees compensate "
            "for the shorter week by working more efficiently during the "
            "days they are present. Reduced commuting, since staff travel "
            "to work one fewer day, brings modest environmental benefits "
            "as well.\n\n"
            "The disadvantages, however, are not trivial. Certain "
            "industries, particularly those involving continuous customer "
            "service or shift-based operations such as healthcare, cannot "
            "easily compress their schedules without hiring additional "
            "staff, which raises costs that a four-day policy was "
            "originally intended to avoid. Some employees also report "
            "that meeting the same workload in fewer days increases "
            "stress on the days they do work, offsetting some of the "
            "wellbeing gains achieved on their day off.\n\n"
            "On balance, the practice appears best suited to roles where "
            "output can be measured independently of hours logged, such "
            "as many office-based positions, while being harder to "
            "implement in sectors that depend on constant staffing.\n\n"
            "In conclusion, a four-day working week offers meaningful "
            "benefits for employee wellbeing and, in many cases, "
            "productivity, but its disadvantages mean it is unlikely to "
            "suit every industry equally, and careful adaptation to each "
            "sector's demands will be necessary for it to succeed widely."
        ),
    },
    {
        "id": "clean_t2_10",
        "task_type": "task_2",
        "topic": "space exploration funding",
        "question": (
            "Some people argue that governments spend too much money on "
            "space exploration when there are more urgent problems on "
            "Earth. To what extent do you agree?"
        ),
        "text": (
            "Government spending on space exploration is frequently "
            "criticised as an extravagance at a time when poverty, "
            "healthcare, and climate change demand urgent attention, but "
            "I only partly agree with this criticism.\n\n"
            "It is true that space programmes require substantial public "
            "funding, and critics rightly point out that this money could "
            "instead be directed toward hospitals, schools, or renewable "
            "energy infrastructure, all of which address problems "
            "affecting people's lives immediately. When public services "
            "are underfunded, allocating billions to missions with no "
            "guaranteed short-term benefit can understandably appear to "
            "be a matter of misplaced priorities.\n\n"
            "However, this framing overstates the trade-off involved. "
            "Space agencies typically receive a comparatively small "
            "share of national budgets, far smaller than spending on "
            "areas such as defence, and many technologies developed for "
            "space missions, including satellite-based weather "
            "forecasting, GPS navigation, and advances in materials "
            "science, have gone on to deliver enormous practical benefit "
            "on Earth. Space-based climate monitoring, in particular, "
            "provides data that is directly relevant to addressing the "
            "very environmental problems critics cite as more urgent.\n\n"
            "In my view, the real question is not whether space "
            "exploration deserves any funding, but whether that funding "
            "is proportionate and well managed. A modest, carefully "
            "justified space budget need not come at the expense of "
            "urgent domestic priorities, provided governments do not "
            "treat the two as competing for the same limited resources "
            "without genuine scrutiny.\n\n"
            "In conclusion, while I accept that public spending must be "
            "carefully prioritised, I do not believe space exploration "
            "and urgent domestic needs are inherently in conflict, and a "
            "measured space budget can coexist with strong investment "
            "in more immediate concerns."
        ),
    },
    {
        "id": "clean_t1_01",
        "task_type": "task_1",
        "topic": "generic bar chart description (no real chart - text-only structural test)",
        "question": (
            "The bar chart below shows the average number of hours per "
            "week that adults in four countries spent on leisure "
            "activities in 2015 and 2020. Summarise the information by "
            "selecting and reporting the main features, and make "
            "comparisons where relevant."
        ),
        "text": (
            "The bar chart compares the average number of hours per week "
            "that adults in four countries devoted to leisure activities "
            "in 2015 and 2020.\n\n"
            "Overall, leisure time increased in every country over the "
            "five-year period, although the size of the increase varied "
            "considerably, and adults in Country A consistently spent "
            "more time on leisure than those in the other three countries "
            "in both years.\n\n"
            "In 2015, adults in Country A reported the highest average, "
            "at around eighteen hours per week, followed by Country B at "
            "fifteen hours. Countries C and D were noticeably lower, at "
            "eleven and nine hours respectively. By 2020, every country "
            "had seen an increase, but the gap between the highest and "
            "lowest figures had narrowed slightly. Country A rose to "
            "twenty-one hours, while Country D, which had recorded the "
            "smallest figure in 2015, showed the largest proportional "
            "increase, reaching thirteen hours.\n\n"
            "Countries B and C followed a broadly similar pattern to one "
            "another, each increasing by roughly three hours over the "
            "period, which kept their relative positions unchanged "
            "despite the overall rise. Throughout both years, the "
            "ranking of the four countries from highest to lowest "
            "remained the same, with Country A leading and Country D "
            "trailing behind, even as the absolute gap between them "
            "narrowed."
        ),
    },
    {
        "id": "clean_t1_02",
        "task_type": "task_1",
        "topic": "generic line graph description (no real chart - text-only structural test)",
        "question": (
            "The line graph below shows the percentage of households with "
            "internet access in three regions between 2000 and 2020. "
            "Summarise the information by selecting and reporting the "
            "main features, and make comparisons where relevant."
        ),
        "text": (
            "The line graph illustrates changes in the percentage of "
            "households with internet access across three regions "
            "between 2000 and 2020.\n\n"
            "Overall, internet access expanded dramatically in all three "
            "regions over the twenty-year period, with the most rapid "
            "growth occurring in the Southern region, which started from "
            "the lowest base but had nearly caught up with the other two "
            "regions by the end of the period.\n\n"
            "In 2000, the Northern region led by a wide margin, with "
            "around thirty percent of households connected, compared "
            "with roughly eighteen percent in the Eastern region and just "
            "six percent in the Southern region. Growth in the Northern "
            "region was steady but gradual throughout the period, "
            "reaching approximately eighty-five percent by 2020.\n\n"
            "The Eastern region showed a similar upward trend, though "
            "starting from a lower point, and its line ran consistently "
            "below the Northern region's throughout the two decades "
            "before reaching around seventy-eight percent by 2020. The "
            "Southern region's trajectory stood out from the other two: "
            "growth was slow until around 2010, after which the rate of "
            "increase accelerated sharply, and household internet access "
            "in that region climbed to roughly seventy-two percent by "
            "2020, closing most of the gap that had separated it from "
            "the other regions at the start of the period."
        ),
    },
    {
        "id": "clean_t1_03",
        "task_type": "task_1",
        "topic": "generic process diagram description",
        "question": (
            "The diagram below shows the process of producing bottled "
            "orange juice. Summarise the information by selecting and "
            "reporting the main features, and make comparisons where "
            "relevant."
        ),
        "text": (
            "The diagram illustrates the stages involved in producing "
            "bottled orange juice, from the initial harvesting of oranges "
            "through to the final packaged product ready for sale.\n\n"
            "Overall, the process consists of six main stages, beginning "
            "with harvesting and ending with distribution, and involves "
            "both mechanical processing and quality inspection at "
            "several points along the way.\n\n"
            "The process begins with oranges being harvested from "
            "orchards and transported to a processing facility, where "
            "they are first washed to remove dirt and any surface "
            "residue. Once cleaned, the oranges pass through a sorting "
            "stage, during which damaged or unripe fruit is removed "
            "before the remaining oranges move on to extraction.\n\n"
            "During extraction, the oranges are pressed mechanically to "
            "separate the juice from the peel and pulp, and the "
            "resulting liquid is then filtered to remove any remaining "
            "solid particles. Following filtration, the juice undergoes "
            "pasteurisation, a heating process that eliminates harmful "
            "bacteria while preserving the juice's flavour, before being "
            "cooled rapidly in preparation for bottling.\n\n"
            "In the final stage, the pasteurised juice is bottled, "
            "sealed, and labelled, after which the finished bottles are "
            "packed into crates and distributed to retailers for sale. "
            "Throughout the process, quality control checks are carried "
            "out at multiple stages to ensure that only juice meeting "
            "the required standard reaches the bottling stage."
        ),
    },
    {
        "id": "clean_t1_04",
        "task_type": "task_1",
        "topic": "generic pie chart description",
        "question": (
            "The pie charts below show the main sources of electricity "
            "generation in one country in 2000 and 2020. Summarise the "
            "information by selecting and reporting the main features, "
            "and make comparisons where relevant."
        ),
        "text": (
            "The two pie charts compare the main sources of electricity "
            "generation in one country in 2000 and 2020.\n\n"
            "Overall, the country's reliance on coal fell sharply over "
            "the twenty-year period, while the combined contribution of "
            "renewable sources such as wind and solar rose substantially, "
            "marking a clear shift away from fossil fuels toward cleaner "
            "alternatives.\n\n"
            "In 2000, coal was by far the dominant source, accounting for "
            "just over half of all electricity generated, with natural "
            "gas contributing a further quarter. Nuclear power made up "
            "around fifteen percent, while renewable sources together "
            "represented a relatively small proportion of the total, at "
            "roughly eight percent.\n\n"
            "By 2020, the picture had changed considerably. Coal's share "
            "had fallen to under twenty percent, while renewables had "
            "expanded to account for close to a third of total "
            "generation, driven mainly by growth in wind power. Natural "
            "gas remained fairly stable as a proportion of the total, "
            "changing only slightly from its 2000 level, whereas nuclear "
            "power's share increased modestly, reaching around twenty "
            "percent by the end of the period.\n\n"
            "Taken together, the two charts show a clear reordering of "
            "the country's electricity mix, with renewables and nuclear "
            "power together accounting for roughly half of all "
            "generation by 2020, compared with less than a quarter "
            "combined two decades earlier."
        ),
    },
    {
        "id": "clean_t1_05",
        "task_type": "task_1",
        "topic": "generic table description",
        "question": (
            "The table below shows the number of visitors, in thousands, "
            "to four museums in one city in 2018, 2019, and 2021. "
            "Summarise the information by selecting and reporting the "
            "main features, and make comparisons where relevant."
        ),
        "text": (
            "The table presents visitor numbers, in thousands, at four "
            "museums in one city over three separate years: 2018, 2019, "
            "and 2021.\n\n"
            "Overall, visitor numbers at every museum fell sharply "
            "between 2019 and 2021, and although the History Museum "
            "attracted the most visitors throughout the period, the "
            "scale of the decline varied considerably between "
            "institutions.\n\n"
            "In 2018, the History Museum recorded the highest attendance, "
            "at 420,000 visitors, followed by the Science Museum with "
            "310,000. The Art Gallery and the Natural History Museum "
            "attracted fewer visitors that year, at 260,000 and 190,000 "
            "respectively. Attendance at all four venues rose slightly "
            "in 2019, with the History Museum reaching its peak figure "
            "of 450,000 and the other three museums each recording "
            "modest gains of between ten and thirty thousand visitors "
            "compared with the previous year.\n\n"
            "By 2021, however, every museum had experienced a steep "
            "decline. The Natural History Museum was affected most "
            "severely in relative terms, falling to just 60,000 "
            "visitors, less than a third of its 2019 figure. The History "
            "Museum, despite remaining the most visited venue, still saw "
            "attendance more than halve, dropping to 190,000. The Art "
            "Gallery and Science Museum followed a similar pattern, with "
            "both losing more than half of their 2019 visitor numbers by "
            "2021."
        ),
    },
    {
        "id": "clean_t1_06",
        "task_type": "task_1",
        "topic": "General Training letter - complaint",
        "question": (
            "You recently bought a piece of furniture online, but it "
            "arrived damaged. Write a letter to the company. In your "
            "letter: describe the item you bought, explain what is wrong "
            "with it, say what you would like the company to do."
        ),
        "text": (
            "Dear Sir or Madam,\n\n"
            "I am writing to inform you of a problem with a recent order "
            "I placed through your website. On the fifteenth of last "
            "month, I purchased a wooden bookshelf, order number 48213, "
            "which was delivered to my home four days later.\n\n"
            "Unfortunately, when I unpacked the item, I discovered that "
            "one of the side panels had a long crack running along its "
            "length, and one of the shelf brackets was missing entirely "
            "from the box. As a result, I have not been able to assemble "
            "the bookshelf, and it currently remains in its original "
            "packaging in my hallway.\n\n"
            "I have attached photographs showing both the crack in the "
            "panel and the packaging, which appeared undamaged on the "
            "outside, so I do not believe the fault occurred during "
            "transit with the courier. I would be grateful if you could "
            "arrange for a replacement bookshelf to be sent as soon as "
            "possible, along with a prepaid return label so that I can "
            "send back the damaged item.\n\n"
            "I have been a satisfied customer of your company for "
            "several years and would appreciate a prompt resolution to "
            "this issue. Please let me know if you require any further "
            "information from me in order to process this request.\n\n"
            "Yours faithfully,\n"
            "A. Whitfield"
        ),
    },
]

# Convenience: id -> entry, for the harness to reference by name.
CLEAN_CORPUS_BY_ID = {entry["id"]: entry for entry in CLEAN_CORPUS}
