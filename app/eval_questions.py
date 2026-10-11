"""The question sets used by  python -m app.evaluate.

Each entry: (id, kind, scope, question, must_match, sources)
  kind        "answer"  the documents answer it: every pattern in must_match has to appear in the answer
              "either"  a correct answer OR an honest refusal is acceptable (the document has no general definition)
              "refuse"  the documents do not answer it: the system has to say so
  scope       part of a document's file name to restrict the question to, or None to search ALL uploaded documents
  must_match  regex patterns (case-insensitive); a pattern may list alternatives with |
  sources     (file name part, page or None) where the answer really is. Used to separate retrieval failures from
              answer failures. Empty for questions that should be refused.
The checks are keyword based and approximate: always read the printed answers as well.
"""

K = "keph101"
LEAVE, HEALTH, VEHICLE, SECURITY, REFUND, REMOTE = (
    "employee_leave", "health_insurance", "vehicle_insurance", "information_security", "refund_cancellation", "remote_work")

# ----------------------------------------------------------------------------------------------- CBSE physics chapter
CBSE = [
    (1, "either", K, "what is the defination of mass", [r"kilogram"], [(K, 2)]),
    (2, "answer", K, "How many base units does SI have, and what two extra units are defined besides them?", [r"seven|\b7\b", r"plane angle|radian", r"solid angle|steradian"], [(K, 2)]),
    (3, "answer", K, "Who developed the SI system and when was it last revised?", [r"BIPM|Bureau International", r"2018"], [(K, 1)]),
    (4, "answer", K, "How many significant figures does 4.700 m have, and does changing it to 0.004700 km change that?", [r"four|\b4\b", r"not change|does not|no change|same|cannot change|doesn.t"], [(K, 4)]),
    (5, "answer", K, "What is the sum of 436.32 g, 227.2 g and 0.301 g to the correct precision, and why?", [r"663\.8(?!\d)", r"decimal place"], [(K, 5)]),
    (6, "answer", K, "Why is the density of 4.237 g in 2.51 cm3 reported as 1.69 g cm-3 and not 1.68804780876?", [r"three|\b3\b", r"significant"], [(K, 5)]),
    (7, "answer", K, "Express the density of 5.74 g occupying 1.2 cm3 with correct significant figures.", [r"4\.8(?!\d)"], [(K, 6)]),
    (8, "answer", K, "Which of the kinetic energy formulas in Example 1.4 can be ruled out by dimensional analysis, and why can't it decide between (b) and (d)?", [r"\(a\)|m2 ?v3|m²v³|\ba,", r"\(c\)|K ?= ?ma|\bc,|and \(?c", r"\(e\)|\be\b", r"same dimension|correct dimension|dimensions? (are|is) the same|\[M L2 T-2\]|\[ML2T-2\]|M L²"], [(K, 9)]),
    (9, "answer", K, "What are the dimensions of force and how are they derived?", [r"M\s*L\s*T\s*(\^|–|-|−|⁻)?\s*2"], [(K, 7)]),
    (10, "answer", K, "Does dimensional consistency prove an equation is correct?", [r"\bno\b|not guarantee|does not|cannot|isn.t|not prove"], [(K, 8)]),
    (11, "answer", K, "Why is the reciprocal of 9.58 a problem when rounded in the middle of a calculation, and what is the recommended fix?", [r"0\.104", r"9\.62", r"extra digit|one more|additional digit"], [(K, 7)]),
    (12, "answer", K, "The diameter of the earth is how many orders of magnitude larger than a hydrogen atom?", [r"\b17\b"], [(K, 4)]),
    (13, "answer", K, "Compare how the metre and the second are defined. What fixed constant does each use?", [r"299792458|speed of light", r"9192631770|caesium|cesium"], [(K, 2)]),
    (14, "answer", K, "Each side of a cube is 7.203 m. Give the surface area and volume with the right significant figures.", [r"311\.3", r"373\.7"], [(K, 6)]),
    (15, "answer", K, "Which units were the base units of length, mass and time in the CGS, FPS and MKS systems?", [r"centimet", r"\bfoot\b|\bfeet\b", r"pound"], [(K, 1)]),
    (16, "refuse", K, "What is Newton's second law of motion in words?", [], []),
    (17, "refuse", K, "What is the value of acceleration due to gravity on the Moon?", [], []),
    (18, "refuse", K, "What is Ohm's law?", [], []),
    (19, "refuse", K, "Who is the author of this textbook?", [], []),
    (20, "refuse", K, "What is the capital of France?", [], []),
]

# ----------------------------------------------------------------------------------------- the six policy documents
# These are asked against ALL uploaded documents (including the physics chapter), like a real user would.
POLICIES = [
    # employee leave
    (101, "answer", None, "How many days of paid annual leave do full-time employees get?", [r"\b24\b"], [(LEAVE, 1)]),
    (102, "answer", None, "How many unused annual leave days can I carry forward to next year?", [r"\b5\b|five"], [(LEAVE, 1)]),
    (103, "answer", None, "When do I need a medical certificate for sick leave?", [r"more than 3|longer than 3|over 3|3 consecutive|three consecutive|exceed"], [(LEAVE, 1)]),
    (104, "answer", None, "How long is parental leave for a secondary caregiver?", [r"\b4\b|four", r"week"], [(LEAVE, 2)]),
    (105, "answer", None, "I want to take 5 days of leave. How early must I ask my manager?", [r"14 days|fourteen|two weeks"], [(LEAVE, 2)]),
    # health insurance
    (106, "answer", None, "What is the waiting period for pre-existing conditions?", [r"24 months|two years|24-month"], [(HEALTH, 1)]),
    (107, "answer", None, "Are accidental injuries covered from the first day of the health policy?", [r"\byes\b|day one|first day|from day|covered"], [(HEALTH, 1)]),
    (108, "answer", None, "How long before planned hospitalisation must cashless pre-authorisation be obtained?", [r"48 hours|48-hour|\b48\b"], [(HEALTH, 2)]),
    (109, "answer", None, "How long does the health insurer take to settle a claim?", [r"\b21\b"], [(HEALTH, 2)]),
    (110, "answer", None, "Does the health policy cover cosmetic surgery?", [r"\bno\b|not covered|exclu|does not cover|not cover"], [(HEALTH, 2)]),
    (111, "answer", None, "What is the initial waiting period for an illness in the health policy?", [r"\b30\b"], [(HEALTH, 1)]),
    (112, "answer", None, "waitng periode for pre-existing condtions in health insurence", [r"24 months|two years|24-month"], [(HEALTH, 1)]),
    # vehicle insurance
    (113, "answer", None, "Within how many hours must I report a car accident to the insurer?", [r"\b48\b"], [(VEHICLE, 1)]),
    (114, "answer", None, "What is the no-claim bonus and when does it apply?", [r"20 ?%|20 percent|twenty", r"claim-free|claim free|no claim|without"], [(VEHICLE, 2)]),
    (115, "answer", None, "How soon is a surveyor assigned after my vehicle claim is acknowledged?", [r"\b5\b|five", r"business day"], [(VEHICLE, 2)]),
    (116, "answer", None, "What must I do if my vehicle is stolen?", [r"police", r"insurer"], [(VEHICLE, 1)]),
    # across documents
    (117, "answer", None, "Which acknowledges claims faster, the health insurer or the vehicle insurer, and by how much?", [r"vehicle", r"one (business )?day|\b1\b|\b2\b|two|\b3\b|three"], [(VEHICLE, 2), (HEALTH, 2)]),
    (118, "answer", None, "What is the claim acknowledgement time?", [r"\b3\b|three", r"\b2\b|two"], [(VEHICLE, 2), (HEALTH, 2)]),
    # information security
    (119, "answer", None, "How long must a password be?", [r"\b12\b|twelve"], [(SECURITY, 2)]),
    (120, "answer", None, "How quickly must a suspected security incident be reported?", [r"24 hours|\b24\b"], [(SECURITY, 2)]),
    (121, "answer", None, "How often must access to sensitive shared folders be reviewed?", [r"quarter"], [(SECURITY, 1)]),
    (122, "answer", None, "When must the account of an employee who leaves be disabled?", [r"last working day|last day"], [(SECURITY, 1)]),
    (123, "answer", None, "Is multi-factor authentication required for remote access?", [r"\byes\b|mandatory|required"], [(SECURITY, 2)]),
    # refunds
    (124, "answer", None, "Within how many days can I return an item after delivery?", [r"\b14\b"], [(REFUND, 1)]),
    (125, "answer", None, "Can I return a gift card?", [r"\bno\b|not eligible|cannot|can.t|not be returned|not"], [(REFUND, 1)]),
    (126, "answer", None, "How long does a refund take to arrive?", [r"\b7\b|seven", r"business day"], [(REFUND, 1)]),
    (127, "answer", None, "Can I cancel my order after it has been dispatched?", [r"return|not free|cannot|can.t|\bno\b|before dispatch"], [(REFUND, 2)]),
    # remote work
    (128, "answer", None, "How many days per week can employees work remotely?", [r"\b3\b|three"], [(REMOTE, None)]),
    (129, "answer", None, "Can someone who is still on probation work from home?", [r"\bno\b|not|cannot|office"], [(REMOTE, None)]),
    (130, "answer", None, "How much is the home office allowance and how soon must receipts be submitted?", [r"15,?000", r"30 days"], [(REMOTE, None)]),
    (131, "answer", None, "What are the core hours when remote employees must be available?", [r"11", r"16"], [(REMOTE, None)]),
    (132, "answer", None, "Can I connect to company data from public Wi-Fi without the VPN?", [r"not allowed|cannot|can.t|\bno\b|prohibit|not permitted"], [(REMOTE, None)]),
    (133, "answer", None, "How much internet cost does the company reimburse each month?", [r"1,?000"], [(REMOTE, None)]),
    (134, "answer", None, "How much notice is given before remote work eligibility is withdrawn?", [r"\b14\b|fourteen"], [(REMOTE, None)]),
    # needs two facts combined
    (135, "answer", None, "I joined 60 days ago. Can I work remotely 2 days a week?", [r"\bno\b|not|cannot|probation", r"90"], [(REMOTE, None)]),
    # must be refused: the documents do not say
    (136, "refuse", None, "What is the monthly premium for the health insurance?", [], []),
    (137, "refuse", None, "What is the maximum amount the vehicle policy pays out?", [], []),
    (138, "refuse", None, "What is the shipping fee for returning an item?", [], []),
    (139, "refuse", None, "How many days of leave do employees get for their own marriage?", [], []),
    (140, "refuse", None, "Who is the chief executive officer of Northwind?", [], []),
]

SETS = {"cbse": CBSE, "policies": POLICIES, "all": CBSE + POLICIES}
