from word2number import w2n

sentence_dict = {
    'M: Excuse me, the micle is in the right way for a few far?': 'M: Excuse me, is this the right way to "hill farm"?',
    "Walk him to I fill Farm.": 'Walk him to "Hill Farm".',
    "In an of fice.": "In an office.",
     "M: It was an exc iting football match. France against Germany. When the France were about to win... "
     "So, I got up late and didn't have breakfast and left home in a hurry.":  "M: It was an exciting football match. "
                                                                               "France against Germany. When the "
                                                                               "French were about to win... "
                                                                               "So, I got up late and didn't have "
                                                                               "breakfast and left home in a hurry.",
    "Man: Well, you know, it ... it kind of irks me it when hotels nickel-and-dime their customers like this. "
    "I mean, I checked with sev(eral hotels) ... I mean I checked with sev(eral hotels) ...": " Man: Well, you know, "
                                                                                              "it ... it kind of irks "
                                                                                              "me it when hotels "
                                                                                              "nickel-and-dime their "
                                                                                              "customers like this. "
                                                                                              "I mean, I checked with "
                                                                                              "several hotels ... I "
                                                                                              "mean I checked with "
                                                                                              "several hotels ...",
    "Where does this conve rsation take place?": "'Where does this conversation take place?",
    "W: Well, to tell the truth I don't car emuch dessert.": " W: Well, to tell the truth I don't care much for "
                                                             "dessert.",
    "W: I forget to tell you that the couple Simth won't come.": "W: I forget to tell you that the Smith couple "
                                                              "won't be coming.",
    "Where does the dialogue most probab ly take place?": "Where does the dialogue most probably take place?",
    'Why is the science exam diff icult?': 'Why is the science exam difficult?',
    'What is the propable relationship between the speakers?': 'What is the relationship between the speakers?',
    "What does the woman imply(ltd')about Kelly?": "What does the woman imply about Kelly?",
    "W: Excuse me, I have to get to the station before 11'o clock.": "W: Excuse me, I have to get to the station before 11 o'clock.",
    "Can the woman arrive at the airport before 11'o clock?": "Can the woman arrive at the airport before 11 o'clock?",
    "Mathew: I do notice some differences-er-I think-ah-I think the younger people in Britain are-they seem to "
    "be-much more radical than the younger people in the United States. I noticed that. Ah-the dress is different. "
    "You see a lot of-I see a lot of males here with earring in one of their-in one of their ears. You don't see that "
    "in America that much. Somet-maybe here and there, but not, not like you see it here. "
    "Ah-so many of the young people wear black-clothing-you know, I don't-you don't see the other colors. "
    "At home you see all different types of bright colors-and in England you see so much black. Especially on the "
    "women.": "Mathew: I do notice some differences -er-I think-ah-I think the younger people in Britain are-they seem"
              " to be-much more radical than the younger people in the United States. I noticed that. Ah-the dress is "
              "different. You see a lot of-I see a lot of males here with earring in one of their-in one of their "
              "ears. You don't see that in America that much. Maybe here and there, but not, not like you see it here. "
              "Ah-so many of the young people wear black-clothing-you know, I don't-you don't see the other colors. "
              "At home you see all different types of bright colors-and in England you see so much black. Especially "
              "on the women.",
    "Woman: Figure it out. Listen. I ('ve) got to go now. If you want to talk more, I'll be at mom's house.": "Woman: "
                "Figure it out. Listen. I've got to go now. If you want to talk more, I'll be at mom's house.",
    "Julie: No. I've only been in a boat once. I sailed down the River Thames on a sightseeing tour. But in many "
    "eases I'd rather to be sea - sick than dead.": "Julie: No. I've only been in a boat once. "
                                                    "I sailed down the River Thames on a sightseeing tour. But in "
                                                    "any case I'd rather be seasick than dead.",
    "W: No. I've only been in a boat once. I sailed down the River Thames on a sightseeing tour, "
    "but in any case I'd rather be sea - sick than dead.": "W: No. I've only been in a boat once. "
                                                           "I sailed down the River Thames on a sightseeing tour, "
                                                           "but in any case I'd rather be seasick than dead.",
"W: It is against the law in England to go into a pub if you are under the age of l4. So many pubs provide a special "
"room for children.": "W: It is against the law in England to go into a pub if you are under the age of 14. So many "
                      "pubs provide a special room for children.",




}

spacing_dict = {
    "can't": "cannot",
    'tothe': 'to the',
    'mayfail': 'may fail',
    'kungfu': 'kung fu',
    'thesame': 'the same',
    'whdavis': 'w . h . davis',
    'gatenine': 'gate nine',
    'youlend': 'you lend',
    "parisin": "paris in",
    'inever': 'i never',
    'hotdogs': 'hot dogs',
    'bymistake': 'by mistake',
    "konghotel": "kong hotel",
    'istill': 'i still',
    '32t': '32 degrees',
    'begood': 'be good',
    'mynew': 'my new',
    'libtary': 'library',
    'suitcasefor': 'suitcase for',
    'inthe': 'in the',
    "davis'class": "davis 's class",
    "3o'clock": "3 o'clock",
    'twomonths': 'two months',
    'managerhere': 'manager here',
    'yourown': 'your own',
    'myown': 'my own',
    'isslower': 'is slower',
    'nextsemester': 'next semester',
    'couldyou': 'could you',
    'betterroads': 'better roads',
    'thelevel': 'the level',
    'theregion': 'the region',
    'ahistory': 'a history',
    'spendthe': 'spend the',
    'whyis': 'why is',
    'hishomework': 'his homework',
    'firstjob': 'first job',
    'twobedroom': 'two bedroom',
    'thestop': 'the stop',
    'ofschool': 'of school',
    'ticketsfor': 'tickets for',
    'dragonprogram': 'dragon program',
    'greensguest': 'greens guest',
    'baskguest': 'bask guest',
    'aminute': 'a minute',
    'somenews': 'some news',
    'werein': 'were in',
    'muchclothes': 'much clothes',
    'gotfor': 'got for',
    'untilthursday': 'until thursday',
    'thecharges': 'the charges',
    'rightafter': 'right after',
    'tomove': 'to move',
    'comfortableand': 'comfortable and',
    'dinnertonight': 'dinner tonight',
    'highschool': 'high school',
    'neverregretted': 'never regretted',
    'andtry': 'and try',
    'sorryto': 'sorry to',
    'ofmodern': 'of modern',
    'tobuy': 'to buy',
    'intown': 'in town',
    'collegenext': 'college next',
    'theirschool': 'their school',
    'tostay': 'to stay',
    'haha': 'ha ha',
    'verydull': 'very dull',
    'toknow': 'to know',
    "hongkong": "hong kong",
    "kingsize": "king size",
    "samecover": "same cover",
    "n't": ' not',
    "tvsets": "tv sets",
    "wednesdaymeeting": 'wednesday meeting',
    "abouthis": 'about his',
    "withthe": 'with the',
    "forit": 'for it',
    "tocome": 'to come',
    "nobath": 'no bath',
    "stationinformation": 'station information',
    "fiftypercent": 'fifty percent',
    "idon't": "i do not",
    "hearthat": 'hear that',
    "newjob": 'new job',
    "sohot": 'so hot',
    "theshelf": 'the shelf',
    "yorkas": "york as",
    "meetagain": 'meet again',
    "offto": 'off to',
    "worldclassics": "world classics",
    "meduring": "me during",
    "tosketch": 'to sketch',
    "gothere": 'go there',
    "supersocial": 'super social',
    "youdecideon": 'you decide on',
    "bestwriters": "best writers",
    "inspiredby": 'inspired by',
    "dothink": 'do think',
    "matterwhich": 'matter which',
    "yousmoke": 'you smoke',
    "booksection": 'book section',
    "fromthe": 'from the',
    "wellas": 'well as',
    "thankyou": 'thank you',
    "haveto": 'have to',
    "homein": 'home in',
    "atweekends": 'on weekends',
    "midtermexam": 'midterm exam',
}

symbols = {
    ".'": " . ",
    '.': ' . ',
    "'?": " ? ",
    '?': ' ? ',
    '!': ' ! ',
    '–': '-',
    '~': ' ~ ',
    '×': ' by ',
    '%': ' % ',
    '°': ' degrees ',
    '$': ' $ ',
    '£': ' £ ',
    '@': ' @ ',
    '"': ' " ',
    '(': ' ( ',
    ')': ' ) ',
    '[': ' [ ',
    ']': ' ] ',
    ':': ' : ',
    ';': ' ; ',
    ',': ' , ',
}

units = {
    r"\d{2,}s-\d{2,}s": "s",
    r"\d{2,}s": "s",
    r"([A-Za-z]+-){0,1}\d+st": "st",
    r"([A-Za-z]+-){0,1}\d+nd": "nd",
    r"([A-Za-z]+-){0,1}\d+rd": "rd",
    r"([A-Za-z]+-){0,1}\d+th": "th",
    r"\d+s": " s",
    r"\d+mph": "mph",
    r"\d+pm": "pm",
    r"\d+kg": "kg",
    r"\d+km": "km",
    r"\d+p": "p",
    r"\d+era": "era",
    r"\d(,)*\d+rmb": "rmb",
    r"[A-Za-z]+(_+)": "_",
    r"(_+)[A-Za-z]+": "_",
}

onomatopoeia = {
    r"s(o+)": "so",
    r"ug(h+)": "ugh",
    r"a+gh": "agh",
    r"(a+)(h+)": "ah",
    r"(u+)(h+)": "uh",
    r"(h+)(m+)": "hm",
    r"hu(m+)": "hum",
    r"m(m+)": "mm",
    r"(o+)(h+)": "oh",
    r"u(m+)": "um",
    r"b(r+)": "brr",
    r"(_+)": "_"
}

abbrivations = {
    "didn't": "did not",
    "don't": "do not",
    "doesn't": "does not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "here's": "here is",
    "he's": "he is",
    "'n'": "and",
    "we're": "we are",
    "'cause": "because",
    "13o": "do",
    "shouldn't": "should not",
    "that'd": "that would",
    "that'll": "that will",
    "what'll": "what will",
    "there'd": "there would",
    "it'll": "it will",
    "it's": "it is",
    "isn't": "is not",
    "i'll": "i will",
    "she'll": "she will",
    "he'll": "he will",
    "you'll": "you will",
    "would've": "would have",
    "they'll": "they will",
    "we'll": "we will",
    "let's": "let us",
    "couldn't": "could not",
    "that's": "that is",
    "what've": "what have",
    "there'll": "there will",
    "mustn't": "must not",
    "could've": "could have",
    "ain't": "are not",
    "where're": "where are",
    "there's": "there is",
    "can't": "cannot",
    "aren't": "are not",
    "needn't": "need not",
    "should've": "should have",
    "'em": "them",
    "'bout": "about",
    "what're": "what are",
    "rock'n'roll": "rock and roll",
    "haven't": "have not",
    "hasn't": "has not",
    "wouldn't": "would not",
    "hadn't": "had not",
    "they're": "they are",
    "you're": "you are",
    "i'm": "i am",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would have",
    "you'd": "you would have",
    "he'd": "he would have",
    "she'd": "she would have",
    "it'd": "it would have",
    "they'd": "they would have",
    "we'd": "we would have",
    "where've": "where have",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "who's": "who is",
    "who'll": "who will",
    "how's": "how is",
}

spelling_regex = {
    r"disturbe": "disturbed",
    r"cant": " cannot",
    r"forme": 'for me',
    r"judg": 'judge',
    r"clini": 'clinic',
    r"fide": 'ride',
    r"machin": "machine",
    r"augh": "ugh",
    r"beens": "beans",
    r"lous": "louis",
    r"inhis": "in his",
    r"wock": "work",
    r"victori": "victoria",
    r"interviewe": 'interviewee',
    r"evenin": 'evening',
    r"buttercu": "buttercup",
    r"woinan's": "woman's",
    r"heve": "have",
    r"somet-maybe": "sometimes maybe",
    r"ican": 'i can',
    r"alway": "always",
    r'alrigh': 'alright',
}

spelling_dict = {
    "restau-rant": "restaurant",
    "joes": "does",
    "theatcr": "theater",
    "does't": "does not",
    "wash't": "was not",
    "volumn": "volume",
    "meanns": "means",
    "speakes": "speakers",
    "like'": "like",
    "number'": "number",
    "l've": "i have",
    "bag'": "bag",
    "red'": "red",
    "professer": "professor",
    "franch": "french",
    "nosiy": "noisy",
    "mter":"before",
    "about>": "about ?",
    "woinan": "woman",
    "accout": "account",
    "there're": "there are",
    "companuy": "company",
    "jetta": "jatta",
    "plice": "price",
    "moring": "morning",
    "aero-planes": "airplanes",
    "germen": "german",
    "dissert": "dessert",
    'octorber': 'october',
    "cornor": "corner",
    "ouhhhh": "ouch",
    "gsry": "gary",
    'boen': 'been',
    "prefes": "prefers",
    "weman": "woman",
    'cohunn': 'column',
    'yeap': 'yeah',
    'infered': 'inferred',
    'idition': 'edition',
    "inspaire": "inspire",
    'consulant': 'consultant',
    "bithday": "birthday",
    "yound": "young",
    "sceond": "second",
    "extremly": "extremely",
    "jime": "jaime",
    "receiner": "receiver",
    "tallking": "talking",
    "goege": "george",
    "transfering": "transferring",
    "raido": "radio",
    "examinadons": "examinations",
    "kindergarden": "kindergarten",
    "uptairs": "upstairs",
    "transfered": "transferred",
    "headeches": "headaches",
    "acutually": "actually",
    "writting": "writing",
    "nomally": "normally",
    "phwoo": "phew",
    "embarrashed": "embarrassed",
    "picinc": "picnic",
    "knid": "kind",
    "sustention": "suspension",
    "seience": "science",
    "wather": "water",
    "counry": "country",
    "flim": "film",
    "tallk": "talk",
    "aweful": "awful",
    "clueing": "cluing",
    "'custard": "custard",
    "routine'": "routine",
    "'where": "where",
    "'detachment'": "detachment",
    "dont": "do not",
    "tenis": "tennis",
    "airpotrs": "airports",
    "gardon": "garden",
    "hord": "lord",
    "wendesay": "wednesday",
    "throwed": "threw",
    "godness": "goodness",
    "aftemoon": "afternoon",
    'firday': 'friday',
    'joinning': 'joining',
    'himeself': 'himself',
    "belive": "believe",
    "runing": "running",
    "is't": "is not",
    "exhange": "exchange",
    "worrrisome": "worrisome",
    "coversation": "conversation",
    "intersting": "interesting",
    "yoghurt": "yogurt",
    "alawys": "always",
    "casuality": "causality",
    "begining": "beginning",
    "pr6fessor": "professor",
    "canda": "canada",
    "chicage": "chicago",
}


def split_up_digits(number):
    new_str = ""
    length_num = len(number)
    for char in number:
        if len(new_str) != 0:
            new_str += f" {char}"
        else:
            new_str = char
    return new_str


def replace_word(word_array, dict_of_words_to_replace):
    new_word_array = []
    for word in word_array:
        if word in dict_of_words_to_replace:
            new_word_array.extend(dict_of_words_to_replace[word])
        else:
            new_word_array.append(word)
    return new_word_array


def clean_word_array(word_array):
    new_word_array = word_array
    for word in word_array:
        if word in spacing_dict:
            new_words = spacing_dict[word].split()
            new_word_array = replace_word(new_word_array, {word: new_words})
        elif word in spelling_dict:
            new_words = [spelling_dict[word]]
            new_word_array = replace_word(new_word_array, {word: new_words})
        elif word.endswith("s'"):
            new_words = [word[:-1], "'s"]
            new_word_array = replace_word(new_word_array, {word: new_words})
        elif word in abbrivations:
            new_words = abbrivations[word].split()
            new_word_array = replace_word(new_word_array, {word: new_words})
        elif word.endswith("'s"):
            new_words = [clean_word_array([word[:-2]])[0], "'s"]
            new_word_array = replace_word(new_word_array, {word: new_words})
        elif word == "point":
            continue
        elif "-" in word:
            try:
                new_words = split_up_digits(str(w2n.word_to_num(word)))
                new_word_array = replace_word(new_word_array, {word: new_words})
            except ValueError:
                split_word = word.split("-")
                new_word_array = replace_word(new_word_array, {word: clean_word_array(split_word)})
        elif word.isnumeric():
            new_words = split_up_digits(word)
            new_word_array = replace_word(new_word_array, {word: new_words})
        else:
            try:
                new_words = split_up_digits(str(w2n.word_to_num(word)))
                new_word_array = replace_word(new_word_array, {word: new_words})
            except ValueError:
                continue
        if "'" in new_word_array:
            print(new_word_array)
    return new_word_array
