# 개선·실패 문항의 최초 결정 분기

기존새40×2seed실행기록80쌍을모두분석. 추가LLM실행/재학습없음. 질문·도구이름·인자·관측이같은prefix에서처음다른action을찾고저장된실제head확률을비교한다. 인자/도구결과가먼저다르면같은state행동대조로분류하지않는다. 확률비의log차이는두행동상대logit변화이며정답확률이아니다.

| seed | 최초 변화 | 개수 | 개선 | 악화 | 동률 |
|---|---|---:|---:|---:|---:|
| 1 | read_chunk -> semantic_search | 17 | 6 | 3 | 8 |
| 1 | no_public_action_or_tool_difference | 14 | 0 | 0 | 14 |
| 1 | read_chunk -> keyword_search | 3 | 2 | 0 | 1 |
| 1 | read_chunk -> FINAL | 5 | 0 | 0 | 5 |
| 1 | keyword_search -> semantic_search | 1 | 0 | 0 | 1 |
| 2 | read_chunk -> keyword_search | 5 | 3 | 0 | 2 |
| 2 | no_public_action_or_tool_difference | 22 | 0 | 0 | 22 |
| 2 | read_chunk -> FINAL | 6 | 0 | 0 | 6 |
| 2 | FINAL -> read_chunk | 1 | 1 | 0 | 0 |
| 2 | read_chunk -> semantic_search | 6 | 1 | 1 | 4 |

## 사례: 개선·악화 각 최대3문항

효과절대값순·qid순으로선정한설명용사례이며대표성/인과효과증거가아니다. 모든80쌍은JSON에보존.

### 2hop__673655_316668 / seed1

질문: What instrument is played by the person from The Blackout All-Stars?
정답: conga
Anchor: The text does not specify the instrument played by any member of The Blackout All-Stars. (F1 0.000)
PPO: conga (F1 1.000)
정답근거chunk검색: anchor=['p12'], PPO=['p12', 'p16'], 새근거=['p16']
최초분기 turn1: read_chunk→semantic_search; new/old logodds변화 +1.856.
두행동확률 anchor(old,new)=(0.409,0.368), PPO=(0.113,0.652).
직전관측: {"tool_name": "keyword_search", "arguments": {"keywords": ["The Blackout All-Stars", "instrument"], "top_k": 5}, "result": [{"chunk_id": "p19", "title": "A cappella", "snippet": "In addition to singing words, some a cappella singers also emulate instrumentation by reproducing instrumental sounds with their vocal cords and mouth. One of the earliest 20th century practitioners of this method were The Mills Brothers whose early recordings of the 1930s clearly stated on the labe"}, {"chunk_id": "p8", "title": "Miami Vice Theme", "snippet": "``Miami Vice Theme ''is a musical piece composed and performed by Jan Hammer as the theme to the television series Miami Vice. It was first presented as part of the television broadcast of the show in September 1984, was released as a single in 1985, and peaked at the number one spot on the Billboar"}, {"chunk_id": "p12", "title": "The Blackout All-Stars", "snippet": "The Blackout All-Stars was a one-off 1994 Latin supergroup consisting of Ray Barretto, Sheila E., Tito Puente, Tito Nieves, Paquito D'Rivera, Dave Valentin, Grover Washington, Jr., and Tony Pabon."}, {"chunk_id": "p11", "title": "Don't Worry, Be Happy", "snippet": "``Do n't Worry, Be H
Anchor후속도구: {"tool_name": "read_chunk", "arguments": {"chunk_ids": ["p12"]}, "result": [{"chunk_id": "p12", "title": "The Blackout All-Stars", "text": "The Blackout All-Stars was a one-off 1994 Latin supergroup consisting of Ray Barretto, Sheila E., Tito Puente, Tito Nieves, Paquito D'Rivera, Dave Valentin, Grover Washington, Jr., and Tony Pabon."}]}
PPO후속도구: {"tool_name": "semantic_search", "arguments": {"query": "What instrument does a member of The Blackout All-Stars play?", "top_k": 3}, "result": [{"chunk_id": "p12", "title": "The Blackout All-Stars", "snippet": "The Blackout All-Stars was a one-off 1994 Latin supergroup consisting of Ray Barretto, Sheila E., Tito Puente, Tito Nieves, Paquito D'Rivera, Dave Valentin, Grover Washington, Jr., and Tony Pabon."}, {"chunk_id": "p0", "title": "Radar (song)", "snippet": "\"Radar\" is a song recorded by American singer Britney Spears for her fifth studio album, \"Blackout\" (2007). It was written and produced by Bloodshy & Avant and The Clutch, with additional writing from Henrik Jonback, as a record that did not relate to any of her personal problems at the time. The re"}, {"chunk_id": "p5", "title": "Chapman Stick", "snippet": "The Chapman Stick is an electric musical instrument devised by Emmett Chapman in the early 1970s. A member of the guitar family, the Chapman Stick usually has ten or twelve individually tuned strings and has been used on music recordings to play bass lines, melody lines, chords, or textures. Designe"}]}

### 2hop__73615_20510 / seed1

질문: When was the company who abolished the dual system of government in bengal founded?
정답: 1600
Anchor: 1773 (F1 0.000)
PPO: 1600 (F1 1.000)
정답근거chunk검색: anchor=['p7'], PPO=['p0', 'p7'], 새근거=['p0']
최초분기 turn2: read_chunk→semantic_search; new/old logodds변화 +1.299.
두행동확률 anchor(old,new)=(0.568,0.219), PPO=(0.279,0.394).
직전관측: {"tool_name": "semantic_search", "arguments": {"query": "company responsible for abolishing the dual system of government in Bengal and its founding date", "top_k": 5}, "result": [{"chunk_id": "p7", "title": "Nawabs of Bengal and Murshidabad", "snippet": "The break - up of the centralised Mughal empire by 1750, led to the creation of numerous semi-independent kingdoms (all provinces of the former Mughal empire). Nawab Siraj ud - Daulah was defeated by the British forces of Sir Robert Clive in the Battle of Plassey in 1757. Thereafter the Nawab of Ben"}, {"chunk_id": "p16", "title": "Permanent Settlement", "snippet": "The Permanent Settlement was introduced first in Bengal and Bihar, and then in the south district of Madras and Varanasi. The system eventually spread all over northern India by a series of regulations dated 1 May 1793. These regulations remained in place until the Charter Act of 1833. The other two"}, {"chunk_id": "p3", "title": "History of India", "snippet": "During this period, Bengal witnessed an intellectual awakening that is in some way similar to the Renaissance in Europe during the 16th century, although Europeans of that age were not confronted with the challe
Anchor후속도구: {"tool_name": "read_chunk", "arguments": {"chunk_ids": ["p7"]}, "result": [{"chunk_id": "p7", "title": "Nawabs of Bengal and Murshidabad", "text": "The break - up of the centralised Mughal empire by 1750, led to the creation of numerous semi-independent kingdoms (all provinces of the former Mughal empire). Nawab Siraj ud - Daulah was defeated by the British forces of Sir Robert Clive in the Battle of Plassey in 1757. Thereafter the Nawab of Bengal became a ``puppet ruler ''depending on military support from British East India company to secure their throne. Siraj - ud - Daulah was replaced by Mir Jaffer. He was personally led to the throne by Robert Clive, after triumph of the British in the battle. He briefly tried to re-assert his power by allying with the Dutch, but this plan was ended by the Battle of Chinsurah. After the defeat at Battle of Buxar and grant of the Diwani (revenue collection) of Bengal by the then Mughal Emperor Shah Alam II, to the British East India Company in August 1765 and the appointment of Warren Hastings by the East India Company as their first Governor General of Bengal in 1773, the Nawabs authority became restricted. By 1773, British East India company
PPO후속도구: {"tool_name": "semantic_search", "arguments": {"query": "founding date of the company that abolished the dual system of government in Bengal", "top_k": 5}, "result": [{"chunk_id": "p7", "title": "Nawabs of Bengal and Murshidabad", "snippet": "The break - up of the centralised Mughal empire by 1750, led to the creation of numerous semi-independent kingdoms (all provinces of the former Mughal empire). Nawab Siraj ud - Daulah was defeated by the British forces of Sir Robert Clive in the Battle of Plassey in 1757. Thereafter the Nawab of Ben"}, {"chunk_id": "p16", "title": "Permanent Settlement", "snippet": "The Permanent Settlement was introduced first in Bengal and Bihar, and then in the south district of Madras and Varanasi. The system eventually spread all over northern India by a series of regulations dated 1 May 1793. These regulations remained in place until the Charter Act of 1833. The other two"}, {"chunk_id": "p0", "title": "British Empire", "snippet": "At the end of the 16th century, England and the Netherlands began to challenge Portugal's monopoly of trade with Asia, forming private joint-stock companies to finance the voyages—the English, later British, East India Company

### 2hop__776377_857193 / seed1

질문: What is the mouth of watercourse for the body of water where Bartram's Covered Bridge is located?
정답: Delaware River
Anchor: Crum Creek (F1 0.000)
PPO: Delaware River (F1 1.000)
정답근거chunk검색: anchor=['p18'], PPO=['p18', 'p9'], 새근거=['p9']
최초분기 turn1: read_chunk→semantic_search; new/old logodds변화 +2.059.
두행동확률 anchor(old,new)=(0.507,0.207), PPO=(0.157,0.502).
직전관측: {"tool_name": "keyword_search", "arguments": {"keywords": ["Bartram's Covered Bridge", "body of water", "mouth of watercourse"], "top_k": 5}, "result": [{"chunk_id": "p18", "title": "Bartram's Covered Bridge", "snippet": "Bartram's Covered Bridge, a historic covered bridge built in 1860, uses a Burr Truss design and carried Goshen Road over Crum Creek on the border between Delaware County and Chester County, Pennsylvania. It is long and wide and is the only covered bridge remaining of the 30 which once stood in Delaw"}]}
Anchor후속도구: {"tool_name": "read_chunk", "arguments": {"chunk_ids": ["p18"]}, "result": [{"chunk_id": "p18", "title": "Bartram's Covered Bridge", "text": "Bartram's Covered Bridge, a historic covered bridge built in 1860, uses a Burr Truss design and carried Goshen Road over Crum Creek on the border between Delaware County and Chester County, Pennsylvania. It is long and wide and is the only covered bridge remaining of the 30 which once stood in Delaware County. The bridge has slanted planks at each entrance and is the only covered bridge in Pennsylvania with this feature. According to an on-site marker from the Newtown Square Historical Preservation Society, the bridge was built to be \"hi and wide as a load of hay\" It was built by Ferdinand Wood and named for Mordecai Bartram."}]}
PPO후속도구: {"tool_name": "semantic_search", "arguments": {"query": "what is the mouth of the watercourse that flows into the body of water where Bartram's Covered Bridge is located", "top_k": 5}, "result": [{"chunk_id": "p18", "title": "Bartram's Covered Bridge", "snippet": "Bartram's Covered Bridge, a historic covered bridge built in 1860, uses a Burr Truss design and carried Goshen Road over Crum Creek on the border between Delaware County and Chester County, Pennsylvania. It is long and wide and is the only covered bridge remaining of the 30 which once stood in Delaw"}, {"chunk_id": "p6", "title": "Burt Henry Covered Bridge", "snippet": "The Burt Henry Covered Bridge, also known as the Henry Covered Bridge or just the Henry Bridge, is a covered bridge that spans the Walloomsac River near Bennington, Vermont. A Town lattice truss bridge, it carries River Road, just south of the village of North Bennington. Originally built about 1840"}, {"chunk_id": "p13", "title": "Ledyard Bridge", "snippet": "The Ledyard Bridge crosses the Connecticut River to connect Hanover, New Hampshire to Norwich, Vermont. It is the third bridge at this crossing to bear the name of the adventurer John Ledyard."}, {"c

### 2hop__442516_595885 / seed1

질문: Which country has a body of water that was inspiration for the name of the Mara Region?
정답: Tanzania
Anchor: Tanzania (F1 1.000)
PPO: Kenya (F1 0.000)
정답근거chunk검색: anchor=[], PPO=['p14', 'p9'], 새근거=['p14', 'p9']
최초분기 turn1: read_chunk→semantic_search; new/old logodds변화 +1.976.
두행동확률 anchor(old,new)=(0.618,0.166), PPO=(0.227,0.440).
직전관측: {"tool_name": "keyword_search", "arguments": {"keywords": ["Mara Region", "body of water", "inspiration"], "top_k": 5}, "result": [{"chunk_id": "p12", "title": "Serengeti District", "snippet": "Serengeti District is one of the seven districts of Mara Region of Tanzania. Its administrative centre is the town of Mugumu. It is home to part of the world-famous Serengeti National Park and contains one of the western gates to the park."}]}
Anchor후속도구: {"tool_name": "read_chunk", "arguments": {"chunk_ids": ["p12"]}, "result": [{"chunk_id": "p12", "title": "Serengeti District", "text": "Serengeti District is one of the seven districts of Mara Region of Tanzania. Its administrative centre is the town of Mugumu. It is home to part of the world-famous Serengeti National Park and contains one of the western gates to the park."}]}
PPO후속도구: {"tool_name": "semantic_search", "arguments": {"query": "body of water that inspired the name Mara Region", "top_k": 5}, "result": [{"chunk_id": "p14", "title": "Mara Triangle", "snippet": "Divided from the rest of the Maasai Mara National Reserve by the Mara River, the Mara Triangle is less visited and less crowded, often with many more game animals grazing on the plains and between the volcanic hills that distinguish this corner of the Mara."}, {"chunk_id": "p8", "title": "Trans Mara District", "snippet": "Trans Mara District was an administrative district in the former Rift Valley Province of Kenya. Its capital town was Kilgoris. The district had a population of 170,591 (1999 census) and an area of 2,846 km² . It was created in 1994, when Trans Mara District was split from Narok District."}, {"chunk_id": "p6", "title": "Ann Mara", "snippet": "Ann Mara (June 18, 1929 – February 1, 2015) was an American businesswoman, socialite, philanthropist, the wife and later widow of Wellington Mara, and the matriarch of the Mara family, which includes New York Giants CEO John Mara, and her granddaughters, actresses Rooney Mara and Kate Mara."}, {"chunk_id": "p9", "title": "Kirumi Bridge", "s

### 2hop__6744_6731 / seed1

질문: What two dialects are included in the other pattern other vowels follow?
정답: Northwestern Catalan and Valencian
Anchor: Northwestern Catalan and Valencian (F1 1.000)
PPO: Western Catalan and Central Catalan (F1 0.444)
정답근거chunk검색: anchor=['p1'], PPO=['p1', 'p14'], 새근거=['p14']
최초분기 turn1: read_chunk→semantic_search; new/old logodds변화 +1.576.
두행동확률 anchor(old,new)=(0.456,0.326), PPO=(0.166,0.574).
직전관측: {"tool_name": "keyword_search", "arguments": {"keywords": ["other pattern", "vowels", "dialects"], "top_k": 5}, "result": [{"chunk_id": "p7", "title": "Russian language", "snippet": "The language possesses five vowels (or six, under the St. Petersburg Phonological School), which are written with different letters depending on whether or not the preceding consonant is palatalized. The consonants typically come in plain vs. palatalized pairs, which are traditionally called hard an"}, {"chunk_id": "p1", "title": "Catalan language", "snippet": "Western Catalan comprises the two dialects of Northwestern Catalan and Valencian; the Eastern block comprises four dialects: Central Catalan, Balearic, Rossellonese, and Alguerese. Each dialect can be further subdivided in several subdialects."}, {"chunk_id": "p3", "title": "Catalan language", "snippet": "In Central Catalan, unstressed vowels reduce to three: /a e ɛ/ > [ə]; /o ɔ u/ > [u]; /i/ remains distinct. The other dialects have different vowel reduction processes (see the section pronunciation of dialects in this article)."}, {"chunk_id": "p0", "title": "Hokkien", "snippet": "Hokkien dialects are analytic; in a sentence, the arrangement of
Anchor후속도구: {"tool_name": "read_chunk", "arguments": {"chunk_ids": ["p1"]}, "result": [{"chunk_id": "p1", "title": "Catalan language", "text": "Western Catalan comprises the two dialects of Northwestern Catalan and Valencian; the Eastern block comprises four dialects: Central Catalan, Balearic, Rossellonese, and Alguerese. Each dialect can be further subdivided in several subdialects."}]}
PPO후속도구: {"tool_name": "semantic_search", "arguments": {"query": "dialects that follow a specific vowel pattern in Catalan language", "top_k": 3}, "result": [{"chunk_id": "p4", "title": "Catalan language", "snippet": "Catalan has inherited the typical vowel system of Vulgar Latin, with seven stressed phonemes: /a ɛ e i ɔ o u/, a common feature in Western Romance, except Spanish. Balearic has also instances of stressed /ə/. Dialects differ in the different degrees of vowel reduction, and the incidence of the pair "}, {"chunk_id": "p3", "title": "Catalan language", "snippet": "In Central Catalan, unstressed vowels reduce to three: /a e ɛ/ > [ə]; /o ɔ u/ > [u]; /i/ remains distinct. The other dialects have different vowel reduction processes (see the section pronunciation of dialects in this article)."}, {"chunk_id": "p1", "title": "Catalan language", "snippet": "Western Catalan comprises the two dialects of Northwestern Catalan and Valencian; the Eastern block comprises four dialects: Central Catalan, Balearic, Rossellonese, and Alguerese. Each dialect can be further subdivided in several subdialects."}]}

### 2hop__238499_431914 / seed1

질문: What administrative territorial entity includes the arrondissement that contains the Digne Cathedral?
정답: Alpes-de-Haute-Provence
Anchor: Alpes-de-Haute-Provence (F1 1.000)
PPO: Alpes-de-Haute-Provence department (F1 0.667)
정답근거chunk검색: anchor=['p12', 'p4'], PPO=['p12', 'p4'], 새근거=[]
최초분기 turn1: read_chunk→semantic_search; new/old logodds변화 +1.696.
두행동확률 anchor(old,new)=(0.411,0.262), PPO=(0.136,0.472).
직전관측: {"tool_name": "keyword_search", "arguments": {"keywords": ["Digne Cathedral", "arrondissement"], "top_k": 5}, "result": [{"chunk_id": "p14", "title": "Arrondissement of Veurne", "snippet": "The Arrondissement of Veurne (; ) is one of the eight administrative arrondissements in the Province of West Flanders, Belgium. It is both an administrative and a judicial arrondissement. However, the Judicial Arrondissement of Veurne also comprises all municipalities in the Arrondissement of Diksmu"}, {"chunk_id": "p4", "title": "Arrondissement of Digne-les-Bains", "snippet": "The arrondissement of Digne-les-Bains is an arrondissement of France in the Alpes-de-Haute-Provence department in the Provence-Alpes-Côte d'Azur region. Since the January 2017 reorganization of the arrondissements of Alpes-de-Haute-Provence, it has had 46 communes."}, {"chunk_id": "p12", "title": "Digne Cathedral", "snippet": "Digne Cathedral () is a Roman Catholic church located in the town of Digne-les-Bains, France. The cathedral has been a \"monument historique\" since 1906. It is the seat of the Bishops of Digne, Riez and Sisteron, formerly Bishops of Digne."}, {"chunk_id": "p1", "title": "Canton of Wormhout", "snipp
Anchor후속도구: {"tool_name": "read_chunk", "arguments": {"chunk_ids": ["p4"]}, "result": [{"chunk_id": "p4", "title": "Arrondissement of Digne-les-Bains", "text": "The arrondissement of Digne-les-Bains is an arrondissement of France in the Alpes-de-Haute-Provence department in the Provence-Alpes-Côte d'Azur region. Since the January 2017 reorganization of the arrondissements of Alpes-de-Haute-Provence, it has had 46 communes."}]}
PPO후속도구: {"tool_name": "semantic_search", "arguments": {"query": "administrative territorial entity that includes the arrondissement of Digne-les-Bains", "top_k": 3}, "result": [{"chunk_id": "p4", "title": "Arrondissement of Digne-les-Bains", "snippet": "The arrondissement of Digne-les-Bains is an arrondissement of France in the Alpes-de-Haute-Provence department in the Provence-Alpes-Côte d'Azur region. Since the January 2017 reorganization of the arrondissements of Alpes-de-Haute-Provence, it has had 46 communes."}, {"chunk_id": "p12", "title": "Digne Cathedral", "snippet": "Digne Cathedral () is a Roman Catholic church located in the town of Digne-les-Bains, France. The cathedral has been a \"monument historique\" since 1906. It is the seat of the Bishops of Digne, Riez and Sisteron, formerly Bishops of Digne."}, {"chunk_id": "p11", "title": "Canton of Villefranche-sur-Mer", "snippet": "The Canton of Villefranche-sur-Mer is a French former administrative division, located in the arrondissement of Nice, in the Alpes-Maritimes \"département\" (Provence-Alpes-Côte d'Azur \"région\"). It was disbanded following the French canton reorganisation which came into effect in March 2015. The comm"}

## 해석 제한

Goldsupport chunk가검색됐다는것은snippet에서필요한근거를모두읽고이해했다는뜻이아니다. 문서ID기반coverage와실제entailment를구분한다.

분기와최종답변차이는관련경로를보여주지만첫action하나의인과효과는아니다. 이후모든action·인자·관측이함께달라진다. 답변F1은문자열overlap지표여서표현/길이변화도점수변화에영향을준다. 개선사례만으로이해를주장하지않고같은전환이실패한사례도함께본다. 이자료를진행중PPO96의학습질문선정이나target에사용하지않음.

## 사례 해석

- Bartram bridge:다리가걸친Crum Creek까지만찾고답한anchor와달리PPO는후속검색에서Crum Creek의하구근거p9를찾아Delaware River로답했다. 첫전환만으로직접정답근거가나온것은아니고두번째semantic검색까지이어졌다.
- Bengal:anchor는1773이라는다른사건연도를답했고,PPO추가검색에British Empire(p0)가포함되며창립연도1600을답했다.
- 악기:첫semantic검색만으로해결되지않았고구성원들을나열한후속검색에서새정답근거p16이검색되어conga답변으로이어졌다.
- 실패:Mara사례는정답근거문서도검색했지만다른지역문서를인용하며Tanzania→Kenya로악화. Catalan사례도근거를더찾았지만이미보인정답을다른분류와혼동했다. 추가근거확보가충분조건은아님.
- Alpes-de-Haute-Provence에department를붙인답은F1이1→.667로하락했다. 적어도이사례는새사실오류라기보다표현길이에민감한채점의영향이다.

새goldsupport가검색된개선사례는seed1 8개중6개,seed2 5개중2개. 악화사례에서도seed1 3개중2개,seed2 1개중1개에서새goldsupport가검색됨. 따라서검색coverage와올바른해석을구분해야한다.
