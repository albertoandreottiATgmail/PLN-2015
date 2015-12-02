Exercise 1:

sents count: 17379
word vocabulary size: 46482
tag vocabulary size: 48

most frequent tags:
==================
[('nc', 92002), ('sp', 79904), ('da', 54552), ('vm', 50609), ('aq', 33904), ('fc', 30148), ('np', 29113), ('fp', 21157), ('rg', 15333), ('cc', 15023)]

% of total for most frequent tags: [('nc', 92002, 17.786137940100684), ('sp', 79904, 15.447311645027337), ('da', 54552, 10.546177223412236), ('vm', 50609, 9.783903121785999), ('aq', 33904, 6.554435998360618), ('fc', 30148, 5.828313369471918), ('np', 29113, 5.628223667421917), ('fp', 21157, 4.0901428273158205), ('rg', 15333, 2.964227441094365), ('cc', 15023, 2.9042971921711764)]

Top words in most frequent tags:
================================
nc [('años', 849), ('presidente', 682), ('millones', 616), ('equipo', 457), ('partido', 438)]
sp [('de', 28475), ('en', 12114), ('a', 8192), ('del', 6518), ('con', 4150)]
da [('la', 17897), ('el', 14524), ('los', 7758), ('las', 4882), ('El', 2817)]
vm [('está', 564), ('tiene', 511), ('dijo', 499), ('puede', 381), ('hace', 350)]
aq [('pasado', 393), ('gran', 275), ('mayor', 248), ('nuevo', 234), ('próximo', 213)]
fc [(',', 30148)]
np [('Gobierno', 554), ('España', 380), ('PP', 234), ('Barcelona', 232), ('Madrid', 196)]
fp [('.', 17513), ('(', 1823), (')', 1821)]
rg [('más', 1707), ('hoy', 772), ('también', 683), ('ayer', 593), ('ya', 544)]
cc [('y', 11211), ('pero', 938), ('o', 895), ('Pero', 323), ('e', 310)]

Ambiguity levels
================
level:  2
# of words:  2194 0.047201067079729785
top words in level [('la', 18100), ('y', 11212), ('"', 9296), ('los', 7824), ('del', 6519)]
level:  3
# of words:  153 0.0032915967471279207
top words in level [('.', 17520), ('a', 8200), ('un', 5198), ('no', 3300), ('es', 2315)]
level:  4
# of words:  19 0.0004087603803622908
top words in level [('de', 28478), ('dos', 917), ('este', 830), ('tres', 425), ('todo', 393)]
level:  5
# of words:  4 8.6054816918377e-05
top words in level [('que', 15391), ('mismo', 247), ('cinco', 224), ('medio', 105)]
level:  6
# of words:  3 6.454111268878275e-05
top words in level [('una', 3852), ('como', 1736), ('uno', 335)]
level:  7
# of words:  0 0.0
top words in level []
level:  8
# of words:  0 0.0
top words in level []
level:  9

Exercise 3:

Accuracy: 89.03%
Accuracy known: 95.35%
Accuracy unknown: 31.80%

Check the matrix plot in tagging/baseline_confusion.png

Exercise 5:
N	Accuracy,		known,		unknown
2 	92.33%			97.40%		46.36%	
3  93.05% 			97.47% 		53.06%
4  92.37%         96.66%      53.47%



Exercise 7:

maxent

N	Accuracy,		known,		unknown
1	89.94%			94.31%		50.32%
2	91.33%			94.12%		66.04%
3  91.89%         94.65%		66.86%
4  92.00%         94.73%		67.23%	

svm

N  Accuracy,      known,      unknown
1  90.91%         95.33%      50.79%
2  93.61%         96.64%      66.17%
3  93.93%         96.83%      67.57%
4  94.01%         96.85%      68.24%   

multinomial

N  Accuracy,      known,      unknown
1  88.86%         93.31%      48.50%
2  72.06%         74.65%      48.51%
3  72.70%         75.19%      50.13%
4  68.03%         70.42%      46.35%   



